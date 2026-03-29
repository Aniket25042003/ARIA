import asyncio
import json
from datetime import datetime, timedelta, timezone

import google.generativeai as genai
from fastapi import APIRouter, Depends
from sqlalchemy import func, select, case, distinct
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.dependencies import get_current_user, get_db
from app.models.postgres.api_usage import APIUsage
from app.models.postgres.navigation_log import NavigationLog
from app.models.postgres.sos_event import SOSEvent
from app.models.postgres.transcript import SignSession, TranscriptEntry
from app.models.postgres.user import User
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/summary")
async def dashboard_summary(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """High-level usage summary for the current user."""
    uid = user.id

    # Sign sessions count + total transcript entries
    sign_sessions = await db.scalar(
        select(func.count()).select_from(SignSession).where(SignSession.user_id == uid)
    )
    transcript_entries = await db.scalar(
        select(func.count())
        .select_from(TranscriptEntry)
        .join(SignSession, TranscriptEntry.session_id == SignSession.id)
        .where(SignSession.user_id == uid)
    )

    # Navigation count + completed
    nav_total = await db.scalar(
        select(func.count()).select_from(NavigationLog).where(NavigationLog.user_id == uid)
    )
    nav_completed = await db.scalar(
        select(func.count())
        .select_from(NavigationLog)
        .where(NavigationLog.user_id == uid, NavigationLog.ended_at.isnot(None))
    )

    # SOS events
    sos_count = await db.scalar(
        select(func.count()).select_from(SOSEvent).where(SOSEvent.user_id == uid)
    )

    return {
        "user_name": user.name,
        "sign_mode": {
            "total_sessions": sign_sessions or 0,
            "total_sentences": transcript_entries or 0,
        },
        "guide_mode": {
            "total_navigations": nav_total or 0,
            "completed_navigations": nav_completed or 0,
        },
        "sos_events": sos_count or 0,
    }


@router.get("/sign-activity")
async def sign_activity(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Detailed SIGN mode activity: emotions used, languages, recent sentences."""
    uid = user.id

    # Emotion distribution across all transcript entries
    emotion_rows = (
        await db.execute(
            select(TranscriptEntry.emotion, func.count().label("count"))
            .join(SignSession, TranscriptEntry.session_id == SignSession.id)
            .where(SignSession.user_id == uid)
            .group_by(TranscriptEntry.emotion)
            .order_by(func.count().desc())
        )
    ).all()
    emotion_distribution = {row.emotion: row.count for row in emotion_rows}

    # Language usage
    lang_rows = (
        await db.execute(
            select(TranscriptEntry.language, func.count().label("count"))
            .join(SignSession, TranscriptEntry.session_id == SignSession.id)
            .where(SignSession.user_id == uid)
            .group_by(TranscriptEntry.language)
            .order_by(func.count().desc())
        )
    ).all()
    language_usage = {row.language: row.count for row in lang_rows}

    # Recent sentences (last 20)
    recent_rows = (
        await db.execute(
            select(
                TranscriptEntry.text,
                TranscriptEntry.raw_letters,
                TranscriptEntry.emotion,
                TranscriptEntry.language,
                TranscriptEntry.created_at,
            )
            .join(SignSession, TranscriptEntry.session_id == SignSession.id)
            .where(SignSession.user_id == uid)
            .order_by(TranscriptEntry.created_at.desc())
            .limit(20)
        )
    ).all()
    recent_sentences = [
        {
            "text": r.text,
            "raw_letters": r.raw_letters,
            "emotion": r.emotion,
            "language": r.language,
            "time": r.created_at.isoformat() if r.created_at else None,
        }
        for r in recent_rows
    ]

    # Average session duration
    session_rows = (
        await db.execute(
            select(SignSession.started_at, SignSession.ended_at)
            .where(SignSession.user_id == uid, SignSession.ended_at.isnot(None))
        )
    ).all()
    if session_rows:
        durations = [(r.ended_at - r.started_at).total_seconds() for r in session_rows]
        avg_duration_s = sum(durations) / len(durations)
    else:
        avg_duration_s = 0

    return {
        "emotion_distribution": emotion_distribution,
        "language_usage": language_usage,
        "recent_sentences": recent_sentences,
        "avg_session_duration_seconds": round(avg_duration_s, 1),
    }


@router.get("/guide-activity")
async def guide_activity(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Detailed GUIDE mode activity: navigation history, obstacle stats."""
    uid = user.id

    # Recent navigations
    nav_rows = (
        await db.execute(
            select(NavigationLog)
            .where(NavigationLog.user_id == uid)
            .order_by(NavigationLog.started_at.desc())
            .limit(10)
        )
    ).scalars().all()
    navigations = [
        {
            "destination": n.destination,
            "total_steps": n.total_steps,
            "completed_steps": n.completed_steps,
            "completed": n.ended_at is not None,
            "started_at": n.started_at.isoformat() if n.started_at else None,
            "ended_at": n.ended_at.isoformat() if n.ended_at else None,
        }
        for n in nav_rows
    ]

    # Obstacle detection stats from APIUsage
    obstacle_total = await db.scalar(
        select(func.count())
        .select_from(APIUsage)
        .where(APIUsage.endpoint == "detect_obstacle")
    )
    obstacle_success = await db.scalar(
        select(func.count())
        .select_from(APIUsage)
        .where(APIUsage.endpoint == "detect_obstacle", APIUsage.success == True)
    )
    avg_latency = await db.scalar(
        select(func.avg(APIUsage.latency_ms))
        .where(APIUsage.endpoint == "detect_obstacle", APIUsage.success == True)
    )

    # Provider breakdown
    provider_rows = (
        await db.execute(
            select(APIUsage.provider, func.count().label("count"))
            .where(APIUsage.endpoint == "detect_obstacle")
            .group_by(APIUsage.provider)
        )
    ).all()
    provider_usage = {row.provider: row.count for row in provider_rows}

    return {
        "recent_navigations": navigations,
        "obstacle_detection": {
            "total_scans": obstacle_total or 0,
            "successful_scans": obstacle_success or 0,
            "avg_latency_ms": round(avg_latency, 1) if avg_latency else 0,
            "provider_breakdown": provider_usage,
        },
    }


@router.get("/api-performance")
async def api_performance(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """API performance metrics: latency, success rates, provider health over time."""

    # Per-endpoint stats
    endpoint_rows = (
        await db.execute(
            select(
                APIUsage.endpoint,
                func.count().label("total"),
                func.sum(case((APIUsage.success == True, 1), else_=0)).label("successes"),
                func.avg(APIUsage.latency_ms).label("avg_latency"),
                func.min(APIUsage.latency_ms).label("min_latency"),
                func.max(APIUsage.latency_ms).label("max_latency"),
            )
            .group_by(APIUsage.endpoint)
            .order_by(func.count().desc())
        )
    ).all()

    endpoints = [
        {
            "endpoint": r.endpoint,
            "total_calls": r.total,
            "success_rate": round((r.successes / r.total * 100) if r.total else 0, 1),
            "avg_latency_ms": round(r.avg_latency, 1) if r.avg_latency else 0,
            "min_latency_ms": r.min_latency or 0,
            "max_latency_ms": r.max_latency or 0,
        }
        for r in endpoint_rows
    ]

    # Per-provider stats
    provider_rows = (
        await db.execute(
            select(
                APIUsage.provider,
                func.count().label("total"),
                func.sum(case((APIUsage.success == True, 1), else_=0)).label("successes"),
                func.avg(APIUsage.latency_ms).label("avg_latency"),
            )
            .group_by(APIUsage.provider)
            .order_by(func.count().desc())
        )
    ).all()

    providers = [
        {
            "provider": r.provider,
            "total_calls": r.total,
            "success_rate": round((r.successes / r.total * 100) if r.total else 0, 1),
            "avg_latency_ms": round(r.avg_latency, 1) if r.avg_latency else 0,
        }
        for r in provider_rows
    ]

    # Recent errors (last 10)
    error_rows = (
        await db.execute(
            select(
                APIUsage.provider,
                APIUsage.endpoint,
                APIUsage.error_message,
                APIUsage.latency_ms,
                APIUsage.created_at,
            )
            .where(APIUsage.success == False)
            .order_by(APIUsage.created_at.desc())
            .limit(10)
        )
    ).all()

    recent_errors = [
        {
            "provider": r.provider,
            "endpoint": r.endpoint,
            "error": r.error_message,
            "latency_ms": r.latency_ms,
            "time": r.created_at.isoformat() if r.created_at else None,
        }
        for r in error_rows
    ]

    # Hourly call volume (last 24h)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    hourly_rows = (
        await db.execute(
            select(
                func.strftime("%H", APIUsage.created_at).label("hour"),
                func.count().label("count"),
            )
            .where(APIUsage.created_at >= cutoff)
            .group_by(func.strftime("%H", APIUsage.created_at))
            .order_by("hour")
        )
    ).all()
    hourly_volume = {r.hour: r.count for r in hourly_rows}

    return {
        "endpoints": endpoints,
        "providers": providers,
        "recent_errors": recent_errors,
        "hourly_volume_24h": hourly_volume,
    }


@router.get("/sos-history")
async def sos_history(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """SOS event history for the current user."""
    uid = user.id

    sos_rows = (
        await db.execute(
            select(SOSEvent)
            .where(SOSEvent.user_id == uid)
            .order_by(SOSEvent.triggered_at.desc())
            .limit(20)
        )
    ).scalars().all()

    return {
        "total": len(sos_rows),
        "events": [
            {
                "id": str(e.id),
                "latitude": e.latitude,
                "longitude": e.longitude,
                "triggered_at": e.triggered_at.isoformat() if e.triggered_at else None,
            }
            for e in sos_rows
        ],
    }


INSIGHTS_PROMPT = """You are an AI analytics assistant for ARIA, an accessibility app that helps deaf/mute people (SIGN mode) and blind people (GUIDE mode).

Analyze this user's usage data and generate a helpful, encouraging dashboard summary. Be concise and insightful.

User data:
{data}

Generate a JSON response with EXACTLY this structure (no markdown, no code fences, just raw JSON):
{{
  "greeting": "A short personalized greeting mentioning the user's name",
  "summary": "A 2-3 sentence overview of their overall usage patterns",
  "sign_insight": "One specific insight about their SIGN mode usage (emotions, languages, frequency). If no data, say they haven't tried it yet and encourage them.",
  "guide_insight": "One specific insight about their GUIDE mode usage. If no data, say they haven't tried it yet and encourage them.",
  "tip": "One actionable tip based on their usage patterns to get more out of ARIA",
  "streak_text": "A motivational line about their activity (e.g., 'You've translated X sentences this week!' or 'Keep going!')"
}}"""

# Models to try in order of preference (different SDK versions support different names)
GEMINI_MODEL_CANDIDATES = [
    "gemini-3-flash-preview",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-pro",
]


def _generate_local_insights(stats: dict) -> dict:
    """Generate contextual insights locally when Gemini is unavailable."""
    name = stats["user_name"]
    sign = stats["sign_mode"]
    guide = stats["guide_mode"]
    sos = stats["sos_events"]

    sessions = sign["total_sessions"]
    sentences = sign["total_sentences"]
    emotions = sign["emotion_distribution"]
    languages = sign["language_usage"]
    avg_dur = sign["avg_session_duration_seconds"]
    navs = guide["total_navigations"]
    scans = guide["total_obstacle_scans"]

    # Greeting
    greeting = f"Hi {name}! Here's your ARIA activity overview."

    # Summary
    total_actions = sessions + sentences + navs + scans
    if total_actions == 0:
        summary = (
            "You're just getting started with ARIA! "
            "Try out SIGN mode to translate your hand signs into speech, "
            "or GUIDE mode to detect obstacles with your camera."
        )
    else:
        parts = []
        if sessions > 0:
            parts.append(f"{sessions} sign session{'s' if sessions != 1 else ''}")
        if sentences > 0:
            parts.append(f"{sentences} translated sentence{'s' if sentences != 1 else ''}")
        if scans > 0:
            parts.append(f"{scans} obstacle scan{'s' if scans != 1 else ''}")
        summary = f"You've completed {', '.join(parts)}. "
        if sessions > 0 and navs > 0:
            summary += "You're actively using both SIGN and GUIDE modes — great versatility!"
        elif sessions > 0:
            summary += "You've been focused on SIGN mode. Try GUIDE mode for obstacle detection!"
        elif navs > 0 or scans > 0:
            summary += "You've been using GUIDE mode. Give SIGN mode a try for ASL translation!"
        else:
            summary += "Keep exploring to get the most out of ARIA."

    # Sign insight
    if sessions == 0 and sentences == 0:
        sign_insight = (
            "You haven't tried SIGN mode yet! "
            "Open the Sign tab, point your front camera at your hands, "
            "and start signing in ASL — ARIA will speak your words aloud."
        )
    else:
        sign_parts = []
        if emotions:
            top_emotion = max(emotions, key=emotions.get)
            sign_parts.append(
                f"your most common emotion is {top_emotion} "
                f"({emotions[top_emotion]} time{'s' if emotions[top_emotion] != 1 else ''})"
            )
        if languages:
            lang_count = len(languages)
            if lang_count > 1:
                sign_parts.append(
                    f"you've used {lang_count} different languages"
                )
            else:
                lang_name = list(languages.keys())[0]
                sign_parts.append(f"all in {lang_name.upper()}")
        if avg_dur > 0:
            if avg_dur > 60:
                sign_parts.append(
                    f"with average sessions of {round(avg_dur / 60, 1)} minutes"
                )
            else:
                sign_parts.append(
                    f"with average sessions of {round(avg_dur)}s"
                )
        sign_insight = (
            f"Across {sessions} session{'s' if sessions != 1 else ''} "
            f"and {sentences} sentence{'s' if sentences != 1 else ''}, "
            + ", ".join(sign_parts) + "."
            if sign_parts
            else f"You've had {sessions} session{'s' if sessions != 1 else ''} "
            f"with {sentences} sentence{'s' if sentences != 1 else ''}. Keep signing!"
        )

    # Guide insight
    if navs == 0 and scans == 0:
        guide_insight = (
            "You haven't tried GUIDE mode yet! "
            "Open the Guide tab, point your back camera ahead, "
            "and ARIA will warn you about obstacles in real-time."
        )
    else:
        parts = []
        if scans > 0:
            parts.append(f"{scans} obstacle scan{'s' if scans != 1 else ''} performed")
        if navs > 0:
            parts.append(f"{navs} navigation{'s' if navs != 1 else ''} started")
            completed = guide["completed_navigations"]
            if completed > 0:
                parts.append(f"{completed} completed")
        guide_insight = "GUIDE mode activity: " + ", ".join(parts) + "."

    # Tip
    if total_actions == 0:
        tip = "Start with SIGN mode — just point your camera at your hands and sign a letter to see ARIA in action!"
    elif sessions > 0 and len(languages) <= 1:
        tip = "Try switching languages in SIGN mode! ARIA supports English, Spanish, French, and Hindi."
    elif sessions > 0 and scans == 0:
        tip = "Give GUIDE mode a try! It uses your camera to detect obstacles and keep you safe while walking."
    elif scans > 0 and sessions == 0:
        tip = "Try SIGN mode to translate ASL into spoken words. It's great for communicating with people around you."
    elif sos > 0:
        tip = "Remember, the SOS button is always available in SIGN mode for emergencies. Stay safe!"
    else:
        tip = "Keep using ARIA regularly to build your communication confidence. Every session makes a difference!"

    # Streak
    if total_actions == 0:
        streak_text = "Your ARIA journey starts now!"
    elif sentences >= 50:
        streak_text = f"Amazing! {sentences} sentences translated — you're a power user!"
    elif sentences >= 10:
        streak_text = f"{sentences} sentences and counting — great progress!"
    elif sessions >= 5:
        streak_text = f"{sessions} sessions completed — you're building a habit!"
    elif total_actions >= 10:
        streak_text = f"{total_actions} total actions — keep the momentum going!"
    else:
        streak_text = "You're off to a great start. Keep exploring!"

    return {
        "greeting": greeting,
        "summary": summary,
        "sign_insight": sign_insight,
        "guide_insight": guide_insight,
        "tip": tip,
        "streak_text": streak_text,
    }


async def _try_gemini_insights(stats: dict) -> dict | None:
    """Attempt to generate insights via Gemini, trying multiple model names."""
    if not settings.gemini_api_key:
        logger.info("gemini_key_not_set", msg="Using local insights fallback")
        return None

    genai.configure(api_key=settings.gemini_api_key)
    prompt = INSIGHTS_PROMPT.format(data=json.dumps(stats, indent=2))

    for model_name in GEMINI_MODEL_CANDIDATES:
        try:
            logger.info("trying_gemini_model", model=model_name)
            model = genai.GenerativeModel(
                model_name,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.7,
                ),
            )
            response = await asyncio.to_thread(
                lambda m=model: m.generate_content(prompt).text
            )
            # Strip markdown fences if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()
            result = json.loads(cleaned)
            logger.info("gemini_insights_ok", model=model_name)
            return result
        except Exception as e:
            logger.warning(
                "gemini_model_failed",
                model=model_name,
                error=str(e),
            )
            continue

    logger.error("all_gemini_models_failed", msg="Falling back to local insights")
    return None


@router.get("/insights")
async def dashboard_insights(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """AI-generated dashboard insights using Gemini with local fallback."""
    uid = user.id

    # ── Gather raw stats ──

    sign_sessions = await db.scalar(
        select(func.count()).select_from(SignSession).where(SignSession.user_id == uid)
    ) or 0
    transcript_entries = await db.scalar(
        select(func.count())
        .select_from(TranscriptEntry)
        .join(SignSession, TranscriptEntry.session_id == SignSession.id)
        .where(SignSession.user_id == uid)
    ) or 0

    emotion_rows = (
        await db.execute(
            select(TranscriptEntry.emotion, func.count().label("count"))
            .join(SignSession, TranscriptEntry.session_id == SignSession.id)
            .where(SignSession.user_id == uid)
            .group_by(TranscriptEntry.emotion)
            .order_by(func.count().desc())
        )
    ).all()
    emotion_dist = {row.emotion: row.count for row in emotion_rows}

    lang_rows = (
        await db.execute(
            select(TranscriptEntry.language, func.count().label("count"))
            .join(SignSession, TranscriptEntry.session_id == SignSession.id)
            .where(SignSession.user_id == uid)
            .group_by(TranscriptEntry.language)
            .order_by(func.count().desc())
        )
    ).all()
    lang_usage = {row.language: row.count for row in lang_rows}

    nav_total = await db.scalar(
        select(func.count()).select_from(NavigationLog).where(NavigationLog.user_id == uid)
    ) or 0
    nav_completed = await db.scalar(
        select(func.count())
        .select_from(NavigationLog)
        .where(NavigationLog.user_id == uid, NavigationLog.ended_at.isnot(None))
    ) or 0

    obstacle_total = await db.scalar(
        select(func.count())
        .select_from(APIUsage)
        .where(APIUsage.endpoint == "detect_obstacle")
    ) or 0

    sos_count = await db.scalar(
        select(func.count()).select_from(SOSEvent).where(SOSEvent.user_id == uid)
    ) or 0

    recent_rows = (
        await db.execute(
            select(TranscriptEntry.text, TranscriptEntry.emotion)
            .join(SignSession, TranscriptEntry.session_id == SignSession.id)
            .where(SignSession.user_id == uid)
            .order_by(TranscriptEntry.created_at.desc())
            .limit(5)
        )
    ).all()
    recent = [{"text": r.text, "emotion": r.emotion} for r in recent_rows]

    session_rows = (
        await db.execute(
            select(SignSession.started_at, SignSession.ended_at)
            .where(SignSession.user_id == uid, SignSession.ended_at.isnot(None))
        )
    ).all()
    avg_dur = 0.0
    if session_rows:
        durations = [(r.ended_at - r.started_at).total_seconds() for r in session_rows]
        avg_dur = round(sum(durations) / len(durations), 1)

    # ── Build stats payload ──
    stats = {
        "user_name": user.name,
        "sign_mode": {
            "total_sessions": sign_sessions,
            "total_sentences": transcript_entries,
            "emotion_distribution": emotion_dist,
            "language_usage": lang_usage,
            "avg_session_duration_seconds": avg_dur,
            "recent_sentences": recent,
        },
        "guide_mode": {
            "total_navigations": nav_total,
            "completed_navigations": nav_completed,
            "total_obstacle_scans": obstacle_total,
        },
        "sos_events": sos_count,
    }

    # ── Try Gemini, fall back to local insights ──
    ai_insights = await _try_gemini_insights(stats)
    if ai_insights is None:
        ai_insights = _generate_local_insights(stats)

    return {
        "stats": stats,
        "insights": ai_insights,
    }
