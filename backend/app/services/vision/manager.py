import asyncio
import time

from app.config import settings
from app.db.postgres import async_session_factory
from app.models.postgres.api_usage import APIUsage
from app.services.vision.base import VisionProvider, VisionResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VisionProviderUnavailableError(Exception):
    pass


class VisionManager:
    """Manages multiple vision providers with ordered fallback and health tracking."""

    def __init__(self):
        self._providers: list[VisionProvider] = []
        self._health_cache: dict[str, bool] = {}
        self._fail_counts: dict[str, int] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize providers based on config. Called during app startup."""
        from app.services.vision.claude_provider import ClaudeProvider
        from app.services.vision.gemini_provider import GeminiProvider
        from app.services.vision.local_provider import LocalProvider
        from app.services.vision.openai_provider import OpenAIProvider

        provider_map = {
            "gemini": (GeminiProvider, settings.gemini_api_key),
            "openai": (OpenAIProvider, settings.openai_api_key),
            "claude": (ClaudeProvider, settings.anthropic_api_key),
            "local": (LocalProvider, "always"),  # Local doesn't need API key
        }

        order = [p.strip() for p in settings.vision_provider_order.split(",")]

        for name in order:
            if name not in provider_map:
                logger.warning("unknown_vision_provider", name=name)
                continue

            cls, key = provider_map[name]
            if key:  # Only add providers that have API keys configured (or "always")
                try:
                    provider = cls()
                    self._providers.append(provider)
                    self._health_cache[provider.name] = True
                    logger.info("vision_provider_registered", name=provider.name)
                except Exception as e:
                    logger.warning("vision_provider_init_failed", name=name, error=str(e))

        self._initialized = True
        logger.info("vision_manager_ready", providers=[p.name for p in self._providers])

    async def detect_obstacle(self, image_b64: str) -> VisionResult:
        # Use local YOLO only — fast, no network dependency, no Gemini timeout risk
        for provider in self._providers:
            if provider.name == "local":
                start = time.monotonic()
                result = await provider.detect_obstacle(image_b64)
                latency = (time.monotonic() - start) * 1000
                await self._log_usage("local", "detect_obstacle", int(latency), True)
                return result
        raise VisionProviderUnavailableError("Local vision provider not available")

    async def build_sentence(self, partial_text: str, emotion: str) -> VisionResult:
        return await self._call_with_fallback("build_sentence", partial_text, emotion)

    async def translate(self, text: str, target_lang: str) -> VisionResult:
        return await self._call_with_fallback("translate", text, target_lang)

    async def _call_with_fallback(self, method: str, *args) -> VisionResult:
        if not self._initialized or not self._providers:
            raise VisionProviderUnavailableError("No vision providers available")

        last_error = None

        for provider in self._providers:
            if not self._health_cache.get(provider.name, True):
                continue

            start = time.monotonic()
            try:
                # 5-second timeout per provider — fail fast, fall through quickly
                result = await asyncio.wait_for(
                    getattr(provider, method)(*args),
                    timeout=5.0,
                )
                latency = (time.monotonic() - start) * 1000
                self._fail_counts[provider.name] = 0
                await self._log_usage(provider.name, method, int(latency), True)
                return result
            except NotImplementedError:
                # This provider doesn't support this method, skip silently
                continue
            except asyncio.TimeoutError:
                latency = (time.monotonic() - start) * 1000
                await self._log_usage(provider.name, method, int(latency), False, "timeout")
                # Timeout counts as immediate unhealthy — provider is too slow
                self._health_cache[provider.name] = False
                self._fail_counts[provider.name] = 3
                last_error = TimeoutError(f"{provider.name} timed out after 5s")
                logger.warning(
                    "vision_provider_timeout",
                    provider=provider.name,
                    method=method,
                    latency_ms=int(latency),
                )
                continue
            except Exception as e:
                latency = (time.monotonic() - start) * 1000
                await self._log_usage(provider.name, method, int(latency), False, str(e))
                # Mark unhealthy immediately on error — don't waste time retrying broken providers
                self._health_cache[provider.name] = False
                self._fail_counts[provider.name] = self._fail_counts.get(provider.name, 0) + 1
                last_error = e
                logger.warning(
                    "vision_provider_failed",
                    provider=provider.name,
                    method=method,
                    error=str(e),
                    fail_count=self._fail_counts[provider.name],
                )
                continue

        raise VisionProviderUnavailableError(
            f"All vision providers failed for {method}. Last error: {last_error}"
        )

    async def refresh_health(self) -> None:
        """Re-check unhealthy providers. Called periodically."""
        for provider in self._providers:
            if not self._health_cache.get(provider.name, True):
                try:
                    healthy = await provider.health_check()
                    self._health_cache[provider.name] = healthy
                    if healthy:
                        logger.info("vision_provider_recovered", name=provider.name)
                except Exception:
                    pass

    async def run_health_loop(self) -> None:
        """Background task to periodically refresh provider health."""
        while True:
            await asyncio.sleep(15)  # Check every 15s for faster recovery
            await self.refresh_health()

    async def _log_usage(
        self,
        provider: str,
        endpoint: str,
        latency_ms: int,
        success: bool,
        error_message: str | None = None,
    ) -> None:
        try:
            async with async_session_factory() as session:
                usage = APIUsage(
                    provider=provider,
                    endpoint=endpoint,
                    latency_ms=latency_ms,
                    success=success,
                    error_message=error_message,
                )
                session.add(usage)
                await session.commit()
        except Exception:
            # Don't let logging failures break the main flow
            pass

    def reset_provider_health(self, name: str | None = None) -> dict[str, str]:
        """Force-reset health cache. If name given, reset only that provider."""
        if name:
            if name in self._health_cache:
                self._health_cache[name] = True
        else:
            for p in self._providers:
                self._health_cache[p.name] = True
        return self.get_provider_status()

    def get_provider_status(self) -> dict[str, str]:
        return {
            p.name: "ok" if self._health_cache.get(p.name, True) else "unhealthy"
            for p in self._providers
        }


# Singleton instance
vision_manager = VisionManager()
