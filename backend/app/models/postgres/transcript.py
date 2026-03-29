import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.postgres.base import Base, UUIDMixin, UUIDType


class SignSession(Base, UUIDMixin):
    __tablename__ = "sign_sessions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUIDType, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    entries: Mapped[list["TranscriptEntry"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class TranscriptEntry(Base, UUIDMixin):
    __tablename__ = "transcript_entries"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUIDType,
        ForeignKey("sign_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    raw_letters: Mapped[str | None] = mapped_column(String(100), nullable=True)
    emotion: Mapped[str] = mapped_column(String(20), default="neutral", nullable=False)
    language: Mapped[str] = mapped_column(String(10), default="en", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    session: Mapped["SignSession"] = relationship(back_populates="entries")
