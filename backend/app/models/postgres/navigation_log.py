import uuid
from datetime import datetime

from sqlalchemy import DateTime, Double, ForeignKey, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.postgres.base import Base, UUIDMixin, UUIDType


class NavigationLog(Base, UUIDMixin):
    __tablename__ = "navigation_logs"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUIDType, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    origin_lat: Mapped[float | None] = mapped_column(Double, nullable=True)
    origin_lng: Mapped[float | None] = mapped_column(Double, nullable=True)
    destination: Mapped[str] = mapped_column(Text, nullable=False)
    total_steps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completed_steps: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
