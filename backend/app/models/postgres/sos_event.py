import uuid
from datetime import datetime

from sqlalchemy import DateTime, Double, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.postgres.base import Base, UUIDMixin, UUIDType


class SOSEvent(Base, UUIDMixin):
    __tablename__ = "sos_events"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUIDType, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    latitude: Mapped[float | None] = mapped_column(Double, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Double, nullable=True)
    triggered_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
