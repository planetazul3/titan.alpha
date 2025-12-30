"""
Base classes for domain entities.
"""
from typing import Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict

class DomainEntity(BaseModel):
    """Base class for all domain entities."""
    model_config = ConfigDict(frozen=True, extra="ignore")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode='json')

class AggregateRoot(DomainEntity):
    """Base class for aggregate roots."""
    pass

def now_utc() -> datetime:
    """Returns current UTC time."""
    return datetime.now(timezone.utc)
