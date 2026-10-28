from datetime import datetime, timezone
import uuid
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, DateTime, String

Base = declarative_base()


def generate_uuid() -> str:
    return str(uuid.uuid4())


def get_utc_now() -> datetime:
    return datetime.now(timezone.utc)
