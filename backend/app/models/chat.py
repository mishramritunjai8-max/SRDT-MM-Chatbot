from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from backend.app.models.base import Base, generate_uuid, get_utc_now


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_mode = Column(String(50), default="general")  # general, rag, emotional, web_search
    title = Column(String(255), default="New Conversation")
    created_at = Column(DateTime, default=get_utc_now)

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    model_used = Column(String(100), nullable=True)
    switched_from = Column(String(100), nullable=True)
    switch_reason = Column(Text, nullable=True)
    sources = Column(JSON, default=list)  # list of chunk or web sources
    created_at = Column(DateTime, default=get_utc_now)

    session = relationship("ChatSession", back_populates="messages")
