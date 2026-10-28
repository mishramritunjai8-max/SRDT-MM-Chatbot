from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.app.db.session import get_db
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["Conversational Chat"])


@router.post("", response_model=ChatResponse)
def handle_chat(req: ChatRequest, db: Session = Depends(get_db)):
    """
    Main conversational chat endpoint.
    Supports auto-switching between Grok and Gemini, RAG context, web search grounding, and emotional mode.
    """
    try:
        service = ChatService(db)
        return service.process_chat(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
