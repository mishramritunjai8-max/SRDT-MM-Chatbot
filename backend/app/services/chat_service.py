import logging
import uuid
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from backend.app.models.chat import ChatSession, ChatMessage
from backend.app.schemas.chat import ChatRequest, ChatResponse, SourceItem
from backend.app.services.provider_router import ProviderRouter
from backend.app.services.emotional_support_service import EmotionalSupportService

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, db: Session):
        self.db = db
        self.router = ProviderRouter()

    def process_chat(self, req: ChatRequest) -> ChatResponse:
        # 1. Check for emotional mode safety trigger
        if req.emotional_mode:
            crisis_response = EmotionalSupportService.check_crisis(req.message)
            if crisis_response:
                return ChatResponse(
                    response=crisis_response,
                    session_id=req.session_id or str(uuid.uuid4()),
                    model_used="safety-guardrail",
                    sources=[]
                )

        # 2. Get or create chat session
        session_id = req.session_id
        session = None
        history: List[Dict[str, str]] = []

        if session_id:
            session = self.db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if session:
                past_messages = self.db.query(ChatMessage).filter(
                    ChatMessage.session_id == session_id
                ).order_by(ChatMessage.created_at.asc()).limit(10).all()
                for pm in past_messages:
                    history.append({"role": pm.role, "content": pm.content})
        
        if not session:
            session_id = str(uuid.uuid4())
            session_mode = "emotional" if req.emotional_mode else ("rag" if req.use_rag else "general")
            # Create session only if not in ephemeral mode or if general chat
            session = ChatSession(id=session_id, session_mode=session_mode, title=req.message[:50])
            self.db.add(session)
            self.db.commit()

        # 3. Determine system instructions
        system_instruction = req.system_prompt
        sources: List[SourceItem] = []
        context_chunks: List[str] = []

        if req.emotional_mode:
            system_instruction = EmotionalSupportService.get_system_prompt()
        else:
            if req.use_rag:
                from backend.app.services.rag_service import RAGService
                rag_service = RAGService(self.db)
                retrieved_chunks, rag_sources = rag_service.retrieve_context(
                    query=req.message,
                    document_id=req.document_id,
                    document_ids=req.document_ids
                )
                context_chunks.extend(retrieved_chunks)
                sources.extend(rag_sources)
                if not system_instruction:
                    system_instruction = RAGService.get_system_prompt()

            if req.use_web_search:
                from backend.app.services.web_search_service import WebSearchService
                search_service = WebSearchService()
                search_snippets, search_sources = search_service.search_and_ground(req.message)
                context_chunks.extend(search_snippets)
                sources.extend(search_sources)
                if not system_instruction:
                    system_instruction = WebSearchService.get_system_prompt()

        # 4. Generate AI response through router with auto-switching
        response_text, model_used, switched_from, switch_reason = self.router.generate_chat_response(
            prompt=req.message,
            preferred_model=req.model,
            system_prompt=system_instruction,
            history=history,
            context_chunks=context_chunks if context_chunks else None,
            emotional_mode=req.emotional_mode
        )

        # 5. Save message history to DB (omit sensitive emotional chats if preferred)
        if not req.emotional_mode:
            user_msg = ChatMessage(
                session_id=session_id,
                role="user",
                content=req.message,
                model_used=model_used
            )
            asst_msg = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=response_text,
                model_used=model_used,
                switched_from=switched_from,
                switch_reason=switch_reason,
                sources=[s.model_dump() for s in sources]
            )
            self.db.add(user_msg)
            self.db.add(asst_msg)
            self.db.commit()

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            model_used=model_used,
            switched_from=switched_from,
            switch_reason=switch_reason,
            sources=sources
        )
