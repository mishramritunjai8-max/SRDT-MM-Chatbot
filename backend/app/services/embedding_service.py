from typing import List
from backend.app.services.provider_router import ProviderRouter


class EmbeddingService:
    def __init__(self):
        self.router = ProviderRouter()

    def embed_text(self, text: str) -> List[float]:
        return self.router.get_embedding(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]
