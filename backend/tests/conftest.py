import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Set test environment variables
os.environ["DATABASE_URL"] = "sqlite:///./test_mm_chatbot.db"
os.environ["DEFAULT_MODEL"] = "mock"
os.environ["GEMINI_API_KEY"] = ""
os.environ["GROK_API_KEY"] = ""
os.environ["GROQ_API_KEY"] = ""

from backend.app.core.config import settings
settings.DATABASE_URL = "sqlite:///./test_mm_chatbot.db"
settings.DEFAULT_MODEL = "mock"
settings.GEMINI_API_KEY = ""
settings.GROK_API_KEY = ""
settings.GROQ_API_KEY = ""

from backend.app.main import app
from backend.app.models.base import Base
from backend.app.db.session import get_db

TEST_DATABASE_URL = "sqlite:///./test_mm_chatbot.db"
test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    if os.path.exists("./test_mm_chatbot.db"):
        try:
            os.remove("./test_mm_chatbot.db")
        except Exception:
            pass


@pytest.fixture(autouse=True)
def mock_external_calls(monkeypatch):
    """Ensure automated tests never make external network calls."""
    from backend.app.services.gemini_service import GeminiService
    from backend.app.services.grok_service import GrokService
    from backend.app.services.image_generation_service import ImageGenerationService
    from backend.app.api.multimodal import image_gen_service
    monkeypatch.setattr(GeminiService, "is_configured", lambda self: False)
    monkeypatch.setattr(GrokService, "is_configured", lambda self: False)
    monkeypatch.setattr(ImageGenerationService, "_fetch_online_image", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(image_gen_service, "_fetch_online_image", lambda *args, **kwargs: None)


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
