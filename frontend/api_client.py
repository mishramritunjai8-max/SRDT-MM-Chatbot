import os
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000").rstrip("/")


class APIClient:
    """Client for communicating with the FastAPI backend."""

    def __init__(self, base_url: str = BACKEND_URL):
        self.base_url = base_url

    def get_health(self) -> Dict[str, Any]:
        """Fetch system health, database status, and active AI providers."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return {"status": "error", "detail": f"Status code {resp.status_code}"}
        except Exception as e:
            return {"status": "offline", "detail": str(e)}

    def get_available_models(self) -> Dict[str, Any]:
        """Fetch list of operational models and provider status."""
        try:
            resp = requests.get(f"{self.base_url}/models/available", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return {"models": [], "default_model": "mock", "active_providers": []}
        except Exception:
            return {"models": [], "default_model": "mock", "active_providers": []}

    def send_chat(
        self,
        message: str,
        model: str = "auto",
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        use_rag: bool = False,
        document_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        use_web_search: bool = False,
        emotional_mode: bool = False
    ) -> Dict[str, Any]:
        """Send chat prompt to backend with auto-switching and grounding options."""
        payload = {
            "message": message,
            "session_id": session_id,
            "model": model,
            "system_prompt": system_prompt,
            "use_rag": use_rag,
            "document_id": document_id,
            "document_ids": document_ids,
            "use_web_search": use_web_search,
            "emotional_mode": emotional_mode
        }
        try:
            resp = requests.post(f"{self.base_url}/chat", json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            error_detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            return {"error": f"Backend Error ({resp.status_code}): {error_detail}"}
        except requests.exceptions.ConnectionError:
            return {"error": "Could not connect to FastAPI backend. Ensure the backend server is running on port 8000."}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}

    # Document management
    def upload_document(self, file_bytes: bytes, filename: str, content_type: str) -> Dict[str, Any]:
        """Upload a file (PDF, DOCX, TXT) for RAG chunking and vector storage."""
        try:
            files = {"file": (filename, file_bytes, content_type)}
            resp = requests.post(f"{self.base_url}/documents/upload", files=files, timeout=60)
            if resp.status_code == 201:
                return resp.json()
            error_detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            return {"error": f"Upload failed ({resp.status_code}): {error_detail}"}
        except Exception as e:
            return {"error": f"Upload error: {str(e)}"}

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all ingested documents in database."""
        try:
            resp = requests.get(f"{self.base_url}/documents", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return []
        except Exception:
            return []

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its cascading chunk embeddings."""
        try:
            resp = requests.delete(f"{self.base_url}/documents/{document_id}", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def create_generated_document(self, title: str, content: str, doc_format: str = "markdown") -> Dict[str, Any]:
        """Save response as a downloadable versioned document."""
        try:
            payload = {"title": title, "content": content, "format": doc_format}
            resp = requests.post(f"{self.base_url}/documents/create", json=payload, timeout=10)
            if resp.status_code == 201:
                return resp.json()
            return {"error": resp.text}
        except Exception as e:
            return {"error": str(e)}

    # Multimodal Vision and Audio
    def analyze_image(
        self,
        image_bytes: bytes,
        filename: str,
        content_type: str,
        prompt: str = "Analyze this image",
        model: str = "auto"
    ) -> Dict[str, Any]:
        """Analyze an uploaded image using vision model."""
        try:
            files = {"file": (filename, image_bytes, content_type)}
            data = {"prompt": prompt, "model": model}
            resp = requests.post(f"{self.base_url}/multimodal/vision", files=files, data=data, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            return {"error": resp.text}
        except Exception as e:
            return {"error": str(e)}

    def synthesize_speech(self, text: str) -> Optional[bytes]:
        """Convert text response to speech audio bytes."""
        try:
            resp = requests.post(f"{self.base_url}/multimodal/synthesize", data={"text": text}, timeout=30)
            if resp.status_code == 200:
                return resp.content
            return None
        except Exception:
            return None

    def generate_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        model: str = "flux",
        upscale: bool = True,
        scale_factor: float = 2.0,
        enhance_prompt: bool = True
    ) -> Optional[bytes]:
        """Generate high-resolution, crystal-clear, artifact-free image with AI upscaling."""
        try:
            data = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "model": model,
                "upscale": upscale,
                "scale_factor": scale_factor,
                "enhance_prompt": enhance_prompt
            }
            resp = requests.post(f"{self.base_url}/multimodal/generate-image", data=data, timeout=90)
            if resp.status_code == 200 and len(resp.content) > 500:
                return resp.content
            return None
        except Exception:
            return None

    def upscale_image(
        self,
        image_bytes: bytes,
        scale_factor: float = 2.0,
        clarity_boost: bool = True
    ) -> Optional[bytes]:
        """Upscale and sharpen an existing image using AI super-resolution."""
        try:
            files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
            data = {"scale_factor": scale_factor, "clarity_boost": clarity_boost}
            resp = requests.post(f"{self.base_url}/multimodal/upscale-image", files=files, data=data, timeout=30)
            if resp.status_code == 200:
                return resp.content
            return None
        except Exception:
            return None
