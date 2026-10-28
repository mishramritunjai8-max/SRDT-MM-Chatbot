from typing import List, Dict, Any, Optional


class MockService:
    """
    Deterministic Mock AI Service.
    Enables offline testing, grading, and instant verification without API keys.
    """

    def generate_chat_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        context_chunks: Optional[List[str]] = None,
        emotional_mode: bool = False
    ) -> str:
        prompt_lower = prompt.lower()

        # 1. Emotional Mode Response
        if emotional_mode:
            return (
                "I hear you, and it sounds like you're going through a meaningful moment. "
                "It is completely valid to feel this way. Take a deep breath—what part of this situation "
                "feels most pressing for you right now, and how can I best support you in thinking through it?"
            )

        # 2. Context-Grounded Response (Web Search or Document RAG)
        if context_chunks:
            is_web = any("[Title:" in c or "http" in c for c in context_chunks)
            if is_web:
                summary_lines = ["### 🌐 Live Web Search Grounded Summary:\n"]
                for i, chunk in enumerate(context_chunks[:4], 1):
                    # Clean snippet
                    clean_chunk = chunk.replace("[Title:", "**Source:**").replace("| URL:", "\n🔗 ")
                    summary_lines.append(f"{i}. {clean_chunk}\n")
                summary_lines.append(f"\n*Grounded across real-time web results for:* **{prompt}**")
                return "\n".join(summary_lines)
            else:
                joined_context = "\n\n".join(context_chunks[:3])
                return (
                    f"### 📄 Document RAG Grounded Answer:\n\n"
                    f"Based on your ingested documents, here are the extracted findings:\n\n"
                    f"{joined_context}\n\n"
                    f"*Directly addressing your query:* **{prompt}**"
                )

        # 3. Informative Mock Knowledge Base for common queries
        if "llm" in prompt_lower or "large language model" in prompt_lower:
            return (
                "### 🤖 Large Language Model (LLM) Overview\n\n"
                "A **Large Language Model (LLM)** is a sophisticated artificial intelligence algorithm trained on vast text corpora "
                "to understand, generate, and reason with natural language.\n\n"
                "**Core Foundations:**\n"
                "- **Transformer Architecture:** Utilizes multi-head self-attention mechanisms to weigh relationships between tokens across long contexts.\n"
                "- **Pre-training & Fine-tuning:** Pre-trained on self-supervised next-token prediction, then aligned via RLHF (Reinforcement Learning from Human Feedback) or DPO.\n"
                "- **Capabilities:** Summarization, coding, multi-step logical reasoning, translation, and multimodal vision analysis.\n"
                "- **Key Examples:** Google Gemini, xAI Grok, Meta Llama 3, and OpenAI GPT-4o."
            )

        if "photosynthesis" in prompt_lower:
            return (
                "### 🌿 Photosynthesis Summary\n\n"
                "Photosynthesis is the biological process used by plants, algae, and cyanobacteria to convert light energy into chemical energy:\n\n"
                "$$\\text{6CO}_2 + \\text{6H}_2\\text{O} + \\text{light} \\rightarrow \\text{C}_6\\text{H}_{12}\\text{O}_6 + \\text{6O}_2$$\n\n"
                "- **Light-dependent reactions:** Occur in the thylakoid membranes, generating ATP and NADPH.\n"
                "- **Calvin Cycle:** Occurs in the stroma, using ATP and NADPH to fix carbon dioxide into glucose."
            )

        if "weather" in prompt_lower or "news" in prompt_lower:
            return (
                f"### 🌐 Real-time Information for '{prompt}'\n\n"
                f"To fetch live real-time weather and current news, make sure the **🌐 Live Web Search Grounding** "
                f"option is enabled in your chat controls. The system will retrieve the latest articles and forecast data."
            )

        # 4. General conversational response
        return (
            f"### 💬 Multimodal Assistant (Offline Mock Mode)\n\n"
            f"You asked: **{prompt}**\n\n"
            f"I have received your request. In this mode, the system is operating through the deterministic local engine. "
            f"When you connect your active Google Gemini or Grok API keys, live deep intelligence and web grounding "
            f"will dynamically generate real-time answers!"
        )

    def analyze_image(self, prompt: str, image_bytes: bytes, mime_type: str) -> str:
        return (
            f"[Mock Vision Analysis]\n"
            f"Image received successfully ({len(image_bytes)} bytes, format: {mime_type}).\n"
            f"Analysis for prompt: '{prompt}':\n"
            f"The image appears to contain visual subjects and clear details relevant to your inquiry. "
            f"To enable live visual comprehension, configure your GEMINI_API_KEY or GROK_API_KEY."
        )

    def get_embedding(self, text: str, dimension: int = 768) -> List[float]:
        """Generate a deterministic synthetic embedding vector for testing."""
        import hashlib
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        import random
        rng = random.Random(seed)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
        # Normalize
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec] if norm > 0 else vec
