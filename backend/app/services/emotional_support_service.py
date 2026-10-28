import re
from typing import Optional, Tuple

CRISIS_KEYWORDS = [
    r"\bsuicid",
    r"\bkill myself\b",
    r"\bend my life\b",
    r"\bself-harm\b",
    r"\bcut myself\b",
    r"\bhurt myself\b",
    r"\bwant to die\b",
    r"\bdon't want to live\b",
    r"\bno reason to live\b",
]

CRISIS_SAFETY_RESPONSE = (
    "I am deeply concerned about what you are sharing, and your life and safety are truly important. "
    "Please know that you do not have to carry this alone, and there is caring, free, confidential support available right now:\n\n"
    "- **In the US & Canada:** Call or text **988**, or chat at [988lifeline.org](https://988lifeline.org/)\n"
    "- **In the UK:** Call **111** or contact Samaritans at **116 123**\n"
    "- **In India:** Call **14416** (Tele-MANAS) or **112** for emergency assistance\n"
    "- **International:** Find local emergency support at [findahelpline.com](https://findahelpline.com/)\n\n"
    "Please reach out to one of these resources, a family member, or a healthcare professional right away."
)

EMOTIONAL_SYSTEM_PROMPT = (
    "You are an empathetic, supportive, and compassionate personal emotional reflection assistant. "
    "Your goal is to actively listen, validate feelings respectfully, and help the user gently reflect on situations "
    "and explore practical, positive coping strategies. "
    "CRITICAL SAFETY AND ETHICAL GUIDELINES:\n"
    "- Never judge, criticize, guilt, or manipulate the user.\n"
    "- You are NOT a doctor, therapist, or licensed mental health professional. Never diagnose conditions or prescribe medications.\n"
    "- Encourage speaking with a trusted friend, family member, or licensed counselor when situations feel overwhelming.\n"
    "- Ask gentle, open-ended follow-up questions to help the user unpack their emotions.\n"
    "- Keep your tone calm, grounded, warm, and conversational."
)


class EmotionalSupportService:
    """Handles empathetic conversations and strict crisis detection guardrails."""

    @staticmethod
    def check_crisis(text: str) -> Optional[str]:
        """Detects explicit crisis / self-harm triggers and returns emergency support response."""
        for pattern in CRISIS_KEYWORDS:
            if re.search(pattern, text, re.IGNORECASE):
                return CRISIS_SAFETY_RESPONSE
        return None

    @staticmethod
    def get_system_prompt() -> str:
        return EMOTIONAL_SYSTEM_PROMPT
