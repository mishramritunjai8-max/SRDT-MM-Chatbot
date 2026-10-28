from typing import Union


def format_file_size(size_bytes: Union[int, float]) -> str:
    """Format bytes into human-readable string (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def clean_snippet(text: str, max_chars: int = 200) -> str:
    """Clean and truncate snippet text for preview."""
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars] + "..."
