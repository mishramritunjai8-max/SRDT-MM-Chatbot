import logging
from typing import List, Tuple
from duckduckgo_search import DDGS
from backend.app.core.config import settings
from backend.app.schemas.chat import SourceItem

logger = logging.getLogger(__name__)

WEB_SEARCH_SYSTEM_PROMPT = (
    "You are an objective, reliable assistant with access to real-time web search results.\n\n"
    "GROUNDING RULES FOR WEB SEARCH:\n"
    "1. Synthesize your answer using ONLY the factual claims in the provided web search snippets.\n"
    "2. If the search results do not contain enough facts to answer the query, clearly admit: "
    "'I could not find sufficiently reliable current web results to answer this question.'\n"
    "3. Never invent, hallucinate, or fabricate search results, websites, authors, or URLs.\n"
    "4. Clearly distinguish information derived from the web results versus general definitions."
)


class WebSearchService:
    def __init__(self):
        self.max_results = settings.MAX_SEARCH_RESULTS

    def search_and_ground(self, query: str) -> Tuple[List[str], List[SourceItem]]:
        """
        Executes a real web search via DuckDuckGo and formats context snippets and clickable sources.
        """
        snippets: List[str] = []
        sources: List[SourceItem] = []

        try:
            with DDGS() as ddgs:
                results = []
                for backend_choice in ["html", "lite", "auto"]:
                    try:
                        results = list(ddgs.text(query, backend=backend_choice, max_results=self.max_results))
                        if results:
                            break
                    except Exception as be_err:
                        logger.debug(f"DDGS backend '{backend_choice}' failed: {be_err}")
                        continue

                for item in results:
                    title = item.get("title", "Web Page")
                    href = item.get("href", "")
                    body = item.get("body", "")

                    if href and body:
                        snippets.append(f"[Title: {title} | URL: {href}]\n{body}")
                        sources.append(
                            SourceItem(
                                title=title,
                                url=href,
                                snippet=body[:180] + "..." if len(body) > 180 else body,
                                source_type="web"
                            )
                        )
        except Exception as e:
            logger.warning(f"DuckDuckGo search error for query '{query}': {e}")

        # If DuckDuckGo returned 0 results or was rate-limited (403), seamlessly use Wikipedia knowledge API
        if not snippets:
            logger.info(f"DuckDuckGo search yielded no results; activating Wikipedia fallback for '{query}'")
            snippets, sources = self._search_wikipedia_fallback(query)

        return snippets, sources

    def _search_wikipedia_fallback(self, query: str) -> Tuple[List[str], List[SourceItem]]:
        """Fallback to Wikipedia REST API when DuckDuckGo is rate-limited or unavailable."""
        import httpx
        clean_q = query.lower()
        for prefix in ["what is a ", "what is an ", "what is ", "who is ", "explain ", "tell me about ", "search for "]:
            if clean_q.startswith(prefix):
                clean_q = clean_q[len(prefix):]
                break

        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "search": clean_q.strip(),
            "limit": self.max_results,
            "namespace": 0,
            "format": "json"
        }
        headers = {"User-Agent": "MM-Chatbot/1.0 (developer@example.com)"}
        snippets: List[str] = []
        sources: List[SourceItem] = []

        try:
            with httpx.Client(timeout=6.0) as client:
                resp = client.get(url, params=params, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    titles = data[1] if len(data) > 1 else []
                    urls = data[3] if len(data) > 3 else []
                    for t, u in zip(titles[:self.max_results], urls[:self.max_results]):
                        sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{t.replace(' ', '_')}"
                        sum_resp = client.get(sum_url, headers=headers)
                        extract = ""
                        if sum_resp.status_code == 200:
                            extract = sum_resp.json().get("extract", "")

                        body_text = extract if extract else f"Wikipedia article for {t}."
                        snippets.append(f"[Title: {t} (Wikipedia) | URL: {u}]\n{body_text}")
                        sources.append(
                            SourceItem(
                                title=f"{t} (Wikipedia)",
                                url=u,
                                snippet=body_text[:180] + "..." if len(body_text) > 180 else body_text,
                                source_type="web"
                            )
                        )
        except Exception as e:
            logger.warning(f"Wikipedia fallback error: {e}")

        return snippets, sources

    @staticmethod
    def get_system_prompt() -> str:
        return WEB_SEARCH_SYSTEM_PROMPT
