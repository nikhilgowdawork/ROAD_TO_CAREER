
import os

from tavily import TavilyClient
from langchain_core.tools import tool


@tool
def search_web(query: str ) -> dict:

    """Search the web for current information relevant to a research query.

    Args:
        query: A specific topic, question, technology, or job requirement.
        max_results: Maximum number of search results to return (1-10).
    """
    max_results = 3

    if not query or not query.strip():
        return {"error": "Search query cannot be empty.", "results": []}

    max_results = max(1, min(max_results, 3))

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "error": "TAVILY_API_KEY is not configured.",
            "results": [],
        }

    try:
        client = TavilyClient(api_key=api_key)

        response = client.search(
            query=query.strip(),
            max_results=max_results,
            search_depth="advanced",
            topic="general",
            include_answer=False,
            include_raw_content=False,
        )

        results = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score", 0),
            }
            for item in response.get("results", [])
        ]

        return {
            "query": query.strip(),
            "result_count": len(results),
            "results": results,
        }

    except Exception as exc:
        return {
            "error": f"Web search failed: {type(exc).__name__}: {exc}",
            "results": [],
        }
