
from langchain_core.tools import tool
from tavily import TavilyClient

from research_agent.config import settings


@tool
def search_web(query: str) -> dict:
    """Search the web for current information relevant to a research question.
    
    Use this tool to discover relevant webpages, job listings, official
    documentation, and industry discussions. Returns results with titles,
    URLs, and content snippets.
    """
    query = query.strip()

    if not query:
        return {"error": "Search query cannot be empty."}

    client = TavilyClient(
        api_key=settings.tavily_api_key.get_secret_value()
    )

    try:
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=5,
        )

        results = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            }
            for item in response.get("results", [])
        ]

        return {
            "query": query,
            "results": results,
            "result_count": len(results),
        }

    except Exception as exc:
        raise RuntimeError(
            f"Web search failed: {type(exc).__name__}"
        ) from exc
