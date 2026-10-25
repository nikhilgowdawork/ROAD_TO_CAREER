
from langchain_groq import ChatGroq

from research_agent.config import settings


def get_llm() -> ChatGroq:
    """Initialize and return the Groq language model."""
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key.get_secret_value(),
        temperature=0.1,
        max_retries=2,
        timeout=30,
    )


def get_tool_calling_llm(tools: list) -> ChatGroq:
    """Return the model configured with available tools."""
    return get_llm().bind_tools(tools)
