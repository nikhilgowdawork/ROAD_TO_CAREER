
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def get_llm() -> ChatGroq:
    """Create and return the configured Groq chat model."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    model = os.getenv(
        "GROQ_MODEL")

    return ChatGroq(
        model=model,
        api_key=api_key,
        temperature=0,
        max_retries=2,
        timeout=60,
    )
