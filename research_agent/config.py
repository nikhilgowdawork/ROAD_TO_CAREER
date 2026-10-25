from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class settings(BaseSettings):

    groq_api_key: SecretStr
    groq_model: str = "openai/gpt-oss-120b"

    tavily_api_key: SecretStr

    max_research_iterations: int = 6
    max_tool_calls: int = 12


    model_config = SettingsConfigDict(
        env_file= ".env", 
        env_file_encoding= "utf-8",
        extra="ignore",
    
    )

settings = settings()