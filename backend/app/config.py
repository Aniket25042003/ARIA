from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database (single SQLite file)
    database_url: str = "sqlite+aiosqlite:///./aria.db"

    # API Keys
    gemini_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    elevenlabs_api_key: str = ""

    # Auth
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24 hours (simple auth)

    # AWS Bedrock
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_session_token: str = ""
    bedrock_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

    # Vision
    vision_provider_order: str = "bedrock,openai,claude,local"

    # Audio
    audio_output_dir: str = "./audio_output"
    audio_cleanup_max_age_seconds: int = 300  # 5 minutes

    # App
    app_version: str = "1.0.0"
    debug: bool = False

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
