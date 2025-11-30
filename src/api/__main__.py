"""Entry point for running the API server."""

import uvicorn

from src.config.settings import settings


def main():
    """Main entry point for the API server."""
    print(
        f"Starting Clinical Trial Assistant API on {settings.api_host}:{settings.api_port}"
    )
    print(
        f"OpenAI-compatible base URL: http://{settings.api_host}:{settings.api_port}/v1"
    )
    print(f"API documentation: http://{settings.api_host}:{settings.api_port}/docs")

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,  # Set to False in production
        log_level="info",
    )


if __name__ == "__main__":
    main()
