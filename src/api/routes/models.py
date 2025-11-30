"""OpenAI-compatible models endpoint."""

from fastapi import APIRouter

from src.api.models.schemas import Model, ModelList

router = APIRouter()


@router.get("/models", response_model=ModelList)
async def list_models():
    """
    List available models (OpenAI-compatible endpoint).

    Returns:
        List of available models
    """
    models = [
        Model(
            id="clinical-trial-assistant",
            owned_by="clinical-trials-org",
        )
    ]

    return ModelList(data=models)
