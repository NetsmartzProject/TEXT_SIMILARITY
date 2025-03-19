from fastapi import APIRouter, HTTPException
from app.views.similarity_request import TextSimilarityInput
from app.models.similarity_model import STSController

router = APIRouter(prefix="/similarity", tags=["Semantic Similarity"])
controller = STSController()

@router.post("/")
def compute_similarity(data: TextSimilarityInput):
    """API to compute semantic similarity between two texts."""
    similarity_score = controller.compute_similarity(data)
    return {
        "text1": data.text1,
        "text2": data.text2,
        "similarity_score": similarity_score
    }
