from fastapi import APIRouter
from app.views.similarity_request import TextSimilarityInput
from fastapi.responses import JSONResponse
from app.models.wordllama_service import WordLlamaController

router = APIRouter(prefix="/wordllama", tags=["WordLlama"])
controller = WordLlamaController()

@router.post("/similarity")
def compute_similarity(data: TextSimilarityInput):
    """API endpoint to compute text similarity."""
    raw_score, normalized_score = controller.compute_similarity(data.text1, data.text2)
    return {
        "text1": data.text1,
        "text2": data.text2,
        "raw_similarity_score": raw_score,
        "normalized_similarity_score": normalized_score
    }
