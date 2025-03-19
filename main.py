from fastapi import FastAPI
from app.controllers.similarity_controller import router
from app.controllers.wordllama_controller import router as wordllama_router



app = FastAPI(title="Semantic Similarity API", version="1.0")

app.include_router(router)
app.include_router(wordllama_router, prefix="/wordllama", tags=["WordLlama"])
