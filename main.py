# from fastapi import FastAPI
# from app.controllers.similarity_controller import router
# from app.controllers.wordllama_controller import router as wordllama_router
# import uvicorn


# app = FastAPI(title="Semantic Similarity API", version="1.0")

# app.include_router(router)
# app.include_router(wordllama_router, prefix="/wordllama", tags=["WordLlama"])

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8801)


import os
from fastapi import FastAPI
from app.controllers.similarity_controller import router
from app.controllers.wordllama_controller import router as wordllama_router
import uvicorn

app = FastAPI(title="Semantic Similarity API", version="1.0")

app.include_router(router)
app.include_router(wordllama_router, prefix="/wordllama", tags=["WordLlama"])

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
