import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from app.views.similarity_request import TextSimilarityInput

def get_device():
    """Get the appropriate device (CPU or CUDA) based on availability."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mean_pooling(model_output, attention_mask):
    """Perform mean pooling on token embeddings."""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class STSController:
    """Handles text similarity calculations using a Transformer-based model."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def encode(self, texts):
        """Generate embeddings for the input text(s)."""
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return mean_pooling(outputs, inputs['attention_mask']).cpu().numpy()
    
    def compute_similarity(self, input_data: TextSimilarityInput) -> float:
        """Compute the semantic similarity between two texts."""
        embeddings = self.encode([input_data.text1, input_data.text2])
        return float(cosine_similarity(embeddings[0], embeddings[1]))
