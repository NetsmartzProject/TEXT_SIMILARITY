from wordllama import WordLlama

class WordLlamaController:
    def __init__(self):
        """Load WordLlama model"""
        self.model = WordLlama.load()  

    def compute_similarity(self, text1: str, text2: str):
        """Compute similarity between two texts."""
        return self.model.similarity(text1, text2)

    def rank_texts(self, query: str, candidates: list):
        """Rank documents based on similarity to a query."""
        return self.model.rank(query, candidates)
