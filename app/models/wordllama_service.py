from wordllama import WordLlama

class WordLlamaController:
    def __init__(self):
        """Load WordLlama model"""
        self.model = WordLlama.load()  
        
    def compute_similarity(self, text1: str, text2: str):
        """Compute similarity between two texts and normalize it."""
        raw_score = self.model.similarity(text1, text2)

        # Ensure raw_score is in a valid range (assuming model outputs between 0 and 1)
        raw_score = max(0.0, min(1.0, raw_score))  # Clamp between 0 and 1

        # Soft normalization: Just scale slightly if needed
        normalized_score = (raw_score - 0.2) / 0.8  # Shifts range to 0.2–1 → 0–1
        normalized_score = max(0.0, min(1.0, normalized_score))  # Ensure valid range

        return raw_score, normalized_score
