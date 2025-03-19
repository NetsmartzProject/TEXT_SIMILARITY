# TEXT_SIMILARITY

This repository contains two APIs for **Semantic Textual Similarity (STS)** using **transformer-based models**.

## **1. STS Model (Sentence Transformers)**
The first API utilizes **Hugging Face's `sentence-transformers/all-mpnet-base-v2` model** to compute similarity scores between two input texts.  
### **Features:**
- Uses PyTorch for deep learning inference.
- Mean pooling technique for embedding generation.
- Computes cosine similarity between two text embeddings.

## **2. STS Model (Word LLaMA)**
The second API leverages **WordLLaMA**, a lightweight and efficient model for text similarity tasks.  
### **Features:**
- Optimized for performance with reduced computational overhead.
- Suitable for sentence-level and document-level similarity tasks.



