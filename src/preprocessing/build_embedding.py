import os
import numpy as np
from typing import List, Dict, Optional
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler

class MalwareEmbeddingPipeline:
    """
    Implements Section III-C & IV-A2:
    - Trains Word2Vec model on opcode sequences (dim=104)
    - Computes malware embedding vector via Average Pooling
    - Normalizes vector to [-1, 1] for Tanh compatibility
    """
    def __init__(self, embedding_dim: int = 104, window: int = 5, min_count: int = 1):
        self.embedding_dim = embedding_dim
        self.window = window
        self.min_count = min_count
        self.w2v_model: Optional[Word2Vec] = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def train_word2vec(self, corpus: List[List[str]], epochs: int = 20) -> Word2Vec:
        """
        Trains Word2Vec model on opcode sequences.
        corpus: List of lists, each sub-list is an opcode sequence of one malware sample.
        """
        print(f"[*] Training Word2Vec model on {len(corpus)} opcode sequences (dim={self.embedding_dim})...")
        self.w2v_model = Word2Vec(
            sentences=corpus,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            epochs=epochs
        )
        return self.w2v_model

    def embed_sample(self, opcode_sequence: List[str]) -> np.ndarray:
        """
        Computes 1D malware embedding by averaging Word2Vec vectors of opcodes in the sequence.
        """
        if self.w2v_model is None:
            raise ValueError("Word2Vec model is not trained yet!")

        vectors = []
        for op in opcode_sequence:
            if op in self.w2v_model.wv:
                vectors.append(self.w2v_model.wv[op])

        if len(vectors) == 0:
            # Fallback zero vector
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Average Pooling across the sequence
        avg_vector = np.mean(vectors, axis=0)
        return avg_vector

    def fit_transform(self, all_opcode_sequences: List[List[str]]) -> np.ndarray:
        """
        Trains Word2Vec and transforms all sequences into normalized [-1, 1] embeddings.
        """
        if self.w2v_model is None:
            self.train_word2vec(all_opcode_sequences)

        raw_embeddings = np.array([self.embed_sample(seq) for seq in all_opcode_sequences], dtype=np.float32)
        scaled_embeddings = self.scaler.fit_transform(raw_embeddings)
        return scaled_embeddings

    def transform(self, all_opcode_sequences: List[List[str]]) -> np.ndarray:
        """
        Transforms opcode sequences using pre-fitted Word2Vec and MinMaxScaler.
        """
        raw_embeddings = np.array([self.embed_sample(seq) for seq in all_opcode_sequences], dtype=np.float32)
        return self.scaler.transform(raw_embeddings)

    def save(self, filepath: str):
        if self.w2v_model:
            self.w2v_model.save(filepath)

    def load(self, filepath: str):
        self.w2v_model = Word2Vec.load(filepath)
