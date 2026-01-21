from gptcache.similarity_evaluation import SimilarityEvaluation

class VerboseSimilarityEvaluation(SimilarityEvaluation):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, similarities):
        """
        similarities: List[float], cosine similarities from FAISS
        """
        if not similarities:
            print("[GPTCache] MISS (empty index)")
            return False

        max_sim = max(similarities)
        hit = max_sim >= self.threshold

        print(
            f"[GPTCache] similarity={max_sim:.4f} "
            f"threshold={self.threshold} "
            f"=> {'HIT' if hit else 'MISS'}"
        )

        return hit
    
from gptcache.manager import VectorBase as BaseVectorBase
import numpy as np

class VerboseFaissVectorBase(BaseVectorBase):
    def search(self, embedding, top_k=1):
        """
        Return:
          - ids
          - similarities (cosine similarity)
        """
        ids, similarities = super().search(embedding, top_k)

        if similarities is None or len(similarities) == 0:
            print("[GPTCache] MISS | empty index")
            return ids, similarities

        max_sim = float(np.max(similarities))
        hit = max_sim >= self.config.similarity_threshold

        print(
            f"[GPTCache] similarity={max_sim:.4f} "
            f"threshold={self.config.similarity_threshold} "
            f"=> {'HIT' if hit else 'MISS'}"
        )

        return ids, similarities