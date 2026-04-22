from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ResumeMatcher:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the matcher with a pre-trained SentenceTransformer model.
        """
        print(f"Loading SentenceTransformer model '{model_name}'...")
        self.model = SentenceTransformer(model_name)

    def compute_similarity(self, job_desc: str, resumes: list[str]) -> np.ndarray:
        """
        Computes cosine similarity between a single job description and a list of resumes.
        
        Args:
            job_desc (str): The cleaned text of the job description.
            resumes (list[str]): A list of cleaned texts of resumes.
            
        Returns:
            np.ndarray: A 1D array of similarity scores (0 to 1).
        """
        # Encode texts to get their embeddings
        job_embedding = self.model.encode([job_desc])
        resume_embeddings = self.model.encode(resumes)
        
        # Compute cosine similarities
        similarities = cosine_similarity(job_embedding, resume_embeddings)[0]
        
        return similarities

    def rank_resumes(self, job_desc: str, resumes: list[str], resume_ids: list[str]) -> list[dict]:
        """
        Ranks resumes based on their similarity to the job description.
        
        Args:
            job_desc (str): Job description text.
            resumes (list[str]): List of resume texts.
            resume_ids (list[str]): Identifiers for the resumes.
            
        Returns:
            list[dict]: A sorted list of dictionaries containing the rank, id, score, and text.
        """
        scores = self.compute_similarity(job_desc, resumes)
        
        results = []
        for r_id, score, text in zip(resume_ids, scores, resumes):
            results.append({
                "id": r_id,
                "score": float(score),
                "text": text
            })
            
        # Sort descending by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
