# healthy.py
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

HEALTHY_MAP = {
    "rice": "brown rice",
    "bread": "whole grain bread",
    "wheat flour": "gram flour",
    "sugar": "honey",
    "butter": "olive oil",
    "margarine": "butter",
    "chips": "nuts",
    "ice cream": "yogurt",
    "noodles": "whole wheat noodles",
    "fried chicken": "grilled chicken",
    "soda": "water",
    "jam": "fruit puree",
    "biscuits": "fruits"
}

class HealthyRecommender:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.keys = list(HEALTHY_MAP.keys())
        self.alternatives = [HEALTHY_MAP[k] for k in self.keys]
        self.key_embeddings = self.model.encode(self.keys, convert_to_numpy=True)

    def suggest_healthier(self, items: List[str]) -> Dict[str, str]:
        items_arr = items or []
        item_vecs = self.model.encode(items_arr, convert_to_numpy=True) if len(items_arr) else np.zeros((0, self.key_embeddings.shape[1]))
        sim_matrix = np.zeros((len(items_arr), len(self.keys))) + (item_vecs.shape[0] == 0)
        sim_matrix = sim_matrix if item_vecs.shape[0] == 0 else (item_vecs @ self.key_embeddings.T) / (np.linalg.norm(item_vecs, axis=1, keepdims=True) * np.linalg.norm(self.key_embeddings, axis=1))
        best_idxs = np.argmax(sim_matrix, axis=1) if sim_matrix.size else np.array([], dtype=int)
        return {items_arr[i]: self.alternatives[int(best_idxs[i])] for i in range(len(items_arr))}
