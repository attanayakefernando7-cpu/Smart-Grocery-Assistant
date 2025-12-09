# recommender.py
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from healthy import HealthyRecommender
from reminder import ReminderSystem

class GroceryRecommender:
    def __init__(self, data_path="data.csv", model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.data = pd.read_csv(data_path)
        self.items = self.data["item"].astype(str).tolist()
        self.item_array = np.array(self.items, dtype=object)
        self.item_embeddings = self.model.encode(self.items, convert_to_numpy=True)
        self.healthy = HealthyRecommender(model_name)
        self.reminder = ReminderSystem()

    def predict_missing_items(self, purchased_items, top_k=5):
        purchased = purchased_items or []
        purchased_vecs = self.model.encode(purchased, convert_to_numpy=True) if len(purchased) else np.zeros((0, self.item_embeddings.shape[1]))
        mean_vec = np.mean(purchased_vecs, axis=0) if purchased_vecs.shape[0] else np.zeros(self.item_embeddings.shape[1])
        norms_items = np.linalg.norm(self.item_embeddings, axis=1)
        norm_mean = np.linalg.norm(mean_vec) + 1e-12
        scores = (self.item_embeddings @ mean_vec) / (norms_items * norm_mean + 1e-12)
        ranked_idx = np.argsort(scores)[::-1]
        purchased_arr = np.array(purchased, dtype=object)
        isin_mask = np.isin(self.item_array[ranked_idx], purchased_arr) if purchased_arr.size else np.zeros_like(ranked_idx, dtype=bool)
        candidate_idx = ranked_idx[~isin_mask]
        selected = candidate_idx[:top_k]
        return self.item_array[selected].tolist()

    def recommend(self, purchased_items):
        predicted = self.predict_missing_items(purchased_items, top_k=5)
        healthier = self.healthy.suggest_healthier(purchased_items)
        reminders = self.reminder.check_reminders(purchased_items, top_n=5)
        return {
            "predicted_items": predicted,
            "healthier_alternatives": healthier,
            "expiring_soon": reminders
        }
