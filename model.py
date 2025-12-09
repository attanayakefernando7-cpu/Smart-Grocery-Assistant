import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

class GroceryCategoryModel:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.clf = LogisticRegression(max_iter=1000, solver='lbfgs')
        self.trained = False

    def embed(self, text: str):
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            output = self.encoder(**tokens)
        return output.last_hidden_state[:, 0, :].squeeze().numpy()

    def train(self, csv_path="data.csv"):
        df = pd.read_csv(csv_path)
        X = np.array([self.embed(t) for t in df["item"]])
        y = df["category"]

        self.clf.fit(X, y)
        self.trained = True
        print("Category model trained!")

    def predict_category(self, item: str):
        if not self.trained:
            raise ValueError("Model not trained!")
        vector = self.embed(item)
        return self.clf.predict([vector])[0]
