# semantic_grouping.py
# Semantic Understanding and Category-Aware Grouping Module
# Solves: "No semantic understanding" problem by relating similar items within categories

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticGrouping:
    """
    Provides semantic understanding by grouping items within categories and finding
    semantically related items. Addresses the problem of not relating "Greek yogurt",
    "milk", and "cheese" even though they belong to the dairy category.
    """
    
    def __init__(self, data_csv_path="data.csv", model_name="all-MiniLM-L6-v2"):
        """
        Initialize the semantic grouping engine with pre-computed embeddings and category mappings.
        
        Args:
            data_csv_path: Path to the CSV file containing items and categories
            model_name: Sentence-BERT model name for semantic embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.data_csv_path = data_csv_path
        
        # Load data and create embeddings
        self.df = pd.read_csv(data_csv_path)
        self.items = self.df['item'].tolist()
        self.categories = self.df['category'].tolist()
        
        # Pre-compute embeddings for all items (384-dimensional vectors)
        self.embeddings = self.model.encode(self.items, convert_to_numpy=True)
        
        # Create category-to-items mapping for fast lookup
        self.category_items = {}
        self.category_embeddings = {}
        for item, category in zip(self.items, self.categories):
            if category not in self.category_items:
                self.category_items[category] = []
            self.category_items[category].append(item)
        
        # Pre-compute category embeddings (mean of all items in category)
        for category in self.category_items.keys():
            category_indices = [i for i, c in enumerate(self.categories) if c == category]
            category_embedding = np.mean(self.embeddings[category_indices], axis=0)
            self.category_embeddings[category] = category_embedding
    
    def find_semantic_relatives(self, item, top_k=5):
        """
        Find semantically related items (items that are similar even if not identical).
        For example, "Greek yogurt" relates to "milk", "cheese", "butter" (dairy relatives).
        
        Args:
            item: The item to find relatives for
            top_k: Number of related items to return (excluding the item itself)
            
        Returns:
            List of tuples: [(related_item, similarity_score, category), ...]
        """
        # Encode the input item
        item_embedding = self.model.encode([item], convert_to_numpy=True)[0]
        
        # Calculate similarity with all items
        similarities = cosine_similarity([item_embedding], self.embeddings)[0]
        
        # Get top k+1 (to exclude the exact item or very similar duplicates)
        top_indices = np.argsort(similarities)[::-1][:top_k + 5]
        
        relatives = []
        seen_items = set()
        
        for idx in top_indices:
            related_item = self.items[idx]
            similarity = float(similarities[idx])
            category = self.categories[idx]
            
            # Skip the exact item or very close duplicates
            if related_item.lower() == item.lower() or related_item in seen_items:
                continue
            
            seen_items.add(related_item)
            relatives.append((related_item, similarity, category))
            
            if len(relatives) == top_k:
                break
        
        return relatives
    
    def find_category_members(self, item, top_k=5):
        """
        Find items in the same category as the given item.
        For example, for "Greek yogurt", returns other dairy items like "milk", "cheese", "butter".
        
        Args:
            item: The item to find category members for
            top_k: Number of category members to return
            
        Returns:
            List of tuples: [(category_item, similarity_to_input, category), ...]
        """
        # Find the category of the input item
        item_embedding = self.model.encode([item], convert_to_numpy=True)[0]
        
        # Get similarities with all items
        similarities = cosine_similarity([item_embedding], self.embeddings)[0]
        
        # Find the most similar item to get its category
        most_similar_idx = np.argmax(similarities)
        item_category = self.categories[most_similar_idx]
        
        # Get all items in that category
        category_indices = [i for i, c in enumerate(self.categories) if c == item_category]
        category_similarities = [(self.items[i], similarities[i], self.categories[i]) 
                                 for i in category_indices]
        
        # Sort by similarity and remove the input item itself
        category_similarities.sort(key=lambda x: x[1], reverse=True)
        category_members = [m for m in category_similarities 
                           if m[0].lower() != item.lower()][:top_k]
        
        return category_members
    
    def find_category_by_semantic(self, item):
        """
        Find the category of an item using semantic similarity.
        More robust than simple keyword matching.
        
        Args:
            item: The item to categorize
            
        Returns:
            Tuple: (category, confidence_score)
        """
        # Encode the item
        item_embedding = self.model.encode([item], convert_to_numpy=True)[0]
        
        # Find most similar item
        similarities = cosine_similarity([item_embedding], self.embeddings)[0]
        most_similar_idx = np.argmax(similarities)
        
        category = self.categories[most_similar_idx]
        confidence = float(similarities[most_similar_idx])
        
        return category, confidence
    
    def get_category_semantic_profile(self, category):
        """
        Get the semantic profile of a category (mean embedding of all items in category).
        Useful for understanding what makes a category semantically cohesive.
        
        Args:
            category: The category name
            
        Returns:
            Dictionary with category info and statistics
        """
        if category not in self.category_items:
            return None
        
        items = self.category_items[category]
        category_embedding = self.category_embeddings[category]
        
        # Calculate average similarity within category
        category_indices = [i for i, c in enumerate(self.categories) if c == category]
        similarities = cosine_similarity([category_embedding], 
                                        self.embeddings[category_indices])[0]
        
        return {
            "category": category,
            "item_count": len(items),
            "items": items[:10],  # First 10 items
            "avg_semantic_cohesion": float(np.mean(similarities)),
            "semantic_embedding_dim": len(category_embedding)
        }
    
    def recommend_within_category(self, item, top_k=5):
        """
        Recommend similar items within the same semantic category.
        For "Greek yogurt", recommends other dairy items most similar to it.
        
        Args:
            item: The item to find recommendations for
            top_k: Number of recommendations
            
        Returns:
            List of tuples: [(recommended_item, similarity_score), ...]
        """
        # Encode the input item
        item_embedding = self.model.encode([item], convert_to_numpy=True)[0]
        
        # Find its category
        item_category, _ = self.find_category_by_semantic(item)
        
        # Get all items in that category
        category_indices = [i for i, c in enumerate(self.categories) if c == item_category]
        
        # Calculate similarities with category items only
        similarities = cosine_similarity([item_embedding], 
                                        self.embeddings[category_indices])[0]
        
        # Get top k recommendations from the category
        category_top_indices = np.argsort(similarities)[::-1]
        
        recommendations = []
        for idx in category_top_indices:
            actual_idx = category_indices[idx]
            recommended_item = self.items[actual_idx]
            similarity = float(similarities[idx])
            
            # Skip the exact item
            if recommended_item.lower() != item.lower():
                recommendations.append((recommended_item, similarity))
            
            if len(recommendations) == top_k:
                break
        
        return recommendations
    
    def semantic_search(self, query, top_k=10, filter_category=None):
        """
        Semantic search to find items semantically similar to a query.
        Can optionally filter by category.
        
        Args:
            query: Search query (can be any text, not just exact item names)
            top_k: Number of results
            filter_category: Optional category to filter results
            
        Returns:
            List of tuples: [(item, similarity_score, category), ...]
        """
        # Encode the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get all items with their similarities
        results = [(self.items[i], similarities[i], self.categories[i]) 
                   for i in range(len(self.items))]
        
        # Filter by category if specified
        if filter_category:
            results = [r for r in results if r[2] == filter_category]
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def relate_items(self, items_list):
        """
        Analyze relationships between multiple items.
        Shows which items are semantically related and their categories.
        
        Args:
            items_list: List of items to relate
            
        Returns:
            Dictionary with relationship information
        """
        if not items_list or len(items_list) == 0:
            return {"error": "Empty items list"}
        
        # Encode all items
        embeddings = self.model.encode(items_list, convert_to_numpy=True)
        
        # Get categories for all items
        categories = []
        for item in items_list:
            category, _ = self.find_category_by_semantic(item)
            categories.append(category)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Group by category
        category_groups = {}
        for item, category in zip(items_list, categories):
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(item)
        
        # Calculate semantic cohesion (average similarity within items)
        all_similarities = similarities[np.triu_indices_from(similarities, k=1)]
        avg_cohesion = float(np.mean(all_similarities)) if len(all_similarities) > 0 else 0
        
        return {
            "items": items_list,
            "categories": categories,
            "category_groups": category_groups,
            "pairwise_similarities": similarities.tolist(),
            "average_semantic_cohesion": avg_cohesion,
            "unique_categories": list(set(categories))
        }
