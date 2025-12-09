# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from model import GroceryCategoryModel
from recommender import GroceryRecommender
from healthy import HealthyRecommender
from reminder import ReminderSystem
from semantic_grouping import SemanticGrouping
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Initialize models ----------------
category_model = GroceryCategoryModel()
category_model.train("data.csv")

recommender = GroceryRecommender("data.csv")
healthy_recommender = HealthyRecommender()
reminder_system = ReminderSystem()
semantic_grouping = SemanticGrouping("data.csv")  


# ---------------- Pydantic request models ----------------
class ItemRequest(BaseModel):
    item: str

class ListRequest(BaseModel):
    items: list

# ---------------- API Endpoints ----------------
@app.post("/category")
def get_category(req: ItemRequest):
    cat = category_model.predict_category(req.item)
    return {"item": req.item, "category": cat}

@app.post("/missing-items")
def missing_items(req: ListRequest):
    result = recommender.recommend(req.items)
    return {"input_list": req.items, "missing_items": result}

@app.post("/healthy")
def healthy_alt(req: ItemRequest):
    alt_dict = healthy_recommender.suggest_healthier([req.item])
    return {"item": req.item, "alternative": alt_dict.get(req.item)}

@app.post("/analyze-item")
def analyze_item(req: ItemRequest):
    """
    Full AI understanding of new unseen items
    """
    category = category_model.predict_category(req.item)
    alt_dict = healthy_recommender.suggest_healthier([req.item])
    alternative = alt_dict.get(req.item)
    return {
        "item": req.item,
        "category": category,
        "healthy_alternative": alternative
    }

@app.post("/log-purchase")
def log_item_purchase(req: ItemRequest):
    """
    Log purchased item for future repurchase suggestions
    """
    reminder_system.log_purchase(req.item)
    return {"status": "success", "item_logged": req.item}

@app.post("/repurchase-suggestions")
def repurchase_suggestions(req: ListRequest):
    """
    Suggest items for repurchase based on history
    Excludes items already in current list
    """
    suggestions = reminder_system.check_reminders(req.items, top_n=5)
    healthier_alts = healthy_recommender.suggest_healthier(suggestions)
    return {
        "current_list": req.items,
        "repurchase_suggestions": suggestions,
        "healthy_alternatives": healthier_alts
    }


@app.post("/semantic-relatives")
def semantic_relatives(req: ItemRequest):
    """
    Find semantically related items to understand category relationships.
    Solves: "Greek yogurt", "milk", and "cheese" are related as dairy items.
    """
    relatives = semantic_grouping.find_semantic_relatives(req.item, top_k=5)
    category, confidence = semantic_grouping.find_category_by_semantic(req.item)
    return {
        "item": req.item,
        "category": category,
        "category_confidence": float(confidence),
        "semantic_relatives": [
            {"item": rel[0], "similarity": float(rel[1]), "category": rel[2]}
            for rel in relatives
        ]
    }


@app.post("/category-members")
def category_members(req: ItemRequest):
    """
    Find all items in the same category as the given item.
    For "Greek yogurt", returns: milk, cheese, butter, etc.
    """
    members = semantic_grouping.find_category_members(req.item, top_k=10)
    category, confidence = semantic_grouping.find_category_by_semantic(req.item)
    return {
        "item": req.item,
        "category": category,
        "category_confidence": float(confidence),
        "category_members": [
            {"item": m[0], "similarity": float(m[1]), "category": m[2]}
            for m in members
        ],
        "total_members_in_category": len(semantic_grouping.category_items.get(category, []))
    }


@app.post("/semantic-search")
def semantic_search(req: ItemRequest):
    """
    Search for items semantically similar to a query using vector embeddings.
    Returns items that are conceptually similar even if not exact matches.
    """
    results = semantic_grouping.semantic_search(req.item, top_k=10)
    return {
        "query": req.item,
        "results": [
            {"item": r[0], "similarity": float(r[1]), "category": r[2]}
            for r in results
        ]
    }


@app.post("/relate-items")
def relate_items(req: ListRequest):
    """
    Analyze semantic relationships between multiple items.
    Shows how items relate to each other and their shared categories.
    """
    relationships = semantic_grouping.relate_items(req.items)
    return relationships


@app.post("/category-profile")
def category_profile(req: ItemRequest):
    """
    Get semantic profile of a category.
    Shows what makes a category semantically cohesive.
    """
    category, confidence = semantic_grouping.find_category_by_semantic(req.item)
    profile = semantic_grouping.get_category_semantic_profile(category)
    
    if profile is None:
        return {"error": f"Category '{category}' not found"}
    
    return {
        "item": req.item,
        "category": category,
        "confidence": float(confidence),
        "profile": profile
    }


@app.post("/recommend-within-category")
def recommend_within_category(req: ItemRequest):
    """
    Recommend similar items within the same category.
    For "Greek yogurt", recommends other dairy products most similar to it.
    """
    recommendations = semantic_grouping.recommend_within_category(req.item, top_k=5)
    category, confidence = semantic_grouping.find_category_by_semantic(req.item)
    return {
        "item": req.item,
        "category": category,
        "category_confidence": float(confidence),
        "recommendations": [
            {"item": rec[0], "similarity": float(rec[1])}
            for rec in recommendations
        ]
    }

