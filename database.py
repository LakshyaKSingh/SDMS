"""
database.py
-----------
MongoDB backend for the Smart Donation Management System.
Falls back gracefully if MongoDB is not running.
"""
import os
import uuid
import datetime
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# ──────────────────────────────────────────────
# Connection
# ──────────────────────────────────────────────
MONGO_URI  = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME    = os.getenv("MONGO_DB",  "smart_donation_db")
COLLECTION = "donations"

_client = None
_db     = None


def _get_db():
    """Lazy-initialise MongoDB connection; returns None if unavailable."""
    global _client, _db
    if _db is not None:
        return _db
    try:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        _client.admin.command("ping")     # will raise if not reachable
        _db = _client[DB_NAME]
        # Create indexes
        _db[COLLECTION].create_index("receipt_id", unique=True)
        _db[COLLECTION].create_index([("timestamp", DESCENDING)])
        return _db
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"[MongoDB] Connection failed: {e}")
        _client = None
        _db     = None
        return None


def is_connected() -> bool:
    return _get_db() is not None


# ──────────────────────────────────────────────
# Receipt ID Generator
# ──────────────────────────────────────────────
def generate_receipt_id() -> str:
    now    = datetime.datetime.now()
    suffix = uuid.uuid4().hex[:6].upper()
    return f"REC-{now.strftime('%Y%m%d')}-{suffix}"


# ──────────────────────────────────────────────
# CRUD helpers
# ──────────────────────────────────────────────
def save_donation(
    receipt_id:   str,
    denomination: str,
    amount:       int,
    confidence:   float,
    image_path:   str | None = None,
) -> dict | None:
    """Insert a new donation document; returns the saved dict or None."""
    db = _get_db()
    if db is None:
        return None
    doc = {
        "receipt_id":   receipt_id,
        "denomination": denomination,
        "amount":       amount,
        "confidence":   round(confidence, 4),
        "image_path":   image_path,
        "timestamp":    datetime.datetime.now(),
    }
    db[COLLECTION].insert_one(doc)
    doc.pop("_id", None)   # strip ObjectId before returning
    return doc


def get_all_donations() -> list[dict]:
    """Return all donations, newest first, as plain dicts."""
    db = _get_db()
    if db is None:
        return []
    docs = list(db[COLLECTION].find({}, {"_id": 0}).sort("timestamp", DESCENDING))
    return docs


def get_recent_donations(limit: int = 10) -> list[dict]:
    db = _get_db()
    if db is None:
        return []
    return list(
        db[COLLECTION].find({}, {"_id": 0})
        .sort("timestamp", DESCENDING)
        .limit(limit)
    )


def get_donation_stats() -> dict:
    """Aggregate stats for the dashboard."""
    db = _get_db()
    if db is None:
        return {"total_amount": 0, "total_count": 0, "denominations": []}

    pipeline_total = [
        {"$group": {"_id": None,
                    "total_amount": {"$sum": "$amount"},
                    "total_count":  {"$sum": 1}}}
    ]
    pipeline_denom = [
        {"$group": {"_id": "$denomination",
                    "count":       {"$sum": 1},
                    "total_value": {"$sum": "$amount"}}},
        {"$sort": {"_id": 1}},
    ]

    total_result = list(db[COLLECTION].aggregate(pipeline_total))
    denom_result = list(db[COLLECTION].aggregate(pipeline_denom))

    total_amount = total_result[0]["total_amount"] if total_result else 0
    total_count  = total_result[0]["total_count"]  if total_result else 0

    denoms = [
        {"denomination": r["_id"], "count": r["count"], "total_value": r["total_value"]}
        for r in denom_result
    ]
    return {"total_amount": total_amount, "total_count": total_count, "denominations": denoms}


def clear_all_donations() -> bool:
    """Delete all documents from the donations collection."""
    db = _get_db()
    if db is None:
        return False
    db[COLLECTION].delete_many({})
    return True
