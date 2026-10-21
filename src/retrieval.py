import re
import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_DIR, COLLECTION, EMBED_MODEL

_model, _col = None, None
DEFAULT_GENDER = ["Men", "Unisex", "Unspecified"]   # fashion catalog is ~97% men's; laptops are "Unisex"


def _init():
    global _model, _col
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
        _col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION)


def _as_list(x):
    return x if isinstance(x, list) else [x]


def _where(f):
    """Hard constraints -> Chroma metadata filter."""
    c = [{"gender": {"$in": _as_list(f.get("gender") or DEFAULT_GENDER)}}]
    if f.get("domain"):        c.append({"domain": f["domain"]})
    if f.get("product_type"):  c.append({"product_type": {"$in": _as_list(f["product_type"])}})
    if f.get("color"):         c.append({"color": {"$in": _as_list(f["color"])}})
    if f.get("brand"):         c.append({"brand": {"$in": _as_list(f["brand"])}})
    if f.get("season"):        c.append({"season": {"$in": [f["season"], "all-season"]}})
    if f.get("max_price"):     c.append({"price": {"$lte": int(f["max_price"])}})
    if f.get("min_price"):     c.append({"price": {"$gte": int(f["min_price"])}})
    if f.get("min_rating"):    c.append({"rating": {"$gte": float(f["min_rating"])}})
    # laptop specs
    if f.get("min_ram_gb"):      c.append({"ram_gb": {"$gte": int(f["min_ram_gb"])}})
    if f.get("min_storage_gb"):  c.append({"storage_gb": {"$gte": int(f["min_storage_gb"])}})
    if f.get("cpu_tier"):        c.append({"cpu_tier": {"$in": _as_list(f["cpu_tier"])}})
    if f.get("gpu_type"):        c.append({"gpu_type": f["gpu_type"]})
    if f.get("max_weight_kg"):   # weight 0 means unknown, so require a real value
        c.append({"weight_kg": {"$gte": 0.5}})
        c.append({"weight_kg": {"$lte": float(f["max_weight_kg"])}})
    return c[0] if len(c) == 1 else {"$and": c}


def _score(m, dist, f):
    sim = 1 - dist
    r = min(m["rating"] if m["rating_known"] else 3.5, 4.7)   # cap: no review counts in data
    s = 0.50 * sim + 0.25 * (r - 1) / 4 + 0.05 * min(m["discount_pct"], 70) / 70
    if f.get("occasion") and f["occasion"] in m["occasions"]: s += 0.10
    if f.get("season") and m["season"] == f["season"]:        s += 0.10
    if r < 3.0: s -= 0.30                                      # don't push poorly rated items
    return s


def _diversify(cands, k, max_per_brand=2):
    out, seen, brands = [], set(), {}
    for c in cands:
        key = (c["brand"], re.sub(r"[^a-z]", "", c["title"].lower()), c["color"])
        if key in seen or brands.get(c["brand"], 0) >= max_per_brand:
            continue
        seen.add(key); brands[c["brand"]] = brands.get(c["brand"], 0) + 1
        out.append(c)
        if len(out) == k: break
    return out


def _query(query, f, pool):
    _init()
    q = _model.encode([query], normalize_embeddings=True).tolist()
    res = _col.query(query_embeddings=q, n_results=pool, where=_where(f))
    cands = []
    for m, d in zip(res["metadatas"][0], res["distances"][0]):
        item = dict(m); item["score"] = round(_score(m, d, f), 4)
        cands.append(item)
    cands.sort(key=lambda x: -x["score"])
    return cands


# Soft constraints are dropped first when there are too few results. Price, type, RAM, CPU and GPU never are.
RELAX_ORDER = ["min_rating", "season", "brand", "color", "max_weight_kg", "min_storage_gb"]


def search(query, filters=None, k=5, pool=80):
    """Returns {'items': [...], 'relaxed': [constraints we had to drop]}"""
    f, relaxed = dict(filters or {}), []
    items = _diversify(_query(query, f, pool), k)
    for key in RELAX_ORDER:
        if len(items) >= k or key not in f: continue
        f.pop(key); relaxed.append(key)
        items = _diversify(_query(query, f, pool), k)
    return {"items": items, "relaxed": relaxed}


def tiered_search(query, filters=None, k=3):
    """Budget / best-value / premium alternatives."""
    f = dict(filters or {})
    mp = f.get("max_price")
    if mp:
        lo = dict(f, max_price=int(mp * 0.6))
        mid = dict(f, min_price=int(mp * 0.6) + 1)
        hi = dict(f, min_price=int(mp) + 1, max_price=int(mp * 1.6))
    else:
        pool = _query(query, f, 120)
        prices = sorted(c["price"] for c in pool)
        if not prices: return {"budget": [], "best_value": [], "premium": []}
        t1, t2 = prices[len(prices) // 3], prices[2 * len(prices) // 3]
        lo, mid, hi = dict(f, max_price=t1), dict(f, min_price=t1 + 1, max_price=t2), dict(f, min_price=t2 + 1)
    return {"budget": search(query, lo, k)["items"],
            "best_value": search(query, mid, k)["items"],
            "premium": search(query, hi, k)["items"]}


def bundle_search(parts, k=2):
    """parts: [{'label': 'Shoes', 'query': '...', 'filters': {...}}, ...] -> {label: items}"""
    return {p["label"]: search(p["query"], p.get("filters"), k)["items"] for p in parts}