import pandas as pd, chromadb
from sentence_transformers import SentenceTransformer
from config import CATALOG_PATH, CHROMA_DIR, COLLECTION, EMBED_MODEL, ROOT

LAPTOP_PATH = ROOT / "data" / "catalog_laptops.parquet"

fashion = pd.read_parquet(CATALOG_PATH)
fashion["domain"] = "fashion"
frames = [fashion]
if LAPTOP_PATH.exists():
    frames.append(pd.read_parquet(LAPTOP_PATH))
df = pd.concat(frames, ignore_index=True)
print("Loaded", df.shape, df["domain"].value_counts().to_dict())

STR_COLS = ["pid", "domain", "title", "brand", "product_type", "gender", "color", "color_all", "fabric_group",
            "fit", "sleeve", "pattern", "occasions", "season", "image_url", "url", "description",
            "laptop_type", "cpu", "cpu_tier", "gpu", "gpu_type", "os", "resolution"]
INT_COLS = ["pack_of", "discount_pct", "cpu_gen", "refresh_hz", "ram_gb", "storage_gb"]
FLT_COLS = ["screen_inch", "weight_kg", "battery_hrs"]

for c in STR_COLS:
    if c not in df: df[c] = ""
    df[c] = df[c].fillna("").astype(str)
for c in INT_COLS:
    df[c] = pd.to_numeric(df[c] if c in df else 0, errors="coerce").fillna(0).astype(int)
for c in FLT_COLS:
    df[c] = pd.to_numeric(df[c] if c in df else 0, errors="coerce").fillna(0.0).astype(float)
df["description"] = df["description"].str[:300]
df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0).astype(float)
df["rating_known"] = df["rating_known"].fillna(False).astype(bool)

def meta(r):
    return {
        "pid": r.pid, "domain": r.domain, "title": r.title, "brand": r.brand, "product_type": r.product_type,
        "gender": r.gender, "color": r.color, "color_all": r.color_all, "fabric_group": r.fabric_group,
        "fit": r.fit, "sleeve": r.sleeve, "pattern": r.pattern, "occasions": r.occasions, "season": r.season,
        "price": int(r.selling_price), "mrp": int(r.actual_price), "discount_pct": int(r.discount_pct),
        "rating": float(r.rating), "rating_known": bool(r.rating_known), "pack_of": int(r.pack_of),
        "image_url": r.image_url, "url": r.url, "description": r.description,
        "laptop_type": r.laptop_type, "cpu": r.cpu, "cpu_tier": r.cpu_tier, "cpu_gen": int(r.cpu_gen),
        "gpu": r.gpu, "gpu_type": r.gpu_type, "os": r.os, "resolution": r.resolution,
        "refresh_hz": int(r.refresh_hz), "ram_gb": int(r.ram_gb), "storage_gb": int(r.storage_gb),
        "screen_inch": float(r.screen_inch), "weight_kg": float(r.weight_kg), "battery_hrs": float(r.battery_hrs),
    }

metadatas = [meta(r) for r in df.itertuples()]
ids = [str(i) for i in range(len(df))]
docs = df["search_text"].tolist()

print("Embedding... (a few minutes on CPU)")
model = SentenceTransformer(EMBED_MODEL)
embs = model.encode(docs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

client = chromadb.PersistentClient(path=CHROMA_DIR)
try: client.delete_collection(COLLECTION)
except Exception: pass
col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

B = 1000
for i in range(0, len(ids), B):
    col.add(ids=ids[i:i+B], documents=docs[i:i+B], embeddings=embs[i:i+B].tolist(), metadatas=metadatas[i:i+B])
print("Indexed:", col.count())