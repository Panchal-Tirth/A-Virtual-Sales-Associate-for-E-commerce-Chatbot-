import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_DIR, COLLECTION, EMBED_MODEL

col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION)
model = SentenceTransformer(EMBED_MODEL)

def show(query, where=None, n=5):
    q = model.encode([query], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q, n_results=n, where=where)
    print(f"\n>>> {query} | filter={where}")
    for m, d in zip(res['metadatas'][0], res['distances'][0]):
        print(f"  {m['title'][:55]:55} | {m['brand'][:14]:14} | ₹{m['price']:<5} | ★{m['rating']} | {m['season']} | {m['occasions']} | d={d:.2f}")

show("light breathable black shirt for beach vacation",
     where={"$and": [{"color": "Black"}, {"price": {"$lte": 1000}}]})
show("comfortable shoes for gym workout running")
show("warm jacket for winter", where={"price": {"$lte": 1500}})
show("formal shirt for office", where={"product_type": "Shirt"})