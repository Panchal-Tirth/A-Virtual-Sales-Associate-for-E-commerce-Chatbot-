from retrieval import search, tiered_search, bundle_search

def show(title, items):
    print(f"\n--- {title}")
    for m in items:
        print(f"  {m['title'][:52]:52} | {m['brand'][:12]:12} | ₹{m['price']:<5} | ★{m['rating']} | {m['season']:10} | {m['score']}")

# 1. Goa black shirts (product_type now filtered)
r = search("light breathable casual shirt for beach vacation",
           {"product_type": ["Shirt", "T-Shirt", "Polo T-Shirt"], "color": "Black", "season": "summer", "occasion": "beach/vacation"})
show("Goa black tops", r["items"]); print("relaxed:", r["relaxed"])

# 2. Budget tiers
t = tiered_search("comfortable jeans for daily wear", {"product_type": "Jeans", "max_price": 1200})
for tier, items in t.items(): show(f"Jeans [{tier}]", items)

# 3. Gym bundle
b = bundle_search([
    {"label": "Shoes",  "query": "running gym shoes",      "filters": {"product_type": "Sports Shoes"}},
    {"label": "T-shirt","query": "dri-fit gym t-shirt",    "filters": {"product_type": "T-Shirt", "occasion": "gym/sports"}},
    {"label": "Shorts", "query": "gym training shorts",    "filters": {"product_type": "Shorts", "occasion": "gym/sports"}},
    {"label": "Socks",  "query": "sports socks",           "filters": {"product_type": "Socks"}},
])
for label, items in b.items(): show(f"Gym bundle: {label}", items)

# 4. Impossible request -> graceful relaxation
r = search("pink formal shirt", {"product_type": "Shirt", "color": "Pink", "occasion": "formal/office", "max_price": 600})
show("Pink formal shirt <=600", r["items"]); print("relaxed:", r["relaxed"])