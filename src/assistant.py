import json, statistics
from llm import chat_json, chat_text
from config import LLM_FAST
from retrieval import search, tiered_search, bundle_search

FASHION_TYPES = ['T-Shirt', 'Polo T-Shirt', 'Shirt', 'Track Pants', 'Kurta/Ethnic', 'Innerwear', 'Trousers',
                 'Shorts', 'Jeans', 'Jacket', 'Socks', 'Sports Shoes', 'Cap/Hat', 'Sweater', 'Tie/Cufflink',
                 'Blazer/Suit', 'Sleepwear', 'Tracksuit', 'Sneakers', 'Handkerchief', 'Scarf/Muffler/Gloves',
                 'Raincoat', 'Sandals/Slippers', 'Formal Shoes', 'Boots']
COLORS = ['Black', 'White', 'Blue', 'Grey', 'Green', 'Red', 'Yellow', 'Maroon', 'Brown', 'Beige', 'Orange',
          'Pink', 'Purple', 'Gold', 'Silver', 'Teal', 'Multicolor']
CPU_TIERS = ["Entry", "Mid", "High", "Ultra"]
FASHION_OCC = ['casual', 'gym/sports', 'formal/office', 'party/festive', 'beach/vacation']
LAPTOP_OCC = ['gaming', 'programming/engineering', 'student/everyday', 'creative/editing', 'business/office', 'travel/portable']
DOMAINS = {
    "fashion": {"types": FASHION_TYPES, "colors": COLORS, "occasions": FASHION_OCC, "seasons": ['summer', 'winter']},
    "laptop":  {"types": ["Laptop"], "colors": [], "occasions": LAPTOP_OCC, "seasons": []},
}
LAPTOP_BRANDS = {"asus": "ASUS", "hp": "HP", "msi": "MSI", "dell": "Dell", "lenovo": "Lenovo", "acer": "Acer",
                 "apple": "Apple", "realme": "realme", "infinix": "Infinix", "vaio": "VAIO", "avita": "Avita",
                 "nokia": "Nokia", "samsung": "Samsung", "mi": "Mi", "lg": "LG", "gigabyte": "GIGABYTE",
                 "razer": "Razer", "microsoft": "Microsoft", "honor": "Honor", "chuwi": "CHUWI"}
TYPE_MEDIAN = {'T-Shirt': 466, 'Polo T-Shirt': 597, 'Shirt': 799, 'Track Pants': 774, 'Shorts': 649, 'Jeans': 1299,
               'Jacket': 1442, 'Sports Shoes': 810, 'Socks': 300, 'Trousers': 879, 'Sweater': 999, 'Cap/Hat': 374,
               'Laptop': 60000}

UNDERSTAND_SYSTEM = f"""You are the understanding module of a shopping assistant. The store sells only two things:
(1) MEN'S FASHION (clothing, innerwear, socks, caps, ties, a few shoes) and (2) LAPTOPS.
It has NO women's or kids' wear, no phones, tablets, TVs, other electronics, cosmetics, bottles or bags.

Return ONLY a JSON object with these keys:
{{
 "intent": "recommend" | "compare" | "bundle" | "advice" | "chitchat" | "out_of_scope",
 "domain": "fashion" | "laptop" | null,
 "is_refinement": true if the message modifies/answers the previous request (e.g. "cheaper", "in blue", "with 16GB", an answer to a clarifying question), else false,
 "product_type": list. Fashion: subset of {FASHION_TYPES}. Laptop: always ["Laptop"],
 "color": one of {COLORS} or null (fashion only),
 "occasion": fashion: one of {FASHION_OCC}; laptop: one of {LAPTOP_OCC}; or null,
 "season": "summer" | "winter" | null (fashion only),
 "max_price": number (INR) or null, "min_price": number or null,
 "brand": string or null,
 "min_ram_gb": number or null, "min_storage_gb": number or null,
 "cpu_tier": list, subset of {CPU_TIERS} (laptops only, else []),
 "gpu_type": "Dedicated" | "Integrated" | null,
 "max_weight_kg": number or null,
 "search_query": a full descriptive search phrase reflecting ALL current requirements,
 "price_adjust": "cheaper" | "premium" | null,
 "want_tiers": true if user asks for options/alternatives across budgets,
 "compare_targets": list of 1-based positions from "previously_shown" (ints) or short product descriptions (strings),
 "bundle_parts": [{{"label": "Shoes", "domain": "fashion" | "laptop", "query": "...", "product_type": [..], "occasion": ..., "min_ram_gb": ...}}],
 "followup_question": one short question or null,
 "out_of_scope_reason": string or null
}}

Rules:
- Fashion: "shirt" -> ["Shirt"]; "t-shirt/tee" -> ["T-Shirt","Polo T-Shirt"]; "tops" -> ["Shirt","T-Shirt","Polo T-Shirt"].
- Translate context into attributes: Goa/beach trip -> season summer, occasion beach/vacation; gym -> gym/sports; winter/hill-station -> season winter; office/interview -> formal/office; college -> casual.
- Laptops: coding/programming/Computer Engineering/CS -> occasion programming/engineering, cpu_tier ["Mid","High","Ultra"], min_ram_gb 8 (16 if virtualization, VMs, Docker, heavy projects or "best"), min_storage_gb 256.
  Gaming -> occasion gaming, gpu_type Dedicated. Video editing/design/3D -> creative/editing, min_ram_gb 16.
  Light/portable/travel -> max_weight_kg 1.5. Basic use/student/browsing -> student/everyday. Office/business -> business/office.
- Questions about general product knowledge ("Is 8GB RAM enough?", "i5 or i7?", "what does SSD do?") -> intent "advice".
- Lifestyle requests needing several items (gym, trip, wardrobe, college setup) -> intent "bundle" with 4-6 bundle_parts, each with its own domain. Gym example: Shoes (Sports Shoes), T-shirt, Shorts, Track Pants, Socks, plus "Water bottle" and "Gym bag" with an empty product_type. Beach trip: T-shirts, Shirts, Shorts, Sandals/Slippers, Cap/Hat. Include parts we don't sell with an empty product_type so we can tell the user honestly.
- Budgets in INR: "under 2k" = 2000, "50k" = 50000.
- Women's/kids' wear, phones, other electronics, non-fashion non-laptop -> intent "out_of_scope" with a reason.
- Ask a followup_question ONLY if intent is recommend and the product category is impossible to infer (e.g. "I need something to wear"), and the previous assistant message was not already a question. Otherwise null. Never ask about things you can sensibly assume.
- If the last assistant message was a clarifying question, the user is answering it: is_refinement = true.
"""

GEN_SYSTEM = """You are Style Assistant, a friendly, knowledgeable sales associate at an online store selling men's fashion and laptops.
Rules:
- Recommend ONLY items from PRODUCTS, referring to them as [number]. Never invent products, prices, ratings or specs.
- Cover EVERY product in PRODUCTS with one short line each. Say WHY it fits the need. For fashion use fabric, fit, sleeve, season, occasion, rating, discount. For laptops translate specs into plain meaning (e.g. "16GB RAM and a Core i5 handle IDEs and virtual machines comfortably", "integrated graphics are fine for coding but not gaming"). Show prices as ₹.
- Compare each product against USER_FILTERS. If a shown item does not match what was asked (wrong colour, below the requested RAM, over budget in a non-premium tier), say so plainly for that item.
- When recommending laptops, mention once that laptop prices are from 2022 listings and may have changed.
- If NOTES mention relaxed filters, items we don't carry, or that a premium tier is intentionally above budget, say so honestly in one sentence.
- No markdown tables, no emojis. Be concise: a short intro, the product lines, then one helpful next step.
- Follow TASK exactly."""

SCOPE_SYSTEM = """You are Style Assistant for an online store. Reply in 2-3 short sentences, warm and honest.
What we sell: men's fashion (t-shirts, polos, shirts, jeans, trousers, shorts, track pants, jackets, sweaters, kurtas, innerwear, socks, caps, ties, sports shoes, sneakers, sandals) and laptops.
- For greetings only: greet and say you can recommend products, compare them, build outfits or bundles, advise on laptop specs and show budget alternatives.
- If asked for something we don't carry (women's or kids' wear, phones, tablets, other electronics, etc.), say clearly what the catalog is and offer ONE nearby thing from the list above. Never suggest anything outside that list. Never pretend to stock anything else."""

ADVICE_SYSTEM = """You are Style Assistant, a knowledgeable sales associate for an online store selling men's fashion and laptops.
Answer the buying-knowledge question in at most 5 short sentences, with a clear practical verdict, using well-established general knowledge (e.g. RAM needs, CPU tiers, SSD vs HDD, fabric choices).
Do not invent specific products or prices. End by offering to show matching products from the catalog."""


def _pick(val, allowed):
    if not isinstance(val, str): return None
    return next((a for a in allowed if a.lower() == val.strip().lower()), None)

def _num(v):
    try:
        v = float(v)
        return int(v) if v > 0 else None
    except Exception:
        return None

def _fnum(v):
    try:
        v = float(v)
        return v if v > 0 else None
    except Exception:
        return None

def _as_list(x):
    return [x] if isinstance(x, str) else (x if isinstance(x, list) else [])

def _clean_slots(u, domain):
    cfg, s = DOMAINS[domain], {"domain": domain}
    pt = [p for p in (_pick(x, cfg["types"]) for x in _as_list(u.get("product_type"))) if p]
    if domain == "laptop": pt = ["Laptop"]
    if pt: s["product_type"] = pt
    for key, allowed in (("color", cfg["colors"]), ("occasion", cfg["occasions"]), ("season", cfg["seasons"])):
        if allowed:
            v = _pick(u.get(key), allowed)
            if v: s[key] = v
    for key in ("max_price", "min_price"):
        v = _num(u.get(key))
        if v: s[key] = v
    if isinstance(u.get("brand"), str) and u["brand"].strip():
        b = u["brand"].strip()
        s["brand"] = LAPTOP_BRANDS.get(b.lower(), b.title()) if domain == "laptop" else b.title()
    if domain == "laptop":
        for key in ("min_ram_gb", "min_storage_gb"):
            v = _num(u.get(key))
            if v: s[key] = v
        tiers = [t for t in (_pick(x, CPU_TIERS) for x in _as_list(u.get("cpu_tier"))) if t]
        if tiers: s["cpu_tier"] = tiers
        g = _pick(u.get("gpu_type"), ["Dedicated", "Integrated"])
        if g: s["gpu_type"] = g
        w = _fnum(u.get("max_weight_kg"))
        if w: s["max_weight_kg"] = w
    return s


class Assistant:
    def __init__(self):
        self.history, self.slots, self.last_items = [], {}, []
        self.last_query, self.asked_followup = "", False

    # ---------- public ----------
    def reply(self, message):
        u = self._understand(message)
        intent = u.get("intent", "recommend")
        if intent in ("chitchat", "out_of_scope"): out = self._scope(message, u)
        elif intent == "advice":                   out = self._advice(message)
        elif intent == "compare":                  out = self._compare(message, u)
        elif intent == "bundle":                   out = self._bundle(message, u)
        else:                                      out = self._recommend(message, u)
        self.history += [{"role": "user", "content": message}, {"role": "assistant", "content": out["text"]}]
        return out

    # ---------- steps ----------
    def _understand(self, message):
        ctx = {"previous_filters": self.slots, "previous_query": self.last_query,
               "previously_shown": [f"{i+1}. {m['title']} ({m['brand']}, ₹{m['price']})" for i, m in enumerate(self.last_items)],
               "last_assistant_message_was_a_question": self.asked_followup}
        msgs = self.history[-6:] + [{"role": "user", "content": json.dumps({"context": ctx, "message": message}, ensure_ascii=False)}]
        u = chat_json(UNDERSTAND_SYSTEM, msgs)
        return u if isinstance(u, dict) else {}

    def _domain(self, u):
        if u.get("domain") in DOMAINS: return u["domain"]
        if u.get("is_refinement") and self.slots.get("domain") in DOMAINS: return self.slots["domain"]
        return "fashion"

    def _merge(self, u):
        domain = self._domain(u)
        new = _clean_slots(u, domain)
        same = u.get("is_refinement") and self.slots.get("domain") == domain
        base = dict(self.slots) if same else {}
        if "max_price" in new and "min_price" not in new: base.pop("min_price", None)
        base.update(new)
        adj, prices = u.get("price_adjust"), [m["price"] for m in self.last_items]
        if adj and prices:
            med = statistics.median(prices)
            if adj == "cheaper": base["max_price"] = int(med * 0.8); base.pop("min_price", None)
            if adj == "premium": base["min_price"] = int(med * 1.2); base.pop("max_price", None)
        if domain == "laptop": base["product_type"] = ["Laptop"]
        self.slots = base
        return base

    def _recommend(self, message, u):
        slots = self._merge(u)
        q = u.get("followup_question")
        if q and not self.asked_followup and not slots.get("product_type"):
            self.asked_followup = True
            return {"text": q, "items": [], "table": None, "kind": "followup"}
        self.asked_followup = False
        query = u.get("search_query") or message
        self.last_query = query
        filters = dict(slots)
        if u.get("want_tiers"):
            t = tiered_search(query, filters, k=2)
            groups = [("Budget pick", t["budget"]), ("Best value", t["best_value"]), ("Premium pick", t["premium"])]
            notes = ("Show three budget tiers: budget / best value / premium. The premium tier is intentionally "
                     "above the user's budget; present it as an upgrade option, not a problem.")
        else:
            r = search(query, filters, k=5)
            groups, notes = [("", r["items"])], ""
            if r["relaxed"]: notes = f"No exact match, so these filters were relaxed: {', '.join(r['relaxed'])}."
        if not any(items for _, items in groups):
            return {"text": "I couldn't find anything matching that in our catalog. Could you loosen the budget or a requirement a little?",
                    "items": [], "table": None, "kind": "empty"}
        return self._compose(message, groups, notes, "Recommend these products for the user's need.", "recommend")

    def _bundle(self, message, u):
        slots = self._merge(u)
        total = slots.get("max_price")
        parts, missing = [], []
        for p in (u.get("bundle_parts") or []):
            dom = p.get("domain") if p.get("domain") in DOMAINS else "fashion"
            pt = ["Laptop"] if dom == "laptop" else [x for x in (_pick(y, FASHION_TYPES) for y in _as_list(p.get("product_type"))) if x]
            (parts if pt else missing).append({**p, "domain": dom, "product_type": pt})
        if not parts:
            return self._scope(message, {"out_of_scope_reason": "None of the requested items are in the catalog."})
        wsum = sum(TYPE_MEDIAN.get(p["product_type"][0], 500) for p in parts)
        specs = []
        for p in parts:
            if p["domain"] == "laptop":
                f = {**_clean_slots(p, "laptop"), "product_type": ["Laptop"]}
                f.pop("max_price", None); f.pop("min_price", None)
            else:
                f = {"domain": "fashion", "product_type": p["product_type"]}
                occ = _pick(p.get("occasion"), FASHION_OCC)
                if occ: f["occasion"] = occ
                if slots.get("season") and slots.get("domain") == "fashion": f["season"] = slots["season"]
            if total: f["max_price"] = int(total * TYPE_MEDIAN.get(p["product_type"][0], 500) / wsum * 1.3)
            specs.append({"label": p.get("label") or p["product_type"][0], "query": p.get("query") or message, "filters": f})
        res = bundle_search(specs, k=2)
        groups = [(label, items) for label, items in res.items() if items]
        est = sum(items[0]["price"] for _, items in groups)
        notes = f"Estimated total if the user picks the first item of each group: ₹{est}" + (f" (budget ₹{total})." if total else ".")
        if missing: notes += f" Not in our catalog (say so honestly): {', '.join(m.get('label', '?') for m in missing)}."
        self.asked_followup = False
        self.last_query = message
        return self._compose(message, groups, notes, "Present this as a complete bundle/outfit, one short line per group.", "bundle")

    def _compare(self, message, u):
        dom = u.get("domain") if u.get("domain") in DOMAINS else self.slots.get("domain")
        items = []
        for t in (u.get("compare_targets") or []):
            if isinstance(t, int) or (isinstance(t, str) and t.strip().isdigit()):
                i = int(t) - 1
                if 0 <= i < len(self.last_items): items.append(self.last_items[i])
            elif isinstance(t, str) and t.strip():
                r = search(t, {"domain": dom} if dom else {}, k=1)["items"]
                if r: items.append(r[0])
        uniq, seen = [], set()
        for m in items:
            if m["pid"] not in seen: seen.add(m["pid"]); uniq.append(m)
        if len(uniq) < 2:
            return {"text": "Which two products should I compare? You can say 'compare 1 and 3' from the list above, or describe them.",
                    "items": [], "table": None, "kind": "followup"}
        uniq = uniq[:3]
        out = self._compose(message, [("", uniq)], "",
                            "Compare these products: key differences, pros/cons of each, then a verdict on who each suits. A spec table is shown separately, so don't repeat every spec.", "compare")
        rate = lambda m: m["rating"] if m["rating_known"] else "n/a"
        if all(m.get("domain") == "laptop" for m in uniq):
            out["table"] = [{"#": i + 1, "Product": m["title"][:60], "Price": f"₹{m['price']}", "Rating": rate(m),
                             "CPU": m["cpu"], "RAM": f"{m['ram_gb']}GB", "Storage": f"{m['storage_gb']}GB",
                             "GPU": m["gpu"] or "Integrated", "Screen": f"{m['screen_inch']:g} in" if m["screen_inch"] else "-",
                             "Weight": f"{m['weight_kg']:g} kg" if m["weight_kg"] else "-",
                             "Battery": f"{m['battery_hrs']:g} h" if m["battery_hrs"] else "-"} for i, m in enumerate(uniq)]
        else:
            out["table"] = [{"#": i + 1, "Product": m["title"], "Brand": m["brand"], "Price": f"₹{m['price']}",
                             "Discount": f"{m['discount_pct']}%", "Rating": rate(m), "Fabric": m["fabric_group"],
                             "Fit": m["fit"] or "-", "Sleeve": m["sleeve"] or "-", "Season": m["season"]}
                            for i, m in enumerate(uniq)]
        return out

    def _advice(self, message):
        text = chat_text(ADVICE_SYSTEM, self.history[-4:] + [{"role": "user", "content": message}])
        return {"text": text, "items": [], "table": None, "kind": "advice"}

    def _scope(self, message, u):
        payload = {"message": message, "reason": u.get("out_of_scope_reason")}
        text = chat_text(SCOPE_SYSTEM, self.history[-4:] + [{"role": "user", "content": json.dumps(payload)}], model=LLM_FAST)
        return {"text": text, "items": [], "table": None, "kind": "scope"}

    def _compose(self, message, groups, notes, task, kind):
        flat, products = [], []
        for label, items in groups:
            for m in items:
                m = dict(m, group=label); flat.append(m)
                p = {"n": len(flat), "group": label, "title": m["title"], "brand": m["brand"], "price": m["price"],
                     "rating": m["rating"] if m["rating_known"] else "not rated"}
                if m.get("domain") == "laptop":
                    p.update({"cpu": m["cpu"], "ram_gb": m["ram_gb"], "storage_gb": m["storage_gb"], "gpu": m["gpu"] or "integrated",
                              "gpu_type": m["gpu_type"], "screen_inch": m["screen_inch"] or "unknown",
                              "weight_kg": m["weight_kg"] or "unknown", "battery_hours": m["battery_hrs"] or "unknown",
                              "laptop_type": m["laptop_type"], "best_for": m["occasions"], "os": m["os"]})
                else:
                    p.update({"mrp": m["mrp"], "discount_pct": m["discount_pct"], "color": m["color"],
                              "fabric": m["fabric_group"], "fit": m["fit"], "sleeve": m["sleeve"],
                              "occasions": m["occasions"], "season": m["season"]})
                products.append(p)
        self.last_items = flat
        payload = {"user_message": message, "USER_FILTERS": self.slots, "TASK": task, "NOTES": notes, "PRODUCTS": products}
        text = chat_text(GEN_SYSTEM, self.history[-4:] + [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
        return {"text": text, "items": flat, "table": None, "kind": kind}