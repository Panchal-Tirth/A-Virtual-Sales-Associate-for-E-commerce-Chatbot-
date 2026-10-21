import re
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "complete laptop data0.csv"
OUT_PATH = ROOT / "data" / "catalog_laptops.parquet"

df = pd.read_csv(CSV_PATH, encoding="cp1252")
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
df = df.rename(columns={"name": "title", "link": "url"})
df["selling_price"] = pd.to_numeric(df["Price"].astype(str).str.replace(r"[^\d]", "", regex=True), errors="coerce")
df = df.dropna(subset=["selling_price", "title"]).reset_index(drop=True)
df["selling_price"] = df["selling_price"].astype(int)
df["title"] = df["title"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

def C(name):
    return df[name] if name in df.columns else pd.Series([np.nan] * len(df), index=df.index)

def first_num(s, default=0.0):
    m = re.search(r"(\d+(?:\.\d*)?)", s) if isinstance(s, str) else None
    return float(m.group(1)) if m else default

def to_gb(s):
    if not isinstance(s, str): return 0
    m = re.search(r"(\d+(?:\.\d+)?)\s*(TB|GB)", s, re.I)
    if not m: return 0
    v = float(m.group(1))
    return int(v * 1024) if m.group(2).upper() == "TB" else int(v)

def join(*xs):
    return " ".join(x.strip() for x in xs if isinstance(x, str) and x.strip())

# ---------- brand ----------
BRAND = {"asus": "ASUS", "hp": "HP", "msi": "MSI", "dell": "Dell", "lenovo": "Lenovo", "acer": "Acer",
         "apple": "Apple", "realme": "realme", "infinix": "Infinix", "vaio": "VAIO", "avita": "Avita",
         "nokia": "Nokia", "samsung": "Samsung", "mi": "Mi", "lg": "LG", "gigabyte": "GIGABYTE",
         "razer": "Razer", "microsoft": "Microsoft", "honor": "Honor", "chuwi": "CHUWI"}
first = df.title.str.split().str[0]
df["brand"] = first.str.lower().map(BRAND).fillna(first.str.title())

# ---------- core specs ----------
df["ram_gb"] = C("RAM").map(to_gb)
ssd_gb = C("SSD Capacity").map(to_gb)
df["storage_gb"] = ssd_gb + C("HDD Capacity").map(to_gb) + C("EMMC Storage Capacity").map(to_gb)
has_ssd = (C("SSD").astype(str).str.lower() == "yes") | (ssd_gb > 0)

def storage_str(ssd, hdd, emmc):
    parts = []
    if ssd: parts.append(f"{ssd} SSD")
    if hdd: parts.append(f"{hdd} HDD")
    if emmc: parts.append(f"{emmc} eMMC")
    return " + ".join(parts) or "unknown"
df["storage_desc"] = [storage_str(*(x if isinstance(x, str) else "" for x in t))
                      for t in zip(C("SSD Capacity"), C("HDD Capacity"), C("EMMC Storage Capacity"))]

df["cpu"] = [join(b, n, g, v) for b, n, g, v in
             zip(C("Processor Brand"), C("Processor Name"), C("Processor Generation"), C("Processor Variant"))]
df["cpu_gen"] = C("Processor Generation").map(lambda s: int(first_num(s)))

def cpu_tier(name, brand):
    n = str(name).lower()
    if re.search(r"i9|ryzen 9|ultra 9", n): return "Ultra"
    if re.search(r"i7|ryzen 7|ultra 7", n): return "High"
    if re.search(r"i5|ryzen 5|ultra 5", n): return "Mid"
    if re.search(r"^m\d", n) or str(brand).lower() == "apple": return "High"
    return "Entry"
df["cpu_tier"] = [cpu_tier(n, b) for n, b in zip(C("Processor Name"), C("Processor Brand"))]

GAMING_GPU = r"GTX|RTX|Radeon RX|Arc A|Quadro"
df["gpu"] = C("Graphic Processor").map(lambda g: re.sub(r"\b(\w+) \1\b", r"\1", g.strip()) if isinstance(g, str) else "")
ded_mem = C("Dedicated Graphic Memory Capacity").notna()
df["gpu_type"] = np.where(ded_mem | df.gpu.str.contains(GAMING_GPU + r"|GeForce|MX\d", regex=True), "Dedicated", "Integrated")
gaming_gpu = df.gpu.str.contains(GAMING_GPU, regex=True)

def screen_in(s, title):
    m = re.search(r"\(([\d.]+)\s*inch\)", str(s), re.I) or re.search(r"([\d.]+)\s*inch", str(title), re.I)
    try: return float(m.group(1)) if m else 0.0
    except ValueError: return 0.0
df["screen_inch"] = [screen_in(s, t) for s, t in zip(C("Screen Size"), df.title)]

def res_of(s):
    m = re.search(r"(\d{3,4})\s*x\s*(\d{3,4})", str(s), re.I)
    return f"{m.group(1)}x{m.group(2)}" if m else ""
df["resolution"] = C("Screen Resolution").map(res_of)

def weight_of(w, title):
    v = first_num(w)
    if not v:
        m = re.search(r"(\d+(?:\.\d+)?)\s*kg", title, re.I)
        v = float(m.group(1)) if m else 0.0
    return v if 0.5 <= v <= 6 else 0.0
df["weight_kg"] = [weight_of(w, t) for w, t in zip(C("Weight"), df.title)]

def hours_of(s):
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(s))] if isinstance(s, str) else []
    v = max(nums) if nums else 0.0
    return v if 0 < v <= 30 else 0.0
df["battery_hrs"] = C("Battery Backup").map(hours_of)

df["refresh_hz"] = C("Refresh Rate").map(lambda s: int(first_num(s)))
df["os"] = C("Operating System").fillna("").astype(str)
df["touch"] = C("Touchscreen").astype(str).str.lower().eq("yes")
df["backlit"] = C("Backlit Keyboard").astype(str).str.lower().eq("yes")
ltype = C("Type").fillna("Laptop").astype(str).str.strip().replace({"Notebook": "Laptop"})
df["laptop_type"] = ltype
suitable = C("Suitable For").fillna("").astype(str)

# ---------- ratings ----------
df["rating"] = pd.to_numeric(C("user rating"), errors="coerce")
df["rating_known"] = df.rating.notna()
df["rating"] = df.rating.fillna(0.0)

# ---------- colour ----------
def base_color(s):
    s = str(s).lower()
    for c in ["black", "grey", "gray", "silver", "blue", "white", "red", "gold", "green", "pink", "purple", "brown", "orange"]:
        if c in s: return "Grey" if c == "gray" else c.title()
    return "Other"
df["color"] = C("Color").map(base_color)

# ---------- use-case tags ----------
def use_tags(lt, st, tier, ram, ssd, ggpu, w, price):
    lt, st, t = lt.lower(), st.lower(), []
    if "gaming" in lt or "gaming" in st or ggpu: t.append("gaming")
    if ram >= 8 and tier in ("Mid", "High", "Ultra") and ssd: t.append("programming/engineering")
    if ram >= 16 and (ggpu or tier in ("High", "Ultra")): t.append("creative/editing")
    if 0 < w <= 1.5: t.append("travel/portable")
    if "business" in lt or ("multitasking" in st and tier in ("Mid", "High") and "gaming" not in lt):
        t.append("business/office")
    if ("everyday" in st or (tier in ("Entry", "Mid") and price <= 60000)) and "gaming" not in lt:
        t.append("student/everyday")
    return ", ".join(t) or "student/everyday"
df["occasions"] = [use_tags(*a) for a in zip(df.laptop_type, suitable, df.cpu_tier, df.ram_gb, has_ssd,
                                              gaming_gpu, df.weight_kg, df.selling_price)]

# ---------- dedupe ----------
before = len(df)
df = df.drop_duplicates(["title", "selling_price"]).reset_index(drop=True)
print(f"rows: {before} -> {len(df)}")

# ---------- embedding text ----------
def search_text(r):
    parts = [r.title, f"Brand: {r.brand}", f"Type: {r.laptop_type}",
             f"Processor: {r.cpu} ({r.cpu_tier} tier)", f"{r.ram_gb}GB RAM", f"Storage: {r.storage_desc}",
             f"Graphics: {r.gpu or 'integrated'} ({r.gpu_type})"]
    if r.screen_inch: parts.append(f"Screen: {r.screen_inch:g} inch {r.resolution}" + (", touchscreen" if r.touch else ""))
    if r.weight_kg: parts.append(f"Weight: {r.weight_kg:g} kg")
    if r.battery_hrs: parts.append(f"Battery: up to {r.battery_hrs:g} hours")
    parts += [f"OS: {r.os}", f"Best for: {r.occasions}"]
    return ". ".join(p for p in parts if p)
df["search_text"] = df.apply(search_text, axis=1)

# ---------- shared-schema fields ----------
df["pid"] = ["lap_" + str(i) for i in range(len(df))]
df["domain"], df["product_type"], df["gender"], df["season"] = "laptop", "Laptop", "Unisex", "all-season"
df["actual_price"], df["discount_pct"], df["color_all"] = df.selling_price, 0, df.color
df["fabric_group"], df["fit"], df["sleeve"], df["pattern"], df["pack_of"] = "", "", "", "", 1
df["description"] = suitable
df["sub_category"], df["image_url"] = df.laptop_type, ""

keep = ["pid", "domain", "title", "brand", "product_type", "gender", "color", "color_all", "fabric_group", "fit",
        "sleeve", "pattern", "pack_of", "occasions", "season", "selling_price", "actual_price", "discount_pct",
        "rating", "rating_known", "description", "sub_category", "image_url", "url", "laptop_type", "cpu",
        "cpu_tier", "cpu_gen", "gpu", "gpu_type", "ram_gb", "storage_gb", "screen_inch", "resolution",
        "weight_kg", "battery_hrs", "refresh_hz", "os", "search_text"]
df[keep].to_parquet(OUT_PATH, index=False)

pd.set_option("display.width", 200, "display.max_colwidth", 120)
print("\nSHAPE:", df[keep].shape)
print("\ncpu_tier:\n", df.cpu_tier.value_counts())
print("\ngpu_type:\n", df.gpu_type.value_counts())
print("\nram_gb:\n", df.ram_gb.value_counts().head(6))
print("\noccasions:\n", df.occasions.value_counts().head(10))
print("\nbrands:\n", df.brand.value_counts().head(10))
print("\nunknown weight %:", round((df.weight_kg == 0).mean() * 100), "| unknown battery %:", round((df.battery_hrs == 0).mean() * 100))
print("\nprice by tier:\n", df.groupby("cpu_tier").selling_price.describe()[["count", "min", "50%", "max"]])
print("\nsearch_text sample:\n", df.search_text.iloc[0])