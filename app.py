import sys, html
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from assistant import Assistant
import retrieval

st.set_page_config(page_title="Style Assistant", page_icon="🛍️", layout="wide")


if "bot" not in st.session_state:
    st.session_state.bot = Assistant()
    st.session_state.msgs = []
    st.session_state.pending = None

EXAMPLES = [
    "I'm going to Goa next month. Suggest some black shirts.",
    "Show me jeans in different budgets, around 1500",
    "I'm joining a gym next week. What should I buy? Budget around 3000",
    "I need a laptop for Computer Engineering",
    "Best gaming laptop under ₹90,000",
    "Is 8GB RAM enough for programming?",
    "I need something to wear",
]
# ---------- sidebar ----------
with st.sidebar:
    st.title("🛍️ Style Assistant")
    st.caption("Your virtual sales associate for men's fashion.")
    st.markdown("**Try asking:**")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    st.divider()
    if st.button("🗑️ New conversation", use_container_width=True):
        st.session_state.bot = Assistant()
        st.session_state.msgs = []
        st.rerun()
    st.caption("Catalog: ~13k Flipkart men's fashion products + ~900 laptops (laptop prices are from 2022). Tip: say \"compare 1 and 3\" to compare items from the last list, or \"cheaper\" to refine.")

@st.cache_resource(show_spinner="Loading search engine (first run takes a moment)...")
def warm():
    retrieval._init()
    return True

st.header("What are you shopping for today?")
try:
    warm()
except Exception as e:
    st.error(f"Search engine failed to load: {type(e).__name__}: {e}")
    st.stop()

# ---------- rendering ----------
def card_html(m, n):
    img = html.escape(m.get("image_url") or "")
    if img:
        img_tag = (f'<img src="{img}" style="width:100%;height:100%;object-fit:contain" '
                   f'onerror="this.style.display=\'none\'">')
    else:
        img_tag = '<div style="display:flex;height:100%;align-items:center;justify-content:center;font-size:64px">💻</div>'
    rating = f"★ {m['rating']}" if m.get("rating_known") else "Not rated"
    off = f"<span style='color:#16a34a;font-weight:600'> {m['discount_pct']}% off</span>" if m["discount_pct"] > 0 else ""
    mrp = f"<s style='color:#888;font-size:0.85em'>₹{m['mrp']}</s> " if m["mrp"] > m["price"] else ""
    if m.get("domain") == "laptop":
        bits = [m["cpu"], f"{m['ram_gb']}GB RAM", f"{m['storage_gb']}GB", m["gpu_type"] + " GPU",
                f"{m['screen_inch']:g}\"" if m["screen_inch"] else "", f"{m['weight_kg']:g}kg" if m["weight_kg"] else ""]
        tags = " · ".join(x for x in bits if x)
    else:
        tags = " · ".join(x for x in [m["color"], m["fabric_group"], m["fit"], m["sleeve"], m["season"]]
                          if x and x not in ("Unknown",))
    return f"""
    <div style="border:1px solid rgba(128,128,128,0.3);border-radius:12px;padding:10px;margin-bottom:6px">
      <div style="height:200px;background:rgba(128,128,128,0.12);border-radius:8px;overflow:hidden;position:relative">
        <span style="position:absolute;top:6px;left:6px;background:#111;color:#fff;border-radius:6px;padding:1px 8px;font-size:0.8em;z-index:2">#{n}</span>
        {img_tag}
      </div>
      <div style="font-weight:600;margin-top:8px;line-height:1.25;min-height:2.5em">{html.escape(m['title'][:70])}</div>
      <div style="opacity:0.7;font-size:0.85em">{html.escape(m['brand'])} · {rating}</div>
      <div style="margin-top:4px"><b style="font-size:1.1em">₹{m['price']}</b> {mrp}{off}</div>
      <div style="opacity:0.65;font-size:0.8em;margin-top:4px">{html.escape(tags)}</div>
    </div>"""

def render(out):
    st.markdown(out["text"])
    if out.get("table"):
        st.dataframe(pd.DataFrame(out["table"]), hide_index=True)
    items = out.get("items") or []
    if not items:
        return
    current = None
    cols, ci = None, 0
    for n, m in enumerate(items, 1):
        label = m.get("group") or ""
        if label != current:
            current = label
            if label:
                st.markdown(f"**{label}**")
            cols, ci = st.columns(3), 0
        elif ci == 3:
            cols, ci = st.columns(3), 0
        with cols[ci]:
            st.markdown(card_html(m, n), unsafe_allow_html=True)
            if m.get("url"):
                st.link_button("View on Flipkart", m["url"], use_container_width=True)
        ci += 1

# ---------- main ----------

if not st.session_state.msgs:
    st.info("Ask in plain language: an occasion, a budget, a vibe. I can recommend, compare, build outfits and show budget alternatives.")

for role, content in st.session_state.msgs:
    with st.chat_message(role):
        if role == "user":
            st.markdown(content)
        else:
            render(content)

user_text = st.chat_input("e.g. Suggest black shirts for a Goa trip under ₹1000")
if st.session_state.pending:
    user_text, st.session_state.pending = st.session_state.pending, None

if user_text:
    st.session_state.msgs.append(("user", user_text))
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Finding the best picks..."):
                out = st.session_state.bot.reply(user_text)
        except Exception as e:
            out = {"text": f"Sorry, something went wrong on my side ({type(e).__name__}). Please try again in a moment.",
                   "items": [], "table": None, "kind": "error"}
        render(out)
    st.session_state.msgs.append(("assistant", out))