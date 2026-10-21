# Style Assistant: An Intelligent Product Recommendation Chatbot

A conversational **virtual sales associate** for e-commerce. Instead of keyword search and filters, customers describe what they need in plain language ("I'm going to Goa next month, suggest some black shirts", "I need a laptop for Computer Engineering") and the assistant understands the intent, asks a follow-up when something essential is missing, recommends products with reasons, compares them side by side, and offers alternatives at different budgets.

Built for the PetPooja assessment (Problem Statement 1). Everything runs on free tooling: **Groq** (LLM), **ChromaDB** (local vector database), **sentence-transformers** (local embeddings) and **Streamlit** (UI).

---

## Features

| Capability from the problem statement | How it is handled |
|---|---|
| Understand natural-language questions and intent | An LLM turns each message into structured intent and filters (JSON) |
| Ask follow-up questions when information is missing | Asks one short clarifying question only when the product category cannot be inferred |
| Recommend suitable products and explain why | Every product comes with a reason grounded in its real attributes (fabric, fit, season, specs, rating) |
| Compare products side by side | "compare 1 and 3" gives a spec table plus pros, cons and a verdict |
| Suggest alternatives within different budgets | Budget, best-value and premium tiers around the user's budget |
| Complete lifestyle bundles | "I'm joining a gym" returns shoes, T-shirt, shorts, track pants and socks with an estimated total |
| Need-based recommendation | "Laptop for Computer Engineering" becomes RAM, CPU and storage requirements |
| Refinement across turns | "cheaper ones", "lighter ones" modify the previous request instead of starting over |
| General buying advice | "Is 8GB RAM enough for programming?" gets a short, practical answer |
| Honesty about the catalog | Says plainly when something is not stocked (phones, women's wear, water bottles) or when a filter had to be relaxed |

## How it works

```
User message
   |
   v
[1] Understanding (Groq, JSON output)
    intent: recommend | compare | bundle | advice | chitchat | out_of_scope
    slots:  domain, product type, colour, occasion, season, budget, brand,
            laptop specs (RAM, storage, CPU tier, GPU type, weight)
   |
   +-- essential info missing?  -> ask ONE follow-up question
   |
   v
[2] Retrieval (ChromaDB + sentence-transformers)
    - hard constraints (budget, type, colour, RAM ...) -> metadata filters
    - semantic search on the expanded query
    - re-rank: similarity + rating + discount + occasion/season boost
    - de-duplicate and cap results per brand
    - if too few results, relax soft filters and report which ones
   |
   v
[3] Response generation (Groq)
    grounded ONLY in the retrieved products: reasons, price, honest mismatches,
    next-step suggestion
   |
   v
[4] Streamlit UI: chat, product cards, comparison table, session memory
```

**Design principle:** the LLM understands and explains, but code enforces the facts. The LLM never sees products that were not retrieved, so it cannot invent items or prices, and hard constraints such as budget and RAM are applied in the vector-store query, not left to the model's judgement.

## Tech stack

| Layer | Choice |
|---|---|
| LLM | Groq API: `openai/gpt-oss-120b` (answers) and `openai/gpt-oss-20b` (intent extraction) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, run locally on CPU |
| Vector database | ChromaDB in embedded mode (`PersistentClient`): no server, no account |
| Data processing | pandas, pyarrow |
| UI | Streamlit |

Model names are set in `src/config.py`. Groq retires models from time to time; if a call returns `model_not_found`, list the available models with `Groq(api_key=...).models.list()` and update the two names.

## Data

No dataset was supplied with the assessment, so public Kaggle datasets were used. Both are Flipkart listings.

| Domain | Source | Raw | After cleaning |
|---|---|---|---|
| Men's fashion | Flipkart Fashion Products dataset (30k-row sample) | 30,000 | about 13,300 unique, in-stock products |
| Laptops | Flipkart "Laptop data 2022" | 984 | 949 unique laptops |

Main cleaning steps:

- **Fashion:** repaired truncated brand names from product URLs, extracted colour, fabric, fit and sleeve from titles and `product_details`, derived a clean `product_type` and gender, rule-based occasion and season tags, removed out-of-stock items and about 14,000 duplicate listings, and built an embedding text per product.
- **Laptops:** fixed the mangled rupee symbol in prices, converted RAM, storage, screen size, weight and battery to numbers, derived CPU tier (Entry / Mid / High / Ultra) and GPU type (Dedicated / Integrated), tagged use cases (gaming, programming/engineering, student, creative, business, portable) and built an embedding text per laptop.

The cleaned catalogs are committed in `data/` (`catalog.parquet`, `catalog_laptops.parquet`), so the raw datasets are only needed if you want to re-run the cleaning.

## Project structure

```
smart-sales-assistant/
├── app.py                      # Streamlit UI
├── requirements.txt
├── .env.example                # copy to .env and add your Groq key
├── data/
│   ├── catalog.parquet         # cleaned fashion catalog
│   ├── catalog_laptops.parquet # cleaned laptop catalog
│   └── chroma_db/              # generated by build_index.py (not committed)
├── src/
│   ├── config.py               # paths, model names, API key
│   ├── clean_laptops.py        # laptop dataset cleaning
│   ├── build_index.py          # embeddings + ChromaDB index
│   ├── retrieval.py            # filters, re-ranking, budget tiers, bundles
│   ├── llm.py                  # Groq client with retry and JSON handling
│   ├── assistant.py            # intent handling, conversation state, prompts
│   ├── test_search.py          # raw vector search sanity check
│   ├── test_retrieval.py       # retrieval engine checks (no LLM)
│   └── test_assistant.py       # end-to-end conversation checks
└── notebooks/                  # (optional) fashion data cleaning notebook
```

## Setup and run

**Prerequisites:** Python 3.10+ and a free Groq API key from [console.groq.com](https://console.groq.com).

```bash
# 1. Clone and enter the project
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Create and activate a virtual environment
python -m venv myenv
myenv\Scripts\activate            # Windows
# source myenv/bin/activate       # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
copy .env.example .env            # Windows   (macOS / Linux: cp .env.example .env)
# then open .env and set GROQ_API_KEY=your_key_here

# 5. Build the vector index (one time, a few minutes on CPU)
python src/build_index.py

# 6. Launch the app
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). The first page load takes a little longer while the embedding model loads.

**Optional checks**

```bash
python src/test_retrieval.py    # retrieval engine, no LLM calls
python src/test_assistant.py    # full conversations (uses the Groq API)
```

**Rebuilding the laptop catalog from raw data** (optional): place `complete laptop data0.csv` in `data/`, run `python src/clean_laptops.py`, then re-run `python src/build_index.py`.

## Try these

- "I'm going to Goa next month. Suggest some black shirts." then "show me cheaper ones" then "compare 1 and 2"
- "Show me jeans in different budgets, around 1500"
- "I'm joining a gym next week. What should I buy? Budget around 3000"
- "I need a laptop for Computer Engineering"
- "Best gaming laptop under 90000"
- "Is 8GB RAM enough for programming?"
- "I need something to wear" (triggers a follow-up question)
- "Suggest the best camera phone under 40000" (honest out-of-scope answer)

## Key design decisions

- **Hybrid retrieval.** Metadata filters guarantee hard requirements (budget, RAM, colour), while vector search handles vague intent ("light, breathable, beach-friendly").
- **Grounded generation.** The model only talks about retrieved products, and is told to flag any item that does not match the request.
- **Honest relaxation.** If too few products match, soft filters (rating, season, brand, colour, weight, storage) are dropped in a fixed order and the user is told which ones.
- **Ranking beyond similarity.** Score combines semantic similarity (50%), rating (25%, capped because the data has no review counts), discount (5%) and occasion/season boosts, with a penalty for poorly rated items and a cap of two results per brand for variety.
- **No heavy frameworks.** Plain Python instead of LangChain keeps the flow easy to read, debug and explain.
- **Domain-extensible.** Products share one schema with a `domain` field and domain-specific attributes stored as flat metadata, so a new catalog needs only a cleaning script and a small config change.

## Limitations

- The catalog covers **men's fashion and laptops only**. There are no phones, women's or kids' wear, or accessories such as water bottles or bags. The assistant says so instead of inventing products.
- Fashion data is a 2021 crawl and laptop prices are from 2022, so prices and availability are not current. The assistant mentions this when recommending laptops.
- The datasets have no review counts, so ratings cannot be confidence-weighted (a 5.0 may be a single review).
- Product images come from the 2021 crawl and some may no longer load; laptops have no images and show a placeholder.
- Occasion, season and use-case tags are rule-based approximations derived from titles and specs.
- Conversation memory lasts for one browser session, and the free Groq tier is rate limited, so heavy concurrent use will slow responses.

## Possible next steps

- Add more categories (phones, women's fashion) with the same cleaning-script-plus-config pattern.
- Use a cross-encoder or LLM re-ranker for the final ordering.
- Add review-count-aware ratings, live prices and stock from an API, and user accounts with saved preferences.
- Add an automated evaluation set of queries with expected product types and constraints.
- Deploy on Hugging Face Spaces or similar.
