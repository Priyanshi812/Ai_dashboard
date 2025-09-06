# factory_super_app_ai.py
import os, re, math, sqlite3, urllib.parse, io
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF

APP_TITLE = "Ceramic Factory – AI Sales, Quotes & CRM"
DB_PATH   = Path("factory.db")         # optional (uses your existing schema)
CSV_PATH  = Path("skus.csv")           # fallback if DB not present
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# -------------------------------------------------------------------
# Sample data fallback
# -------------------------------------------------------------------
SAMPLE_ROWS = [
    {"sku_id": 1, "name": "Tile A - Classic White", "size": "12x12", "finish": "Glossy",  "unit": "pcs", "price": 30.0, "in_stock": 180, "image_url": ""},
    {"sku_id": 2, "name": "Tile B - Urban Grey",    "size": "24x24", "finish": "Matte",   "unit": "pcs", "price": 50.0, "in_stock": 85,  "image_url": ""},
    {"sku_id": 3, "name": "Tile C - Sand Beige",    "size": "24x24", "finish": "Glossy",  "unit": "pcs", "price": 55.0, "in_stock": 120, "image_url": ""},
    {"sku_id": 4, "name": "Tile D - Rustic Stone",  "size": "12x12", "finish": "Rustic",  "unit": "pcs", "price": 42.0, "in_stock": 40,  "image_url": ""},
    {"sku_id": 5, "name": "Tile E - Polished Marble","size": "24x24","finish": "Polished","unit": "pcs", "price": 95.0, "in_stock": 30,  "image_url": ""},
]

def ensure_csv():
    if not CSV_PATH.exists():
        pd.DataFrame(SAMPLE_ROWS).to_csv(CSV_PATH, index=False)

# -------------------------------------------------------------------
# Save helpers (Admin tab)
# -------------------------------------------------------------------
def normalize_sku_df(df: pd.DataFrame) -> pd.DataFrame:
    want_cols = ["sku_id","name","size","finish","unit","price","in_stock","image_url","description"]
    for c in want_cols:
        if c not in df.columns:
            df[c] = np.nan
    df["sku_id"]   = pd.to_numeric(df["sku_id"], errors="coerce").astype("Int64")
    df["name"]     = df["name"].fillna("").astype(str)
    df["size"]     = df["size"].fillna("").astype(str)
    df["finish"]   = df["finish"].fillna("").astype(str)
    df["unit"]     = df["unit"].fillna("pcs").astype(str)
    df["price"]    = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["in_stock"] = pd.to_numeric(df["in_stock"], errors="coerce").fillna(0).astype(int)
    df["image_url"]= df["image_url"].fillna("").astype(str)
    df["description"]= df["description"].fillna("").astype(str)

    if df["sku_id"].isna().any():
        max_id = int(df["sku_id"].dropna().max() or 0)
        n_missing = df["sku_id"].isna().sum()
        df.loc[df["sku_id"].isna(), "sku_id"] = list(range(max_id+1, max_id+1+n_missing))
        df["sku_id"] = df["sku_id"].astype(int)

    if df["sku_id"].duplicated().any():
        raise ValueError("Duplicate sku_id values. Please make each sku_id unique.")
    return df[want_cols]

def save_skus_to_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)

def save_skus_to_db(df: pd.DataFrame, db_path: Path):
    conn = sqlite3.connect(str(db_path))
    df.to_sql("sku", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

# -------------------------------------------------------------------
# Data loaders
# -------------------------------------------------------------------
def has_db() -> bool:
    return DB_PATH.exists()

def load_from_db() -> Tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(str(DB_PATH))
    skus = pd.read_sql_query("SELECT * FROM sku", conn)
    inv  = pd.read_sql_query("SELECT * FROM inventory_txn", conn)
    conn.close()
    return skus, inv

def compute_stock_from_txn(inv_txn: pd.DataFrame) -> pd.Series:
    if inv_txn.empty: return pd.Series(dtype=float)
    sku_txn = inv_txn[inv_txn["item_type"]=="sku"].copy()
    stock = sku_txn.groupby("item_id")["qty"].sum()
    stock.index.name = "sku_id"
    stock.name = "in_stock"
    return stock

def load_from_csv(uploaded: Optional[pd.DataFrame]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_csv()
    df = uploaded if uploaded is not None else pd.read_csv(CSV_PATH)
    for c in ["name","size","finish","unit","image_url","description"]:
        if c in df.columns: df[c] = df[c].fillna("").astype(str)
    for c in ["price","in_stock"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    inv_dummy = pd.DataFrame(columns=["item_type","item_id","qty","reason","ts"])
    return df, inv_dummy

# -------------------------------------------------------------------
# Intent parsing and recommenders
# -------------------------------------------------------------------
SIZE_PAT = re.compile(r"(\d{2}x\d{2}|\d{1,2}\s?x\s?\d{1,2})", re.I)
QTY_PAT  = re.compile(r"\b(\d{1,5})\s?(pcs|pieces|tiles|boxes|box)?\b", re.I)
FINISH_WORDS = ["matte","glossy","satin","rustic","polished"]
ROOM_WORDS   = ["kitchen","bathroom","living","outdoor","balcony","commercial","hall"]

def parse_query(q: str) -> Dict[str, Any]:
    ql = q.lower()
    size = (m.group(1).replace(" ", "") if (m:=SIZE_PAT.search(ql)) else None)
    finish = next((f for f in FINISH_WORDS if f in ql), None)
    qty = None
    if (m:=QTY_PAT.findall(ql)):
        nums = [int(x[0]) for x in m if x[0].isdigit()]
        if nums: qty = max(nums)
    budget = None
    if (b:=re.search(r"(?:under|below|upto|up to|budget|<=|less than)\s*₹?\s?(\d{2,6})", ql)):
        budget = float(b.group(1))
    room = next((r for r in ROOM_WORDS if r in ql), None)
    toks = re.findall(r"[a-zA-Z]{3,}", ql)
    stop = set(FINISH_WORDS + ROOM_WORDS + ["need","want","for","tile","tiles","piece","pieces","box","boxes","pcs","size","looking","recommend","recommendation","order","buy","get","quote","under","below","upto","budget"])
    keywords = [t for t in toks if t not in stop]
    return {"size": size, "finish": finish, "qty": qty, "budget": budget, "room": room, "keywords": keywords}

def filter_candidates(skus: pd.DataFrame, intent: Dict[str,Any]) -> pd.DataFrame:
    df = skus.copy()
    if "size" in df.columns and intent["size"]:
        df = df[df["size"].str.lower().str.replace(" ","") == intent["size"].lower()]
    if "finish" in df.columns and intent["finish"]:
        df = df[df["finish"].str.lower().str.contains(intent["finish"].lower())]
    if intent["budget"] is not None and "price" in df.columns:
        df = df[df["price"].fillna(0) <= 1.2*intent["budget"]]
    return df

def fuzzy_score(text: str, query: str) -> float:
    if not query.strip(): return 0.0
    return SequenceMatcher(None, query, text).ratio()

def score_by_keywords(cands: pd.DataFrame, intent: Dict[str,Any]) -> np.ndarray:
    if cands.empty: return np.array([])
    q = " ".join(intent["keywords"]) if intent["keywords"] else ""
    docs = (cands["name"].fillna("") + " " + cands.get("finish","").fillna("") + " " + cands.get("size","").fillna("")).tolist()
    return np.array([fuzzy_score(d,q) for d in docs], dtype=float)

def rank_candidates(cands: pd.DataFrame, stock_series: Optional[pd.Series], intent: Dict[str,Any]) -> pd.DataFrame:
    df = cands.copy()
    if "in_stock" not in df.columns or df["in_stock"].isna().all():
        if stock_series is not None and not stock_series.empty:
            df = df.merge(stock_series, left_on="sku_id", right_index=True, how="left")
    df["in_stock"] = df["in_stock"].fillna(0)

    kw = score_by_keywords(df, intent)
    if kw.size == 0: kw = np.zeros(len(df))
    df["kw_score"] = kw

    if df["in_stock"].max() > 0:
        df["avail_score"] = df["in_stock"] / df["in_stock"].max()
    else:
        df["avail_score"] = 0.0

    if intent["budget"] is not None and "price" in df.columns and df["price"].notna().any():
        price = df["price"].fillna(df["price"].median() if df["price"].notna().any() else 0)
        diff  = np.abs(price - intent["budget"])
        norm  = (diff.max() - diff) / (diff.max() if diff.max()>0 else 1.0)
        df["budget_score"] = norm
    else:
        df["budget_score"] = 0.0
    df["score"] = 0.55*df["kw_score"] + 0.30*df["avail_score"] + 0.15*df["budget_score"]
    return df.sort_values("score", ascending=False)

def similar_items(all_skus: pd.DataFrame, row: pd.Series, k:int=2) -> pd.DataFrame:
    base = (row.get("name","") + " " + row.get("finish","") + " " + row.get("size",""))
    sims = []
    for _, r in all_skus.iterrows():
        if r["sku_id"] == row["sku_id"]: continue
        text = (r.get("name","") + " " + r.get("finish","") + " " + r.get("size",""))
        sims.append((SequenceMatcher(None, base, text).ratio(), r))
    sims.sort(reverse=True, key=lambda x: x[0])
    return pd.DataFrame([s[1] for s in sims[:k]])

# -------------------------------------------------------------------
# Operations helpers
# -------------------------------------------------------------------
def estimate_eta(in_stock: float, needed: Optional[int]) -> str:
    if needed is None:
        return "Dispatch in 2–3 days" if in_stock > 0 else "Made-to-order: 7–10 days"
    if in_stock >= needed:
        return "Dispatch in 2–3 days"
    shortage = max(0, needed - in_stock)
    extra_days = 5 + int(math.ceil(shortage/200))
    return f"Partial dispatch now; balance in ~{extra_days} days"

ZONE_SHIP = {"Local/City":2, "In-State":3, "Inter-State":5, "Remote":7}

def quote_totals(cart: List[dict], discount_pct: float, gst_pct: float, shipping_days:int) -> Tuple[float,float,float,float]:
    sub = sum(float(it.get("price") or 0)*int(it["qty"]) for it in cart)
    disc = sub * (discount_pct/100.0)
    tax  = (sub - disc) * (gst_pct/100.0)
    grand = sub - disc + tax
    return sub, disc, tax, grand

def quote_to_pdf(cart: list, buyer: str, discount_pct: float, gst_pct: float, zone:str, filename="quote.pdf") -> Path:
    sub, disc, tax, grand = quote_totals(cart, discount_pct, gst_pct, ZONE_SHIP.get(zone,3))
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16); pdf.cell(0, 10, "Quotation", ln=True)
    pdf.set_font("Arial", size=11); pdf.cell(0, 8, f"Customer: {buyer}", ln=True); pdf.ln(2)
    pdf.set_font("Arial","B",11)
    pdf.cell(70,8,"Item",1); pdf.cell(20,8,"Qty",1); pdf.cell(20,8,"Unit",1)
    pdf.cell(30,8,"Price",1); pdf.cell(30,8,"Subtotal",1,ln=True)
    pdf.set_font("Arial", size=11)
    for it in cart:
        price = float(it.get("price") or 0.0); subl = price*int(it["qty"])
        pdf.cell(70,8,str(it["name"])[:34],1)
        pdf.cell(20,8,str(it["qty"]),1)
        pdf.cell(20,8,str(it.get("unit","pcs")),1)
        pdf.cell(30,8,f"₹{price:.2f}",1)
        pdf.cell(30,8,f"₹{subl:.2f}",1,ln=True)
    pdf.cell(140,8,"Subtotal",1); pdf.cell(30,8,f"₹{sub:.2f}",1,ln=True)
    pdf.cell(140,8,f"Discount ({discount_pct:.1f}%)",1); pdf.cell(30,8,f"-₹{disc:.2f}",1,ln=True)
    pdf.cell(140,8,f"GST ({gst_pct:.1f}%)",1); pdf.cell(30,8,f"₹{tax:.2f}",1,ln=True)
    pdf.cell(140,8,"Grand Total",1); pdf.cell(30,8,f"₹{grand:.2f}",1,ln=True)
    pdf.ln(4); pdf.cell(0,8,f"Estimated shipping: {ZONE_SHIP.get(zone,3)} days ({zone})", ln=True)
    out = Path(filename); pdf.output(str(out)); return out

def whatsapp_share_text(cart: list, buyer: str, discount_pct: float, gst_pct: float, zone:str) -> str:
    sub, disc, tax, grand = quote_totals(cart, discount_pct, gst_pct, ZONE_SHIP.get(zone,3))
    lines = [f"Quotation for {buyer}"]
    for it in cart:
        price = float(it.get("price") or 0.0)
        lines.append(f"- {it['name']} x{it['qty']} @ ₹{price:.2f}")
    lines += [f"Subtotal: ₹{sub:.2f}", f"Discount ({discount_pct}%): -₹{disc:.2f}", f"GST ({gst_pct}%): ₹{tax:.2f}",
              f"Grand Total: ₹{grand:.2f}", f"Shipping ETA: {ZONE_SHIP.get(zone,3)} days ({zone})"]
    return "\n".join(lines)

def save_order(cart: list, buyer:str, discount_pct: float, gst_pct: float, zone:str) -> str:
    orders = Path("orders.csv")
    sub, disc, tax, grand = quote_totals(cart, discount_pct, gst_pct, ZONE_SHIP.get(zone,3))
    order_id = f"ORD{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
    rows = []
    for it in cart:
        rows.append({
            "order_id": order_id, "buyer": buyer, "sku_id": it["sku_id"], "name": it["name"],
            "qty": it["qty"], "unit": it.get("unit","pcs"), "price": it.get("price",0.0),
            "discount_pct": discount_pct, "gst_pct": gst_pct, "zone": zone,
            "subtotal": sub, "discount": disc, "tax": tax, "grand_total": grand
        })
    df = pd.DataFrame(rows)
    if orders.exists():
        pd.concat([pd.read_csv(orders), df], ignore_index=True).to_csv(orders, index=False)
    else:
        df.to_csv(orders, index=False)
    return order_id

# -------------------------------------------------------------------
# FAQ (Gemini chat)
# -------------------------------------------------------------------
DEFAULT_FAQ = [
    ("How to clean matte tiles?", "Use pH-neutral cleaners; avoid acids/bleach. Mop with warm water weekly."),
    ("Are glossy tiles slippery?", "Glossy can be slippery when wet; consider matte for bathrooms and outdoors."),
    ("Best size for kitchen floors?", "24x24 or 12x12; choose matte/satin for grip and easier maintenance."),
    ("Delivery time?", "In-stock dispatch 2–3 days; made-to-order 7–10 days plus shipping."),
]

def gemini_answer(history: List[Dict[str,str]], kb_text: str) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        sys = ("You are a ceramic factory assistant. Use the FAQ when relevant. "
               "Be concise, polite, and helpful. If unsure, say what extra info you need.")
        messages = [{"role":"user","parts":[sys + "\n\nFAQ:\n" + kb_text]}]
        for m in history:
            messages.append({"role":"user","parts":[m["user"]]})
            if m.get("assistant"): messages.append({"role":"model","parts":[m["assistant"]]})
        resp = model.generate_content(messages)
        return resp.text
    except Exception as e:
        return f"(Gemini error: {e})"

# -------------------------------------------------------------------
# Semantic search (Gemini embeddings)
# -------------------------------------------------------------------
def gemini_embed(texts: List[str]) -> Optional[np.ndarray]:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        if isinstance(texts, str):
            texts = [texts]
        out = genai.embed_content(model="text-embedding-004", content=texts)
        if "embeddings" in out:
            vecs = [e["values"] for e in out["embeddings"]]
            return np.array(vecs, dtype="float32")
        if "embedding" in out:
            return np.array([out["embedding"]], dtype="float32")
    except Exception:
        return None
    return None

def cosine_sim_matrix(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = A / (np.linalg.norm(A, axis=1, keepdims=True)+1e-9)
    b = b / (np.linalg.norm(b, keepdims=True)+1e-9)
    return (A @ b.T).ravel()

# -------------------------------------------------------------------
# Visual search (DL / PyTorch) helpers
# -------------------------------------------------------------------
def _import_torch_stack():
    try:
        import torch
        import requests
        from PIL import Image
        import torchvision.transforms as T
        from torchvision.models import resnet18, ResNet18_Weights
        return {"ok": True, "torch": torch, "requests": requests, "Image": Image,
                "T": T, "resnet18": resnet18, "weights": ResNet18_Weights}
    except Exception as e:
        return {"ok": False, "err": str(e)}

def _load_image_any(path_or_url, Image, requests):
    try:
        if isinstance(path_or_url, str) and (path_or_url.startswith("http://") or path_or_url.startswith("https://")):
            r = requests.get(path_or_url, timeout=10)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        return Image.open(path_or_url).convert("RGB")
    except Exception:
        return None

def _build_resnet_feature_fn(torch, resnet18, weights, T):
    model = resnet18(weights=weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval()
    preprocess = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=weights.IMAGENET1K_V1.transforms().mean,
                    std=weights.IMAGENET1K_V1.transforms().std)
    ])
    @torch.no_grad()
    def featurize(img_pil):
        x = preprocess(img_pil).unsqueeze(0)
        feat = model(x).squeeze(0).numpy()
        n = (feat**2).sum() ** 0.5
        return feat / (n if n>0 else 1.0)
    return featurize

def _cosine(a, b):
    denom = (np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b) / denom) if denom>0 else 0.0

# -------------------------------------------------------------------
# Optional DL LSTM forecast helper
# -------------------------------------------------------------------
def lstm_forecast(values: List[float], horizon: int = 14):
    try:
        import tensorflow as tf
        x = np.array(values, dtype="float32")
        if len(x) < 20:
            return None
        win = 14
        X = []; y=[]
        for i in range(len(x)-win):
            X.append(x[i:i+win]); y.append(x[i+win])
        X = np.array(X)[...,None]; y = np.array(y)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(win,1)),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=20, verbose=0)
        hist = x[-win:].copy()
        preds = []
        for _ in range(horizon):
            p = float(model.predict(hist[None,...,None], verbose=0).ravel()[0])
            preds.append(max(0.0, p))
            hist = np.concatenate([hist[1:], [p]])
        return np.array(preds)
    except Exception:
        return None

# ========================== UI ==========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Data Source")
    mode = "SQLite (factory.db)" if has_db() else "CSV (skus.csv)"
    st.write(f"Detected: {mode}")
    uploaded = None
    if not has_db():
        uploaded = st.file_uploader("Upload your SKUs CSV", type=["csv"])
        if st.button("Use sample data"): ensure_csv()

    st.divider()
    st.header("Quote Cart")
    if "quote" not in st.session_state: st.session_state["quote"] = []
    if st.session_state["quote"]:
        for q in st.session_state["quote"]:
            st.write(f"- {q['name']} — {q['qty']} {q.get('unit','pcs')} @ ₹{q.get('price','-')}")
        if st.button("Clear Cart"): st.session_state["quote"].clear()
    else:
        st.caption("Add items from recommendations…")

tabs = st.tabs([
    "Customer Wizard", "Order Assistant", "Forecast", "Re-order",
    "FAQ (Gemini)", "Quote & Share", "Admin & Data", "Leads/CRM",
    "Visual Search (DL)", "Semantic Search (AI)"
])

# ---------- load data ----------
if has_db():
    skus, inv = load_from_db()
    stock_series = compute_stock_from_txn(inv)
else:
    skus, inv = load_from_csv(pd.read_csv(uploaded) if uploaded is not None else None)
    stock_series = None

# ---------------- TAB 1: Customer Wizard ----------------
with tabs[0]:
    st.subheader("Tell us your needs — we will pick ideal tiles")
    room  = st.selectbox("Room/Use", ["kitchen","bathroom","living","balcony","outdoor","commercial"])
    size  = st.selectbox("Size", sorted(skus["size"].dropna().unique().tolist()))
    finish= st.selectbox("Finish", sorted(skus["finish"].dropna().unique().tolist()))
    budget= st.slider("Budget per piece (₹)", 10, 200, 50)
    qty   = st.number_input("Quantity (pcs)", min_value=1, value=100, step=5)
    if st.button("Get recommendations"):
        query = f"{size} {finish} tiles for {room}, {qty} pcs under {budget}"
        st.info(f"Using query: {query}")
        intent = parse_query(query)
        cands = filter_candidates(skus, intent)
        if cands.empty:
            st.error("No matches. Try a different finish/size or higher budget.")
        else:
            ranked = rank_candidates(cands, stock_series, intent)
            for _, row in ranked.head(3).iterrows():
                with st.container(border=True):
                    L,R = st.columns([2,1], vertical_alignment="center")
                    with L:
                        st.markdown(f"### {row['name']}")
                        if str(row.get("image_url","")).startswith("http"):
                            st.image(row["image_url"], width=200)
                        st.caption(f"Size: {row.get('size','-')} | Finish: {row.get('finish','-')} | Unit: {row.get('unit','pcs')}")
                        price = float(row.get("price") or 0.0)
                        st.write(f"Price: ₹{price:.2f}")
                        in_stock = float(row.get("in_stock", 0) or 0)
                        st.write(f"In stock: {int(in_stock)}")
                        st.write(f"ETA: {estimate_eta(in_stock, qty)}")
                    with R:
                        pick = st.number_input(f"Qty for {row['name']}", min_value=1, value=int(qty), key=f"wiz_qty_{int(row['sku_id'])}")
                        if st.button("Add to Quote", key=f"wiz_add_{int(row['sku_id'])}"):
                            st.session_state["quote"].append({
                                "sku_id": int(row["sku_id"]), "name": row["name"], "qty": int(pick),
                                "unit": row.get("unit","pcs"), "price": float(row.get("price") or 0.0)
                            })
                            st.success(f"Added {row['name']} to quote.")
                sims = similar_items(skus, row, k=2)
                if not sims.empty:
                    st.caption("You may also like:")
                    st.write(", ".join(sims["name"].tolist()))

# ---------------- TAB 2: Order Assistant (free text) ----------------
with tabs[1]:
    st.subheader("Describe what you need (free text)")
    st.caption('Example: "24x24 glossy kitchen tiles, 120 pcs, under 50"')
    user_q = st.text_input("Your requirement")
    colA, colB = st.columns([1,1])
    go = colA.button("Recommend")
    if colB.button("Demo"): user_q, go = "need 24x24 glossy tiles for kitchen, 120 pcs under 50", True
    if go:
        if not user_q.strip(): st.warning("Please type a requirement.")
        else:
            intent = parse_query(user_q); st.write("Parsed Intent"); st.json(intent)
            cands = filter_candidates(skus, intent)
            if cands.empty: st.error("No matches. Try removing a constraint or widening budget.")
            else:
                ranked = rank_candidates(cands, stock_series, intent)
                for _, row in ranked.head(3).iterrows():
                    with st.container(border=True):
                        L,R = st.columns([2,1], vertical_alignment="center")
                        with L:
                            st.markdown(f"### {row['name']}")
                            if str(row.get("image_url","")).startswith("http"):
                                st.image(row["image_url"], width=200)
                            st.caption(f"Size: {row.get('size','-')} | Finish: {row.get('finish','-')} | Unit: {row.get('unit','pcs')}")
                            price = float(row.get("price") or 0.0); in_stock = float(row.get("in_stock",0) or 0)
                            st.write(f"Price: ₹{price:.2f}")
                            st.write(f"In stock: {int(in_stock)}")
                            st.write(f"ETA: {estimate_eta(in_stock, intent['qty'])}")
                        with R:
                            qp = st.number_input(f"Qty for {row['name']}", min_value=1, value=int(intent['qty'] or 1), key=f"free_qty_{int(row['sku_id'])}")
                            if st.button("Add to Quote", key=f"free_add_{int(row['sku_id'])}"):
                                st.session_state["quote"].append({
                                    "sku_id": int(row["sku_id"]), "name": row["name"],
                                    "qty": int(qp), "unit": row.get("unit","pcs"),
                                    "price": float(row.get("price") or 0.0)
                                })
                                st.success(f"Added {row['name']} to quote.")
                st.caption("Tip: include finish, size, room, quantity, and budget.")

# ---------------- TAB 3: Forecast (value graph + optional LSTM) ----------------
with tabs[2]:
    st.subheader("Usage trend (requires DB transactions)")
    if has_db() and not inv.empty:
        import altair as alt
        s = inv[inv["item_type"]=="sku"].copy()
        s["demand"] = (-s["qty"]).clip(lower=0)
        s["date"]   = pd.to_datetime(s["ts"]).dt.date
        ts = s.groupby("date")["demand"].sum().rename("demand")
        ts.index = pd.to_datetime(ts.index)
        df = pd.DataFrame(ts).reset_index().rename(columns={"index":"date"})
        df["ma7"] = df["demand"].rolling(7, min_periods=1).mean()
        alpha = st.slider("Smoothing alpha", 0.1, 0.9, 0.3, 0.1)
        sm = []; prev = df["demand"].iloc[0] if not df.empty else 0
        for x in df["demand"]:
            prev = alpha*x + (1-alpha)*prev; sm.append(prev)
        df["smooth"] = sm
        mu = df["demand"].mean() if not df.empty else 0
        sd = df["demand"].std(ddof=1) if len(df) > 1 else 0
        zthr = st.slider("Anomaly z-score >", 1.5, 4.0, 2.5, 0.1)
        df["anomaly"] = (np.abs(df["demand"] - mu)/(sd if sd>0 else 1)) > zthr
        horizon = st.slider("Forecast horizon (days)", 7, 30, 14, 1)
        mean7 = df["demand"].tail(7).mean() if len(df) >= 7 else df["demand"].mean()
        fut_idx = pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=horizon)
        df_fc = pd.DataFrame({"date": fut_idx, "forecast": [mean7]*horizon})
        st.toggle("Show value labels", key="show_labels", value=False)
        x_enc = alt.X("date:T", title="Date")
        tooltip = [
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("demand:Q", title="Demand", format=".0f"),
            alt.Tooltip("ma7:Q", title="7-day MA", format=".1f"),
            alt.Tooltip("smooth:Q", title="Smoothed", format=".1f"),
        ]
        bars = alt.Chart(df).mark_bar().encode(x=x_enc, y=alt.Y("demand:Q", title="Units"), tooltip=tooltip)
        ma_line = alt.Chart(df).mark_line(point=True).encode(x=x_enc, y="ma7:Q", tooltip=tooltip)
        smooth_line = alt.Chart(df).mark_line().encode(x=x_enc, y="smooth:Q", tooltip=tooltip)
        anomalies = alt.Chart(df[df["anomaly"]]).mark_point(size=80, color="red").encode(x=x_enc, y="demand:Q", tooltip=tooltip)
        fc_line = alt.Chart(df_fc).mark_line(strokeDash=[6,4]).encode(x="date:T", y="forecast:Q", tooltip=[alt.Tooltip("forecast:Q", format=".1f")])
        labels = alt.Chart(df).mark_text(dy=-6, fontSize=11).encode(x=x_enc, y="demand:Q", text=alt.Text("demand:Q", format=".0f")).transform_filter(alt.datum.demand > 0)
        chart = bars + ma_line + smooth_line + anomalies + fc_line
        if st.session_state.get("show_labels"):
            chart = chart + labels
        st.altair_chart(chart.properties(height=340, width="container"), use_container_width=True)

        if st.checkbox("Use DL LSTM forecast"):
            preds = lstm_forecast(df["demand"].tolist(), horizon)
            if preds is not None:
                df_lstm = pd.DataFrame({"date": df_fc["date"], "lstm": preds})
                lstm_line = alt.Chart(df_lstm).mark_line(strokeDash=[2,2]).encode(
                    x="date:T", y="lstm:Q", tooltip=[alt.Tooltip("lstm:Q", format=".1f")]
                )
                st.altair_chart((chart + lstm_line).properties(height=340), use_container_width=True)
            else:
                st.info("LSTM not available (needs tensorflow and enough history).")
    else:
        st.info("Connect SQLite with transactions (inventory_txn) to see the value graph.")

# ---------------- TAB 4: Re-order ----------------
with tabs[3]:
    st.subheader("Lead-time aware re-order hints")
    lead_time_days = st.slider("Assumed lead time (days)", 3, 21, 7)
    demand7 = {}
    if has_db() and not inv.empty:
        s = inv[inv["item_type"]=="sku"].copy()
        s["demand"] = (-s["qty"]).clip(lower=0)
        s["date"]   = pd.to_datetime(s["ts"]).dt.date
        recent = s[s["date"] >= (pd.Timestamp.today().date() - pd.Timedelta(days=7))]
        demand7 = recent.groupby("item_id")["demand"].sum().to_dict()
    view = skus.copy()
    if "in_stock" not in view.columns or view["in_stock"].isna().all():
        if stock_series is not None and not stock_series.empty:
            view = view.merge(stock_series, left_on="sku_id", right_index=True, how="left")
    view["in_stock"] = view["in_stock"].fillna(0)
    advice = []
    for _, r in view.iterrows():
        d7 = float(demand7.get(int(r["sku_id"]), 0.0))
        cover = (r["in_stock"]/(d7/7)) if d7>0 else float("inf")
        advice.append("Reorder soon" if cover < (lead_time_days+2) else "OK")
    view["advice"] = advice
    st.dataframe(view[["sku_id","name","size","finish","price","in_stock","advice"]], use_container_width=True)

# ---------------- TAB 5: FAQ (Gemini) ----------------
with tabs[4]:
    st.subheader("Customer FAQ — Gemini chat")
    if "faq" not in st.session_state: st.session_state["faq"] = DEFAULT_FAQ.copy()
    if "chat" not in st.session_state: st.session_state["chat"] = []
    with st.expander("Edit FAQ Knowledge Base"):
        df_faq = pd.DataFrame(st.session_state["faq"], columns=["question","answer"])
        st.dataframe(df_faq, use_container_width=True)
    prompt = st.text_input("Ask a question")
    if st.button("Send"):
        kb_text = "\n".join([f"Q: {q}\nA: {a}" for q,a in st.session_state["faq"]])
        st.session_state["chat"].append({"user": prompt})
        if GEMINI_KEY:
            reply = gemini_answer(st.session_state["chat"], kb_text)
        else:
            best = max(st.session_state["faq"], key=lambda qa: SequenceMatcher(None, prompt.lower(), qa[0].lower()).ratio())
            reply = best[1]
        st.session_state["chat"][-1]["assistant"] = reply
    for m in st.session_state["chat"]:
        st.markdown(f"You: {m['user']}")
        st.markdown(f"Assistant: {m.get('assistant','...')}")

# ---------------- TAB 6: Quote & Share ----------------
with tabs[5]:
    st.subheader("Build totals, export PDF, WhatsApp share, save order")
    buyer = st.text_input("Buyer / Company name", value="Valued Customer")
    discount_pct = st.slider("Discount %", 0.0, 25.0, 5.0, 0.5)
    gst_pct = st.slider("GST %", 0.0, 28.0, 18.0, 0.5)
    zone = st.selectbox("Shipping zone", list(ZONE_SHIP.keys()))
    if st.session_state["quote"]:
        sub, disc, tax, grand = quote_totals(st.session_state["quote"], discount_pct, gst_pct, ZONE_SHIP[zone])
        st.write(f"Subtotal: ₹{sub:.2f} | Discount: -₹{disc:.2f} | GST: ₹{tax:.2f} | Grand: ₹{grand:.2f}")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Generate PDF"):
                path = quote_to_pdf(st.session_state["quote"], buyer, discount_pct, gst_pct, zone, "quote.pdf")
                with open(path, "rb") as f:
                    st.download_button("Download Quote PDF", f, file_name="quote.pdf", mime="application/pdf")
        with c2:
            text = whatsapp_share_text(st.session_state["quote"], buyer, discount_pct, gst_pct, zone)
            link = "https://wa.me/?text=" + urllib.parse.quote(text)
            st.link_button("Share on WhatsApp", link)
        with c3:
            if st.button("Save Order"):
                oid = save_order(st.session_state["quote"], buyer, discount_pct, gst_pct, zone)
                st.success(f"Saved as {oid} to orders.csv")

        st.markdown("Explain this quote (AI)")
        if st.button("Explain & Upsell"):
            if not GEMINI_KEY:
                st.info("Set GEMINI_API_KEY to enable.")
            else:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_KEY)
                model = genai.GenerativeModel("gemini-1.5-flash")
                items_str = "\n".join([f"- {i['name']} x{i['qty']} at ₹{i.get('price',0)}" for i in st.session_state['quote']])
                prompt = (f"You are a sales assistant for a tile factory. The quote items are:\n{items_str}\n"
                          f"Discount {discount_pct}%, GST {gst_pct}%, zone {zone}.\n"
                          f"Write a short, friendly explanation for the customer with one or two upsell suggestions.")
                try:
                    msg = model.generate_content(prompt).text
                    st.write(msg)
                except Exception as e:
                    st.error(f"AI error: {e}")
    else:
        st.info("Cart is empty. Add items from the Order Assistant or Wizard.")

# ---------------- TAB 7: Admin & Data ----------------
with tabs[6]:
    st.subheader("Bulk edit SKUs (live edit and save)")
    if "skus_edit" not in st.session_state:
        st.session_state["skus_edit"] = skus.copy()
    st.caption("You can add or remove rows. Leave image_url blank if you don't have photos.")
    edited = st.data_editor(
        st.session_state["skus_edit"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "sku_id":   st.column_config.NumberColumn("sku_id", help="Unique product id", step=1),
            "name":     st.column_config.TextColumn("name"),
            "size":     st.column_config.TextColumn("size", help="e.g., 12x12, 24x24"),
            "finish":   st.column_config.TextColumn("finish", help="Matte/Glossy/Rustic/..."),
            "unit":     st.column_config.TextColumn("unit", help="pcs/box/etc"),
            "price":    st.column_config.NumberColumn("price", format="₹%.2f", step=1.0),
            "in_stock": st.column_config.NumberColumn("in_stock", step=1),
            "image_url":st.column_config.TextColumn("image_url", help="http(s) link to product image"),
            "description": st.column_config.TextColumn("description", help="Optional marketing copy"),
        },
        hide_index=True
    )
    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        if st.button("Reset (discard unsaved)"):
            st.session_state["skus_edit"] = skus.copy()
            st.rerun()
    with col2:
        if st.button("Save changes"):
            try:
                clean = normalize_sku_df(edited.copy())
                st.session_state["skus_edit"] = clean.copy()
                if has_db():
                    save_skus_to_db(clean, DB_PATH)
                    st.success(f"Saved to SQLite: {DB_PATH.name}")
                else:
                    save_skus_to_csv(clean, CSV_PATH)
                    st.success(f"Saved to CSV: {CSV_PATH.name}")
                skus = clean.copy()
            except Exception as e:
                st.error(f"Could not save: {e}")
    with col3:
        st.download_button(
            "Download current SKUs CSV",
            (edited if isinstance(edited, pd.DataFrame) else pd.DataFrame(edited)).to_csv(index=False),
            file_name="skus_export.csv"
        )
    st.caption("Changes are applied to the running app after you click Save changes.")
    st.divider()
    st.markdown("Generate descriptions with AI")
    gen_pick = st.multiselect("Pick SKUs", skus["name"].tolist())
    if st.button("Generate descriptions"):
        if not GEMINI_KEY:
            st.error("Set GEMINI_API_KEY to use AI generation.")
        else:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            new_rows = st.session_state["skus_edit"].copy()
            for name in gen_pick:
                r = new_rows[new_rows["name"]==name].iloc[0]
                prompt = (f"Write a concise, persuasive product description for a ceramic tile.\n"
                          f"Name: {r['name']}\nSize: {r.get('size','')}\nFinish: {r.get('finish','')}\n"
                          f"Use cases: kitchen, bathroom, living.\nLimit to 80 words.")
                try:
                    txt = model.generate_content(prompt).text.strip()
                    new_rows.loc[new_rows["name"]==name, "description"] = txt
                except Exception as e:
                    st.warning(f"Failed for {name}: {e}")
            st.session_state["skus_edit"] = new_rows
            st.success("Descriptions generated into the working table (remember to Save changes).")

# ---------------- TAB 8: Leads / CRM ----------------
with tabs[7]:
    st.subheader("Capture a lead")
    name = st.text_input("Customer name")
    phone = st.text_input("Phone / WhatsApp")
    city = st.text_input("City")
    note = st.text_area("Requirement notes")
    if st.button("Save Lead"):
        leads = Path("leads.csv")
        row = {"ts": pd.Timestamp.now(), "name": name, "phone": phone, "city": city, "note": note}
        if leads.exists():
            pd.concat([pd.read_csv(leads), pd.DataFrame([row])], ignore_index=True).to_csv(leads, index=False)
        else:
            pd.DataFrame([row]).to_csv(leads, index=False)
        st.success("Lead saved to leads.csv")

# ---------------- TAB 9: Visual Search (DL) ----------------
with tabs[8]:
    st.subheader("Find similar tiles from a photo (Deep Learning)")
    stack = _import_torch_stack()
    if not stack["ok"]:
        st.warning("PyTorch visual search not available. Install: pip install torch torchvision pillow requests")
    else:
        if "vs_featurize" not in st.session_state:
            st.session_state["vs_featurize"] = _build_resnet_feature_fn(
                stack["torch"], stack["resnet18"], stack["weights"], stack["T"]
            )
        if "vs_index" not in st.session_state:
            feats = []; rows  = []
            for _, r in skus.iterrows():
                img_url = str(r.get("image_url","")).strip()
                if not img_url: continue
                img = _load_image_any(img_url, stack["Image"], stack["requests"])
                if img is None:  continue
                f = st.session_state["vs_featurize"](img)
                feats.append(f); rows.append(r)
            st.session_state["vs_index"] = {"feats": np.array(feats) if feats else np.zeros((0,512)),
                                            "rows": rows}
        index = st.session_state["vs_index"]
        up = st.file_uploader("Upload a reference photo (jpg/png)", type=["jpg","jpeg","png"])
        topk = st.slider("Show top matches", 1, 10, 3)
        if up is not None:
            from PIL import Image
            img_q = Image.open(up).convert("RGB")
            st.image(img_q, caption="Query image", width=280)
            qf = st.session_state["vs_featurize"](img_q)
            if index["feats"].shape[0]==0:
                st.info("No SKU images indexed. Add image_url for your SKUs in Admin & Data.")
            else:
                sims = [(_cosine(qf, f), i) for i,f in enumerate(index["feats"])]
                sims.sort(reverse=True)
                st.write("Top matches:")
                for s, i in sims[:topk]:
                    row = index["rows"][i]
                    with st.container(border=True):
                        c1, c2 = st.columns([1,2])
                        with c1:
                            img = _load_image_any(str(row.get("image_url","")), stack["Image"], stack["requests"])
                            if img: st.image(img, width=160)
                        with c2:
                            st.markdown(f"**{row['name']}**")
                            st.caption(f"Size: {row.get('size','-')} | Finish: {row.get('finish','-')} | Score: {s:.3f}")
                            price = float(row.get("price") or 0.0)
                            st.write(f"Price: ₹{price:.2f}")
                            qty = st.number_input(f"Qty for {row['name']}", 1, 9999, 50, key=f"vs_qty_{int(row['sku_id'])}")
                            if st.button("Add to Quote", key=f"vs_add_{int(row['sku_id'])}"):
                                st.session_state["quote"].append({
                                    "sku_id": int(row["sku_id"]), "name": row["name"],
                                    "qty": int(qty), "unit": row.get("unit","pcs"),
                                    "price": price
                                })
                                st.success("Added to cart.")

# ---------------- TAB 10: Semantic Search (AI embeddings) ----------------
with tabs[9]:
    st.subheader("Semantic Search (Gemini embeddings)")
    if not GEMINI_KEY:
        st.info("Set GEMINI_API_KEY to enable AI embeddings.")
    else:
        if "emb_skus" not in st.session_state:
            texts = (skus["name"].fillna("") + " | " +
                     skus.get("finish","").fillna("") + " | " +
                     skus.get("size","").fillna("")).tolist()
            vecs = gemini_embed(texts)
            if vecs is None:
                st.error("Embedding failed. Check API key/quota.")
            else:
                st.session_state["emb_skus"] = vecs.astype("float32")
        if "emb_skus" in st.session_state:
            query = st.text_input("Describe your need (for example: premium glossy white, large size for living room)")
            if st.button("Search"):
                qv = gemini_embed([query])
                if qv is None:
                    st.error("Embedding failed for query.")
                else:
                    sims = cosine_sim_matrix(st.session_state["emb_skus"], qv)
                    top_idx = np.argsort(-sims)[:5]
                    for i in top_idx:
                        row = skus.iloc[i]
                        with st.container(border=True):
                            st.markdown(f"**{row['name']}**  — score {sims[i]:.3f}")
                            st.caption(f"Size: {row.get('size','-')} | Finish: {row.get('finish','-')}")
                            qty = st.number_input(f"Qty for {row['name']}", 1, 9999, 50, key=f"emb_qty_{int(row['sku_id'])}")
                            price = float(row.get("price") or 0.0)
                            if st.button("Add to Quote", key=f"emb_add_{int(row['sku_id'])}"):
                                st.session_state["quote"].append({
                                    "sku_id": int(row["sku_id"]), "name": row["name"],
                                    "qty": int(qty), "unit": row.get("unit","pcs"),
                                    "price": price
                                })
                                st.success("Added to cart.")
