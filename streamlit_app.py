import streamlit as st
import pandas as pd
import re
from io import StringIO

# -------------------------------------------------
# Streamlit Dictionary Classification Bot (v1.1)
# -------------------------------------------------
# Robust to messy ground‑truth values such as "#REF!",
# non‑numeric labels ("yes"/"no"), missing data, etc.
# -------------------------------------------------

# ---------- Helper functions --------------------------------------------------

def safe_regex(keyword: str) -> re.Pattern:
    """Return a compiled word‑boundary regex for a keyword."""
    return re.compile(r"\b" + re.escape(keyword.lower()) + r"\b")


def compile_dictionary(keywords):
    """Pre‑compile regex patterns for faster matching."""
    return [safe_regex(kw) for kw in keywords]


def parse_gt(value):
    """Parse ground‑truth to 0/1 or return None when unparsable."""
    if pd.isna(value):
        return None

    # Handle common spreadsheet errors like '#REF!' gracefully
    val_str = str(value).strip().lower()

    # Map typical booleans / labels → ints
    truthy = {"1", "true", "yes", "y", "t"}
    falsy = {"0", "false", "no", "n", "f"}

    if val_str in truthy:
        return 1
    if val_str in falsy:
        return 0

    # Try numeric conversion last
    try:
        return int(float(val_str))
    except ValueError:
        return None  # Leave as missing if still invalid


def classify(df, text_col, gt_col, dictionary):
    """Run dictionary classification and compute basic metrics."""
    compiled_dict = compile_dictionary(dictionary)
    records = []

    for _, row in df.iterrows():
        text = str(row[text_col])
        pred = int(any(p.search(text.lower()) for p in compiled_dict))

        gt = parse_gt(row[gt_col]) if gt_col else None

        # Categorize for error analysis
        if gt is None:
            category = "Ungraded"
        elif pred == 1 and gt == 1:
            category = "TP"
        elif pred == 1 and gt == 0:
            category = "FP"
        elif pred == 0 and gt == 1:
            category = "FN"
        else:
            category = "TN"

        records.append({
            "text": row[text_col],
            "pred": pred,
            "gt": gt,
            "category": category,
        })

    res_df = pd.DataFrame(records)

    # --- Metrics (skip rows with missing GT) ------------------------
    if gt_col:
        eval_df = res_df.dropna(subset=["gt"])
        tp = ((eval_df.pred == 1) & (eval_df.gt == 1)).sum()
        fp = ((eval_df.pred == 1) & (eval_df.gt == 0)).sum()
        fn = ((eval_df.pred == 0) & (eval_df.gt == 1)).sum()
        tn = ((eval_df.pred == 0) & (eval_df.gt == 0)).sum()

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / len(eval_df) if len(eval_df) else 0.0

        metrics = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(len(eval_df)),
        }
    else:
        metrics = {}

    return res_df, metrics


# ---------- Streamlit UI ------------------------------------------------------

st.set_page_config(page_title="Dictionary Classifier", layout="wide")

st.title("📚 Dictionary Classification Bot")
st.write("Upload text data and quickly test a keyword dictionary. Now survives spreadsheet oddities like **#REF!** in the ground‑truth column.")

# 1. Data upload / preview ------------------------------------------------------

uploaded_file = st.file_uploader("Step 1 – Upload a CSV file", type=["csv"])  # noqa: E501

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.success(f"Loaded **{len(df_raw):,}** rows.")
    st.dataframe(df_raw.head(), use_container_width=True, height=150)

    text_col = st.selectbox("Step 2 – Choose the text column", df_raw.columns)
    gt_options = ["<None>"] + df_raw.columns.tolist()
    gt_col = st.selectbox("Ground‑truth column (optional)", gt_options)
else:
    df_raw = None
    text_col = None
    gt_col = "<None>"

# 2. Dictionary input -----------------------------------------------------------

st.subheader("Step 3 – Enter keywords (one per line)")
initial = """custom
bespoke
made‑to‑measure""" if not st.session_state.get("seeded") else ""

keywords_text = st.text_area("Keywords", value=initial, height=150)
keywords = [kw.strip() for kw in keywords_text.splitlines() if kw.strip()]

# 3. Classification -------------------------------------------------------------

if st.button("🚀 Classify statements"):
    if df_raw is None or not text_col or not keywords:
        st.error("Please upload data, choose a text column, and enter at least one keyword.")
    else:
        res_df, metrics = classify(
            df_raw,
            text_col,
            None if gt_col == "<None>" else gt_col,
            keywords,
        )

        # Save in session_state for follow‑on analysis
        st.session_state["results"] = res_df
        st.session_state["metrics"] = metrics

        st.subheader("Results")
        if metrics:
            st.write(metrics)
        st.dataframe(res_df.head(50), use_container_width=True)

        # Download link
        csv = res_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download full results as CSV", csv, "dictionary_classification_results.csv", "text/csv")

# 4. Keyword impact analysis ----------------------------------------------------

if "results" in st.session_state and st.button("🔍 Analyze keyword impact"):
    res_df = st.session_state["results"]
    # Build per‑keyword recall table
    impact = []
    for kw in keywords:
        pattern = safe_regex(kw)
        matched = res_df[res_df.text.str.lower().str.contains(pattern)]
        gt_available = matched.dropna(subset=["gt"])
        tp = ((gt_available.pred == 1) & (gt_available.gt == 1)).sum()
        recall = tp / ((res_df.gt == 1).sum() or 1)
        impact.append({"keyword": kw, "match_count": len(matched), "recall": round(recall, 4)})

    impact_df = pd.DataFrame(impact).sort_values("recall", ascending=False)
    st.dataframe(impact_df, use_container_width=True)
