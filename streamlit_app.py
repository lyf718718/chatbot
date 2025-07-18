import streamlit as st
import pandas as pd
import re
import io

# -------------------------------------------------
#  Streamlit Dictionary Classification Bot
# -------------------------------------------------
#  A Python conversion of the original React app
#  for interactive dictionary‑based text classification
# -------------------------------------------------

st.set_page_config(page_title="Dictionary Classification Bot", layout="wide")

# ------------------
# Helper functions
# ------------------

def escape_regex(token: str) -> str:
    """Escape special regex characters in a keyword."""
    return re.sub(r"[.*+?^${}()|[\]\\]", lambda m: f"\\{m.group(0)}", token)


def safe_regex(keyword: str):
    return re.compile(rf"\b{escape_regex(keyword.lower())}\b")


def parse_csv(text: str) -> pd.DataFrame:
    """Read CSV text into a DataFrame."""
    return pd.read_csv(io.StringIO(text))


def classify(df: pd.DataFrame, text_col: str, gt_col: str | None, dictionary: list[str]):
    """Run keyword‑based classification and compute metrics."""
    rows = []
    tp = fp = tn = fn = 0

    for _, row in df.iterrows():
        statement = str(row[text_col]).lower()
        matched = [kw for kw in dictionary if safe_regex(kw).search(statement)]
        pred = int(bool(matched))
        gt = int(row[gt_col]) if gt_col else None

        category = None
        if gt_col:
            if pred == 1 and gt == 1:
                tp += 1
                category = "TP"
            elif pred == 1 and gt == 0:
                fp += 1
                category = "FP"
            elif pred == 0 and gt == 1:
                fn += 1
                category = "FN"
            else:
                tn += 1
                category = "TN"

        rows.append(
            {
                "statement": row[text_col],
                "predicted": pred,
                "ground_truth": gt,
                "category": category,
                "matched_keywords": ", ".join(matched),
            }
        )

    res_df = pd.DataFrame(rows)

    metrics = None
    if gt_col:
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        }

    return res_df, metrics


def keyword_impact(df: pd.DataFrame, text_col: str, gt_col: str, dictionary: list[str]):
    """Compute recall, precision & F1 for each keyword (top‑10 by metric)."""
    impact = []
    for kw in dictionary:
        regex = safe_regex(kw)
        tp = fp = pos_total = 0
        tp_examples, fp_examples = [], []

        for _, row in df.iterrows():
            contains = bool(regex.search(str(row[text_col]).lower()))
            gt = int(row[gt_col])
            if gt == 1:
                pos_total += 1
                if contains:
                    tp += 1
                    if len(tp_examples) < 3:
                        tp_examples.append(row[text_col])
            elif gt == 0 and contains:
                fp += 1
                if len(fp_examples) < 3:
                    fp_examples.append(row[text_col])

        recall = tp / pos_total if pos_total else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0

        impact.append(
            {
                "keyword": kw,
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "tp_examples": tp_examples,
                "fp_examples": fp_examples,
            }
        )

    top_by_recall = sorted(impact, key=lambda d: d["recall"], reverse=True)[:10]
    top_by_precision = sorted(impact, key=lambda d: d["precision"], reverse=True)[:10]
    top_by_f1 = sorted(impact, key=lambda d: d["f1"], reverse=True)[:10]

    return {
        "recall": top_by_recall,
        "precision": top_by_precision,
        "f1": top_by_f1,
    }


# ------------------
# Sample CSV (same as React demo)
# ------------------
SAMPLE_CSV = """ID,Statement,Answer
1,"It's SPRING TRUNK SHOW week!",1
2,"I am offering 4 shirts styled the way you want and the 5th is using MAGNETIC COLLAR STAY to help keep your collars in place!",1
3,"In recognition of Earth Day, I would like to showcase our collection of Earth Fibers!",0
4,"It is now time to do some wardrobe crunches, and check your basics! Never on sale.",1
5,"He's a hard worker and always willing to lend a hand. The prices are the best I've seen in 17 years of servicing my clients.",0
"""


# =================================================
# UI LAYOUT
# =================================================

st.title("Dictionary Classification Bot")

# -- STEP 1 : Load data ------------------------------------------------------
with st.expander("① Upload or paste CSV data"):
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    csv_text = st.text_area("…or paste CSV content here", height=150)
    if st.button("Load sample CSV"):
        csv_text = SAMPLE_CSV
        st.session_state["csv_text"] = csv_text

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
    elif csv_text:
        df_raw = parse_csv(csv_text)
    else:
        df_raw = None

    if df_raw is not None:
        st.success(f"Loaded {len(df_raw)} rows.")
        st.dataframe(df_raw.head())

# -- STEP 2 : Select columns --------------------------------------------------
if df_raw is not None:
    columns = df_raw.columns.tolist()
    text_col = st.selectbox("Text column for analysis", columns, index=columns.index("Statement") if "Statement" in columns else 0)
    gt_col = st.selectbox("Ground‑truth column (optional, 0/1)", ["None"] + columns, index=(columns.index("Answer") + 1) if "Answer" in columns else 0)
    gt_col = None if gt_col == "None" else gt_col
else:
    text_col = gt_col = None

# -- STEP 3 : Dictionary ------------------------------------------------------
with st.expander("② Enter / edit keyword dictionary"):
    dict_text = st.text_area('Keywords (comma‑separated or "word1","word2")', height=100)
    if st.button("Save dictionary"):
        kw_in_quotes = re.findall(r'"([^"]+)"', dict_text)
        keywords = kw_in_quotes or [k.strip() for k in dict_text.split(",")]
        keywords = [k for k in keywords if k]
        st.session_state["dictionary"] = keywords

    if "dictionary" in st.session_state:
        st.markdown(f"**{len(st.session_state['dictionary'])} keywords:** {', '.join(st.session_state['dictionary'])}")

# -- STEP 4 : Classification --------------------------------------------------
if st.button("③ Classify statements") and df_raw is not None and text_col and st.session_state.get("dictionary"):
    result_df, metrics = classify(df_raw, text_col, gt_col, st.session_state["dictionary"])
    st.session_state["results"] = result_df
    st.session_state["metrics"] = metrics

# -- Display results ----------------------------------------------------------
if "metrics" in st.session_state and st.session_state["metrics"]:
    m = st.session_state["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
    c2.metric("Precision", f"{m['precision']*100:.2f}%")
    c3.metric("Recall", f"{m['recall']*100:.2f}%")
    c4.metric("F1 Score", f"{m['f1']*100:.2f}%")

    st.write(f"TP: {m['TP']} | FP: {m['FP']} | FN: {m['FN']} | TN: {m['TN']}")

    res_df = st.session_state["results"]
    fp_df = res_df[res_df["category"] == "FP"]
    fn_df = res_df[res_df["category"] == "FN"]

    with st.expander(f"False Positives ({len(fp_df)})"):
        st.table(fp_df[["statement", "matched_keywords"]].head(50))

    with st.expander(f"False Negatives ({len(fn_df)})"):
        st.table(fn_df[["statement"]].head(50))

    # -- Keyword impact ------------------------------------------------------
    if gt_col:
        if st.button("④ Analyze keyword impact"):
            analysis = keyword_impact(res_df, "statement", "ground_truth", st.session_state["dictionary"])
            st.session_state["analysis"] = analysis

# -- Display keyword impact ---------------------------------------------------
if "analysis" in st.session_state:
    analysis = st.session_state["analysis"]
    metric_choice = st.selectbox("Sort top‑10 by", ["recall", "precision", "f1"], key="metric_choice")
    top_list = analysis[metric_choice]

    st.subheader(f"Top 10 keywords by {metric_choice}")
    display_df = pd.DataFrame(
        [
            {
                "Keyword": d["keyword"],
                "Recall %": d["recall"] * 100,
                "Precision %": d["precision"] * 100,
                "F1 %": d["f1"] * 100,
                "True Pos": d["tp"],
                "False Pos": d["fp"],
            }
            for d in top_list
        ]
    )
    st.table(display_df)

    # Download button
    csv_buf = io.StringIO()
    display_df.to_csv(csv_buf, index=False)
    st.download_button(
        "Download CSV", csv_buf.getvalue(), file_name="keyword_impact.csv", mime="text/csv"
    )
