# dictionary_classification_bot.py
"""
Streamlit implementation of the Dictionary Classification Bot originally specified as a React/Tailwind
application.  The app lets users upload or paste CSV data, supply a keyword dictionary, and obtain word‑
boundary‑aware, case‑insensitive binary classifications together with performance metrics and keyword
impact analysis.

Core workflow (see Technical Specs §8) :
 1. Load data  ➜  2. Select columns  ➜  3. Enter keywords  ➜  4. Run classification  ➜  5. Review
    results  ➜  6. Analyse keywords  ➜  7. Export results.

The code mirrors the feature set and limits in the specification: up to 10 000 rows and 1 000 keywords
for optimal performance (§10.2). All processing is client‑side; no data are persisted.
"""

from __future__ import annotations

import io
import re
import textwrap
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

###############################################################################
# ---------------------------- Utility functions ---------------------------- #
###############################################################################


def parse_keywords(text: str) -> List[str]:
    """Convert a comma‑separated string (quoted or unquoted) into a clean list of
    unique keywords.
    """
    if not text:
        return []
    # Remove new‑lines, then split on commas.
    raw_parts = [p.strip() for p in text.replace("\n", " ").split(",")]
    cleaned = [p.strip("\"' ") for p in raw_parts if p]
    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for kw in cleaned:
        low = kw.lower()
        if low and low not in seen:
            seen.add(low)
            unique.append(kw)
    return unique


def compile_keyword_patterns(keywords: List[str]) -> Dict[str, re.Pattern]:
    """Compile case‑insensitive, word‑boundary regex patterns for each keyword."""
    patterns = {}
    for kw in keywords:
        escaped = re.escape(kw.lower())
        pattern = re.compile(rf"\b{escaped}\b", flags=re.IGNORECASE)
        patterns[kw] = pattern
    return patterns


def classify_statement(stmt: str, patterns: Dict[str, re.Pattern]) -> Tuple[int, List[str]]:
    """Return binary prediction (1/0) and list of matched keywords for a statement."""
    matched = [kw for kw, pat in patterns.items() if pat.search(stmt)]
    return (1 if matched else 0, matched)


def compute_overall_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate accuracy, precision, recall, and F1 as specified in §4.4."""
    tp = ((df["predicted"] == 1) & (df["groundTruth"] == 1)).sum()
    fp = ((df["predicted"] == 1) & (df["groundTruth"] == 0)).sum()
    tn = ((df["predicted"] == 0) & (df["groundTruth"] == 0)).sum()
    fn = ((df["predicted"] == 0) & (df["groundTruth"] == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(df) if len(df) else 0.0

    return dict(tp=tp, fp=fp, tn=tn, fn=fn, accuracy=accuracy, precision=precision, recall=recall, f1=f1)


def keyword_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per‑keyword TP/FP counts and derived metrics; return sorted DataFrame."""
    stats = defaultdict(lambda: dict(tp=0, fp=0))
    for _, row in df.iterrows():
        truth = row["groundTruth"]
        for kw in row["matchedKeywords"]:
            if truth == 1 and row["predicted"] == 1:
                stats[kw]["tp"] += 1
            elif truth == 0 and row["predicted"] == 1:
                stats[kw]["fp"] += 1

    records = []
    for kw, s in stats.items():
        tp, fp = s["tp"], s["fp"]
        recall = tp / tp if tp else 0  # per‑keyword recall relative to itself
        precision = tp / (tp + fp) if (tp + fp) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        records.append(dict(keyword=kw, truePositives=tp, falsePositives=fp,
                            recall=recall, precision=precision, f1=f1))

    return pd.DataFrame(records)


def to_csv_download(df: pd.DataFrame) -> Tuple[str, str]:
    """Return (filename, csv‑string) tuple suitable for st.download_button."""
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return "keyword_analysis.csv", csv_buffer.getvalue()

###############################################################################
# ---------------------------- Streamlit interface --------------------------- #
###############################################################################

def main():
    st.set_page_config(page_title="Dictionary Classification Bot (Streamlit)", layout="wide")
    st.title("📝 Dictionary Classification Bot")
    st.caption("Keyword‑driven text classification with real‑time performance analytics.")

    # 1. Data Input Section -----------------------------------------------------------------
    st.header("1  📂 Load CSV Data")
    with st.expander("Upload or paste CSV", expanded=True):
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")
        pasted_text = st.text_area("…or paste CSV data here", height=150, key="csv_paste")
        if st.button("Load sample data"):
            pasted_text = """ID,Statement,Answer\n1,50% off sale on all shoes!,1\n2,Looking forward to the weekend ride.,0\n3,Exclusive discount just for you,1\n4,Our new blog post is live,0"""
            st.session_state["csv_paste"] = pasted_text

    data_df: pd.DataFrame | None = None
    error_msg = None
    if uploaded_file or pasted_text.strip():
        try:
            csv_source = uploaded_file.read().decode("utf‑8") if uploaded_file else pasted_text
            data_df = pd.read_csv(io.StringIO(csv_source))
        except Exception as e:
            error_msg = f"CSV parse error: {e}"

    if error_msg:
        st.error(error_msg)
        st.stop()

    if data_df is not None:
        # 2. Column Selection ----------------------------------------------------------------
        st.header("2  📑 Select Columns")
        with st.form("column_select_form"):
            text_col = st.selectbox("Text column", options=list(data_df.columns), index=next((i for i, c in enumerate(data_df.columns) if c.lower() in ("statement", "text")), 0))
            truth_col = st.selectbox("Ground‑truth column (0/1)", options=list(data_df.columns), index=next((i for i, c in enumerate(data_df.columns) if c.lower() in ("answer", "label", "truth")), 1 if len(data_df.columns) > 1 else 0))
            submitted = st.form_submit_button("Confirm columns")
            if not submitted:
                st.stop()

        # 3. Keyword Input -------------------------------------------------------------------
        st.header("3  🗝️ Enter Keywords")
        default_kw_placeholder = "keyword1, keyword2, \"keyword3 with spaces\""
        kw_input = st.text_area("Comma‑separated keywords (quoted or unquoted)", placeholder=default_kw_placeholder, height=120, key="kw_input")
        kw_list = parse_keywords(kw_input)
        st.write(f"**{len(kw_list)}** keywords parsed.")
        if len(kw_list) > 1000:
            st.warning("Only the first 1000 keywords will be used for performance reasons.")
            kw_list = kw_list[:1000]

        # 4. Classification ------------------------------------------------------------------
        if st.button("🚀 Classify Statements", disabled=(data_df is None or not kw_list)):
            with st.spinner("Running classification…"):
                patterns = compile_keyword_patterns(kw_list)
                preds, matched_kw = [], []
                for stmt in data_df[text_col].astype(str):
                    pred, m = classify_statement(stmt, patterns)
                    preds.append(pred)
                    matched_kw.append(m)

                data_df = data_df.copy()
                data_df["predicted"] = preds
                data_df["groundTruth"] = data_df[truth_col].astype(int)
                data_df["matchedKeywords"] = matched_kw
                data_df["category"] = data_df.apply(lambda r: (
                    "TP" if r.predicted == 1 and r.groundTruth == 1 else
                    "FP" if r.predicted == 1 and r.groundTruth == 0 else
                    "FN" if r.predicted == 0 and r.groundTruth == 1 else
                    "TN"), axis=1)

                # Store in session_state for later analysis.
                st.session_state["classified_df"] = data_df

        if "classified_df" not in st.session_state:
            st.info("Provide keywords and click **Classify Statements** to continue.")
            st.stop()

        classified_df: pd.DataFrame = st.session_state["classified_df"]

        # 5. Results ------------------------------------------------------------------------
        st.header("4  📊 Results Summary")
        metrics = compute_overall_metrics(classified_df)
        m_cols = st.columns(4)
        m_cols[0].metric("Accuracy", f"{metrics['accuracy']*100:,.2f}%")
        m_cols[1].metric("Precision", f"{metrics['precision']*100:,.2f}%")
        m_cols[2].metric("Recall", f"{metrics['recall']*100:,.2f}%")
        m_cols[3].metric("F1 Score", f"{metrics['f1']*100:,.2f}%")

        with st.expander("Confusion Matrix & Examples", expanded=False):
            st.write(f"**TP:** {metrics['tp']}  **FP:** {metrics['fp']}  **TN:** {metrics['tn']}  **FN:** {metrics['fn']}")
            fp_examples = classified_df[classified_df.category == "FP"][text_col].head(10)
            fn_examples = classified_df[classified_df.category == "FN"][text_col].head(10)
            st.subheader("False Positive Examples")
            for txt in fp_examples:
                st.markdown(f"- {textwrap.shorten(txt, width=50, placeholder='…')}")
            st.subheader("False Negative Examples")
            for txt in fn_examples:
                st.markdown(f"- {textwrap.shorten(txt, width=50, placeholder='…')}")

        # 6. Keyword Impact Analysis --------------------------------------------------------
        st.header("5  🔎 Keyword Impact Analysis")
        impact_df = keyword_impact(classified_df)
        if impact_df.empty:
            st.info("No keywords matched any statements.")
        else:
            tabs = st.tabs(["By Recall", "By Precision", "By F1 Score"])
            for col_name, tab in zip(["recall", "precision", "f1"], tabs):
                with tab:
                    top10 = impact_df.sort_values(col_name, ascending=False).head(10)
                    st.dataframe(top10.reset_index(drop=True), use_container_width=True)

            fname, csv_content = to_csv_download(impact_df)
            st.download_button("📥 Download keyword analysis CSV", csv_content, file_name=fname, mime="text/csv")

        # 7. Export Classification Results --------------------------------------------------
        st.header("6  💾 Export Results")
        res_fname, res_csv = to_csv_download(classified_df)
        st.download_button("📥 Download full results CSV", res_csv, file_name=res_fname, mime="text/csv")

        with st.expander("Preview first 20 rows"):
            st.dataframe(classified_df.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
