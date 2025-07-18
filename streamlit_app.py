# dictionary_classification_bot.py
"""
Streamlit implementation of the **Dictionary Classification Bot**.  
Fix rev‑1 (2025‑07‑18):
•  Persist *column selection* in `st.session_state` so the app no longer bounces the user back to the
  “Enter Keywords” step after each rerun.  
•  Removed the hard *stop()* gate that caused the loop and added a visual confirmation banner.  
•  Minor: renamed a few session‑state keys for clarity and added helpful status messaging.
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
    """Convert a comma‑separated string into a list of unique, stripped keywords."""
    if not text:
        return []
    parts = [p.strip().strip('"\' ') for p in text.replace("\n", " ").split(',') if p.strip()]
    seen, clean = set(), []
    for kw in parts:
        low = kw.lower()
        if low not in seen:
            seen.add(low)
            clean.append(kw)
    return clean

def compile_keyword_patterns(keywords: List[str]) -> Dict[str, re.Pattern]:
    return {kw: re.compile(rf"\b{re.escape(kw)}\b", flags=re.IGNORECASE) for kw in keywords}

def classify_statement(stmt: str, patterns: Dict[str, re.Pattern]) -> Tuple[int, List[str]]:
    hits = [kw for kw, pat in patterns.items() if pat.search(stmt)]
    return (1 if hits else 0, hits)

def compute_overall_metrics(df: pd.DataFrame) -> Dict[str, float]:
    tp = ((df.predicted == 1) & (df.groundTruth == 1)).sum()
    fp = ((df.predicted == 1) & (df.groundTruth == 0)).sum()
    tn = ((df.predicted == 0) & (df.groundTruth == 0)).sum()
    fn = ((df.predicted == 0) & (df.groundTruth == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    acc  = (tp + tn) / len(df) if len(df) else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=prec, recall=rec, f1=f1, accuracy=acc)

def keyword_impact(df: pd.DataFrame) -> pd.DataFrame:
    stats = defaultdict(lambda: dict(tp=0, fp=0))
    for _, r in df.iterrows():
        for kw in r.matchedKeywords:
            if r.predicted == 1 and r.groundTruth == 1:
                stats[kw]['tp'] += 1
            elif r.predicted == 1 and r.groundTruth == 0:
                stats[kw]['fp'] += 1
    rows = []
    for kw, s in stats.items():
        tp, fp = s['tp'], s['fp']
        prec = tp/(tp+fp) if (tp+fp) else 0
        rec  = 1.0 if tp else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        rows.append(dict(keyword=kw, truePositives=tp, falsePositives=fp,
                         precision=prec, recall=rec, f1=f1))
    return pd.DataFrame(rows)

def to_csv_download(df: pd.DataFrame) -> Tuple[str, str]:
    buf = io.StringIO(); df.to_csv(buf, index=False); return 'results.csv', buf.getvalue()

###############################################################################
# ---------------------------- Streamlit interface --------------------------- #
###############################################################################

def main():
    st.set_page_config('Dictionary Classification Bot', layout='wide')
    st.title('📝 Dictionary Classification Bot')

    # ---- 1 · Load data ------------------------------------------------------ #
    with st.expander('1  📂 Load CSV', expanded=True):
        up_file = st.file_uploader('Upload CSV', type='csv')
        txt_area = st.text_area('…or paste CSV data here', height=120, key='paste')
        if st.button('Load sample'):
            sample = """ID,Statement,Answer\n1,50% off shoes!,1\n2,Weekend ride.,0\n3,Custom jersey for you,1\n4,Our blog is live,0"""
            st.session_state.paste = sample

    if up_file or txt_area.strip():
        try:
            csv_raw = up_file.read().decode() if up_file else txt_area
            df = pd.read_csv(io.StringIO(csv_raw))
            st.session_state['raw_df'] = df
        except Exception as e:
            st.error(f'CSV parse error: {e}')
            return
    if 'raw_df' not in st.session_state:
        st.info('Upload or paste a CSV to continue.')
        return

    df = st.session_state['raw_df']

    # ---- 2 · Column selection ---------------------------------------------- #
    st.header('2  📑 Select Columns')
    if 'cols_confirmed' not in st.session_state:
        with st.form('col_form'):
            text_c  = st.selectbox('Text column',  df.columns, index=next((i for i,c in enumerate(df.columns) if c.lower() in ('statement','text')),0))
            truth_c = st.selectbox('Ground‑truth column (0/1)', df.columns, index=next((i for i,c in enumerate(df.columns) if c.lower() in ('answer','label','truth')),1 if len(df.columns)>1 else 0))
            if st.form_submit_button('Confirm columns'):
                st.session_state['text_col']  = text_c
                st.session_state['truth_col'] = truth_c
                st.session_state['cols_confirmed'] = True
                st.experimental_rerun()
        st.stop()

    text_col  = st.session_state['text_col']
    truth_col = st.session_state['truth_col']
    st.success(f'Using **{text_col}** as text and **{truth_col}** as ground‑truth.')

    # ---- 3 · Keywords ------------------------------------------------------- #
    st.header('3  🗝️ Enter Keywords')
    kw_input = st.text_area('Comma‑separated keywords', height=120)
    kw_list  = parse_keywords(kw_input)
    st.write(f'**{len(kw_list)}** keywords parsed.')

    if st.button('🚀 Classify Statements', disabled=not kw_list):
        with st.spinner('Classifying…'):
            pats = compile_keyword_patterns(kw_list)
            preds, matches = zip(*(classify_statement(s, pats) for s in df[text_col].astype(str)))
            out_df = df.copy()
            out_df['predicted'] = preds
            out_df['groundTruth'] = df[truth_col].astype(int)
            out_df['matchedKeywords'] = matches
            out_df['category'] = out_df.apply(lambda r: 'TP' if (r.predicted==r.groundTruth==1) else
                                                          'FP' if (r.predicted==1 and r.groundTruth==0) else
                                                          'FN' if (r.predicted==0 and r.groundTruth==1) else 'TN', axis=1)
            st.session_state['classified_df'] = out_df
            st.experimental_rerun()

    if 'classified_df' not in st.session_state:
        st.info('Provide keywords and click **Classify Statements** to continue.')
        return

    cls_df = st.session_state['classified_df']

    # ---- 4 · Results summary ------------------------------------------------ #
    st.header('4  📊 Results Summary')
    m = compute_overall_metrics(cls_df)
    cols = st.columns(4)
    cols[0].metric('Accuracy',  f"{m['accuracy']*100:0.2f}%")
    cols[1].metric('Precision', f"{m['precision']*100:0.2f}%")
    cols[2].metric('Recall',    f"{m['recall']*100:0.2f}%")
    cols[3].metric('F1',        f"{m['f1']*100:0.2f}%")

    with st.expander('Confusion details'):
        st.write(f"TP {m['tp']} | FP {m['fp']} | TN {m['tn']} | FN {m['fn']}")
        st.subheader('False Positives')
        for t in cls_df[cls_df.category=='FP'][text_col].head(10):
            st.markdown(' • '+textwrap.shorten(t,40,'…'))
        st.subheader('False Negatives')
        for t in cls_df[cls_df.category=='FN'][text_col].head(10):
            st.markdown(' • '+textwrap.shorten(t,40,'…'))

    # ---- 5 · Keyword impact ------------------------------------------------- #
    st.header('5  🔎 Keyword Impact')
    imp = keyword_impact(cls_df)
    if imp.empty:
        st.info('No keyword hits.')
    else:
        for coln in ('recall','precision','f1'):
            st.subheader(f'Top 10 by {coln.capitalize()}')
            st.dataframe(imp.sort_values(coln,ascending=False).head(10), use_container_width=True)
        name, csv = to_csv_download(imp)
        st.download_button('📥 Download keyword analysis CSV', csv, file_name='keyword_impact.csv', mime='text/csv')

    # ---- 6 · Export full results ------------------------------------------- #
    st.header('6  💾 Export Results')
    name, csv = to_csv_download(cls_df)
    st.download_button('📥 Download full results CSV', csv, file_name='classified_results.csv', mime='text/csv')
    with st.expander('Preview first 20 rows'):
        st.dataframe(cls_df.head(20), use_container_width=True)


if __name__ == '__main__':
    main()
