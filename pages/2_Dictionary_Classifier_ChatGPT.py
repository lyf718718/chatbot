"""
Streamlit Marketing-Language Classifier
======================================
Upload a CSV that contains a **Statement** column, tweak the category dictionaries
however you like, and instantly see which rows contain urgency- or exclusivity-
style marketing language (or any new categories you add!).

Run locally with:
    streamlit run marketing_classifier_app.py
"""

from __future__ import annotations

import io
import re
from typing import Dict, Set

import pandas as pd
import streamlit as st

###############################################################################
# 1. Default dictionaries (feel free to extend!)
###############################################################################
DEFAULT_DICTIONARIES: Dict[str, Set[str]] = {
    "urgency_marketing": {
        "limited",
        "limited time",
        "limited run",
        "limited edition",
        "order now",
        "last chance",
        "hurry",
        "while supplies last",
        "before they're gone",
        "selling out",
        "selling fast",
        "act now",
        "don't wait",
        "today only",
        "expires soon",
        "final hours",
        "almost gone",
    },
    "exclusive_marketing": {
        "exclusive",
        "exclusively",
        "exclusive offer",
        "exclusive deal",
        "members only",
        "vip",
        "special access",
        "invitation only",
        "premium",
        "privileged",
        "limited access",
        "select customers",
        "insider",
        "private sale",
        "early access",
    },
}

def _compile_patterns(dictionary: Dict[str, Set[str]]) -> Dict[str, list[re.Pattern]]:
    """Compile regex patterns for every phrase in every category."""
    return {
        category: [
            re.compile(rf"\b{re.escape(phrase)}\b", flags=re.IGNORECASE)
            for phrase in phrases
        ]
        for category, phrases in dictionary.items()
    }

def _contains_phrase(text: str | float, patterns: list[re.Pattern]) -> bool:
    """Return *True* if *any* regex pattern is found in *text*."""
    if pd.isna(text):
        return False
    text = str(text).lower()
    return any(pat.search(text) for pat in patterns)


def classify_dataframe(df: pd.DataFrame, dictionary: Dict[str, Set[str]]) -> pd.DataFrame:
    """Add a boolean column per category indicating the presence of a phrase."""
    compiled = _compile_patterns(dictionary)

    for category, pats in compiled.items():
        df[category] = df["Statement"].apply(lambda x: _contains_phrase(x, pats))

    return df


###############################################################################
# Streamlit UI
###############################################################################

st.set_page_config(
    page_title="Marketing-Language Classifier",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Marketing-Language Classifier")
st.markdown(
    "Upload a CSV containing a **`Statement`** column. Edit the dictionaries in the “Categories & Phrases” sidebar, then click **Classify** to tag each row."
)

# ---- Sidebar: dictionary editor ------------------------------------------------
with st.sidebar:
    st.header("🗂️ Categories & Phrases")

    # Initialise session-state on first run
    if "dictionary" not in st.session_state:
        st.session_state.dictionary = {k: set(v) for k, v in DEFAULT_DICTIONARIES.items()}

    # Existing categories
    for cat in list(st.session_state.dictionary.keys()):
        with st.expander(f"✏️ {cat}"):
            phrases = st.text_area(
                "One phrase per line", "\n".join(sorted(st.session_state.dictionary[cat])), key=f"ta_{cat}"
            )
            st.session_state.dictionary[cat] = {
                p.strip() for p in phrases.splitlines() if p.strip()
            }
            # Option to delete a category entirely
            if st.button("Delete category", key=f"del_{cat}"):
                st.session_state.dictionary.pop(cat)
                st.experimental_rerun()

    st.divider()
    st.subheader("➕ Add new category")
    new_cat = st.text_input("Category name", placeholder="e.g. scarcity_marketing")
    new_phrases = st.text_area("Phrases (one per line)", key="new_phrases")
    if st.button("Add category"):
        if not new_cat:
            st.warning("Please provide a category name.")
        elif new_cat in st.session_state.dictionary:
            st.warning("Category already exists.")
        else:
            st.session_state.dictionary[new_cat] = {
                p.strip() for p in new_phrases.splitlines() if p.strip()
            }
            st.success(f"Added category '{new_cat}'.")
            st.experimental_rerun()

# ---- Main area: file upload & classification -----------------------------------

uploaded_file = st.file_uploader("📄 Upload CSV", type=["csv"], accept_multiple_files=False)

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        st.stop()

    if "Statement" not in df_input.columns:
        st.error("❌ The CSV must contain a column named **`Statement`**.")
        st.stop()

    st.success(f"Loaded **{len(df_input):,}** rows.")

    if st.button("🚀 Classify"):
        with st.spinner("Running classification…"):
            df_out = classify_dataframe(df_input.copy(), st.session_state.dictionary)
        st.success("Classification complete!")

        st.subheader("Preview of classified data")
        st.dataframe(df_out.head(50), use_container_width=True)

        # Prepare download
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download full CSV",
            data=csv_bytes,
            file_name="classified_data.csv",
            mime="text/csv",
        )
else:
    st.info("👈 Upload a CSV to get started.")
