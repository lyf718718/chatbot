import streamlit as st
import pandas as pd
import re
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="Text Classification App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default dictionaries
DEFAULT_DICTIONARIES = {
    'urgency_marketing': {
        'limited', 'limited time', 'limited run', 'limited edition', 'order now',
        'last chance', 'hurry', 'while supplies last', 'before they\'re gone',
        'selling out', 'selling fast', 'act now', 'don\'t wait', 'today only',
        'expires soon', 'final hours', 'almost gone'
    },
    'exclusive_marketing': {
        'exclusive', 'exclusively', 'exclusive offer', 'exclusive deal',
        'members only', 'vip', 'special access', 'invitation only',
        'premium', 'privileged', 'limited access', 'select customers',
        'insider', 'private sale', 'early access'
    }
}

def classify_text(text, dictionaries):
    """Classify text based on dictionary matches"""
    if pd.isna(text):
        return {}
    
    text_lower = text.lower()
    results = {}
    
    for category, keywords in dictionaries.items():
        # Count matches for each category
        matches = []
        for keyword in keywords:
            if keyword.lower() in text_lower:
                matches.append(keyword)
        
        results[category] = {
            'count': len(matches),
            'matches': matches,
            'present': len(matches) > 0
        }
    
    return results

def process_dataframe(df, text_column, dictionaries):
    """Process the dataframe with classification"""
    # Apply classification
    df['classification'] = df[text_column].apply(lambda x: classify_text(x, dictionaries))
    
    # Extract results into separate columns
    for category in dictionaries.keys():
        df[f'{category}_count'] = df['classification'].apply(lambda x: x.get(category, {}).get('count', 0))
        df[f'{category}_present'] = df['classification'].apply(lambda x: x.get(category, {}).get('present', False))
        df[f'{category}_matches'] = df['classification'].apply(lambda x: x.get(category, {}).get('matches', []))
    
    return df

def main():
    st.title("📊 Text Classification App")
    st.markdown("Upload your dataset and classify text using customizable dictionaries!")
    
    # Sidebar for dictionary management
    with st.sidebar:
        st.header("📚 Dictionary Management")
        
        # Initialize session state for dictionaries
        if 'dictionaries' not in st.session_state:
            st.session_state.dictionaries = DEFAULT_DICTIONARIES.copy()
        
        # Dictionary editing
        st.subheader("Edit Categories")
        
        # Add new category
        with st.expander("➕ Add New Category"):
            new_category = st.text_input("Category Name", key="new_category")
            new_keywords = st.text_area("Keywords (one per line)", key="new_keywords")
            
            if st.button("Add Category"):
                if new_category and new_keywords:
                    keywords_set = set(line.strip().lower() for line in new_keywords.split('\n') if line.strip())
                    st.session_state.dictionaries[new_category] = keywords_set
                    st.success(f"Added category: {new_category}")
                    st.rerun()
        
        # Edit existing categories
        for category in list(st.session_state.dictionaries.keys()):
            with st.expander(f"✏️ Edit {category}"):
                # Convert set to string for editing
                current_keywords = '\n'.join(sorted(st.session_state.dictionaries[category]))
                
                edited_keywords = st.text_area(
                    f"Keywords for {category}",
                    value=current_keywords,
                    key=f"edit_{category}",
                    height=150
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Update", key=f"update_{category}"):
                        keywords_set = set(line.strip().lower() for line in edited_keywords.split('\n') if line.strip())
                        st.session_state.dictionaries[category] = keywords_set
                        st.success(f"Updated {category}")
                        st.rerun()
                
                with col2:
                    if st.button(f"Delete", key=f"delete_{category}"):
                        del st.session_state.dictionaries[category]
                        st.success(f"Deleted {category}")
                        st.rerun()
        
        # Reset to default
        if st.button("🔄 Reset to Default"):
            st.session_state.dictionaries = DEFAULT_DICTIONARIES.copy()
            st.success("Reset to default dictionaries")
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Upload Dataset")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file containing the text data you want to classify"
        )
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                # Column selection
                st.subheader("📋 Select Text Column")
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
                
                if text_columns:
                    selected_column = st.selectbox(
                        "Choose the column containing text to classify:",
                        text_columns,
                        help="Select the column that contains the text you want to analyze"
                    )
                    
                    # Preview original data
                    st.subheader("👀 Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Process button
                    if st.button("🚀 Process Classification", type="primary"):
                        with st.spinner("Processing classification..."):
                            # Process the data
                            processed_df = process_dataframe(df, selected_column, st.session_state.dictionaries)
                            
                            # Store in session state
                            st.session_state.processed_df = processed_df
                            st.session_state.selected_column = selected_column
                            
                            st.success("✅ Classification completed!")
                else:
                    st.error("❌ No text columns found in the uploaded file.")
                    
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    with col2:
        st.header("📊 Current Categories")
        
        # Display current dictionaries
        for category, keywords in st.session_state.dictionaries.items():
            with st.expander(f"{category} ({len(keywords)} keywords)"):
                st.write(", ".join(sorted(keywords)[:10]))
                if len(keywords) > 10:
                    st.write(f"... and {len(keywords) - 10} more")
    
    # Results section
    if hasattr(st.session_state, 'processed_df'):
        st.header("📈 Classification Results")
        
        processed_df = st.session_state.processed_df
        selected_column = st.session_state.selected_column
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        
        summary_data = []
        for category in st.session_state.dictionaries.keys():
            count_col = f'{category}_count'
            present_col = f'{category}_present'
            
            if present_col in processed_df.columns:
                total_detected = processed_df[present_col].sum()
                percentage = (total_detected / len(processed_df)) * 100
                avg_matches = processed_df[count_col].mean()
                
                summary_data.append({
                    'Category': category,
                    'Texts with Category': total_detected,
                    'Percentage': f"{percentage:.1f}%",
                    'Avg Matches per Text': f"{avg_matches:.2f}"
                })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Detailed results
        st.subheader("🔍 Detailed Results")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_all = st.checkbox("Show all rows", value=True)
        
        with col2:
            if not show_all:
                category_filter = st.selectbox(
                    "Filter by category:",
                    ['All'] + list(st.session_state.dictionaries.keys())
                )
        
        with col3:
            max_rows = st.number_input("Max rows to display", min_value=10, max_value=1000, value=100)
        
        # Prepare display dataframe
        display_columns = [selected_column]
        for category in st.session_state.dictionaries.keys():
            display_columns.extend([f'{category}_count', f'{category}_present', f'{category}_matches'])
        
        display_df = processed_df[display_columns].copy()
        
        # Apply filters
        if not show_all and 'category_filter' in locals() and category_filter != 'All':
            filter_column = f'{category_filter}_present'
            display_df = display_df[display_df[filter_column] == True]
        
        # Limit rows
        display_df = display_df.head(max_rows)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download options
        st.subheader("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Full results CSV
            csv_buffer = StringIO()
            processed_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📄 Download Full Results (CSV)",
                data=csv_data,
                file_name="classified_data.csv",
                mime="text/csv"
            )
        
        with col2:
            # Summary CSV
            summary_csv = StringIO()
            summary_df.to_csv(summary_csv, index=False)
            summary_data = summary_csv.getvalue()
            
            st.download_button(
                label="📊 Download Summary (CSV)",
                data=summary_data,
                file_name="classification_summary.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
