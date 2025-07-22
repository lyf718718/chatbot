import streamlit as st
import pandas as pd
import re
from collections import defaultdict

def create_classification_dictionary():
    """Create predefined keyword dictionaries for classification"""
    return {
        "product_quality": ["excellent", "perfect", "classic", "gorgeous", "quality", "premium", "superior", "outstanding"],
        "urgency": ["last week", "limited time", "hurry", "deadline", "expires", "final", "closing", "urgent"],
        "personal_connection": ["smile", "goal", "personal", "relationship", "connect", "touch", "care", "understand"],
        "travel_business": ["travel", "suit", "business", "professional", "work", "office", "corporate", "formal"],
        "seasonal_holiday": ["holidays", "christmas", "season", "winter", "summer", "spring", "fall", "celebration"]
    }

def classify_text(text, dictionary):
    """Classify text based on keyword dictionary"""
    if not text or pd.isna(text):
        return {}
    
    text_lower = text.lower()
    results = {}
    
    for category, keywords in dictionary.items():
        matches = []
        for keyword in keywords:
            if keyword.lower() in text_lower:
                matches.append(keyword)
        
        if matches:
            results[category] = {
                'matched_keywords': matches,
                'count': len(matches)
            }
    
    return results

def classify_dataframe(df, text_column, dictionary):
    """Apply classification to entire dataframe"""
    classifications = []
    
    for idx, row in df.iterrows():
        text = row[text_column]
        classification = classify_text(text, dictionary)
        classifications.append(classification)
    
    return classifications

st.title("Dictionary-Based Text Classification")
st.write("Upload your CSV file and classify text using keyword dictionaries")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Column selection
    text_column = st.selectbox("Select text column to classify:", df.columns)
    
    # Classification dictionary
    st.subheader("Classification Dictionary")
    classification_dict = create_classification_dictionary()
    
    # Display dictionary
    for category, keywords in classification_dict.items():
        st.write(f"**{category.replace('_', ' ').title()}:** {', '.join(keywords)}")
    
    # Classify button
    if st.button("Classify Text"):
        with st.spinner("Classifying..."):
            classifications = classify_dataframe(df, text_column, classification_dict)
            
            # Add classifications to dataframe
            df['classifications'] = classifications
            
            # Create summary columns
            for category in classification_dict.keys():
                df[f'{category}_match'] = df['classifications'].apply(
                    lambda x: x.get(category, {}).get('count', 0)
                )
            
            # Display results
            st.subheader("Classification Results")
            
            # Summary statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Classification Summary:**")
                total_texts = len(df)
                for category in classification_dict.keys():
                    matches = sum(1 for c in classifications if category in c)
                    percentage = (matches / total_texts) * 100
                    st.write(f"{category.replace('_', ' ').title()}: {matches}/{total_texts} ({percentage:.1f}%)")
            
            with col2:
                st.write("**Most Common Categories:**")
                category_counts = defaultdict(int)
                for classification in classifications:
                    for category in classification.keys():
                        category_counts[category] += 1
                
                sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
                for category, count in sorted_categories[:5]:
                    st.write(f"{category.replace('_', ' ').title()}: {count}")
            
            # Detailed results table
            st.subheader("Detailed Results")
            
            # Create display dataframe
            display_df = df[[text_column] + [f'{cat}_match' for cat in classification_dict.keys()]].copy()
            
            # Add classification details
            def format_classification(row):
                details = []
                classification = df.loc[row.name, 'classifications']
                for category, info in classification.items():
                    keywords = ', '.join(info['matched_keywords'])
                    details.append(f"{category}: {keywords}")
                return '; '.join(details) if details else "No matches"
            
            display_df['matched_categories'] = display_df.apply(format_classification, axis=1)
            
            st.dataframe(display_df)
            
            # Download results
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download Classification Results",
                data=csv,
                file_name="classification_results.csv",
                mime="text/csv"
            )
else:
    # Load sample data if no file uploaded
    try:
        df = pd.read_csv("sample_data.csv")
        st.subheader("Sample Data Loaded")
        st.write("Using sample_data.csv for demonstration")
        st.dataframe(df.head())
        
        # Quick classification on sample data
        if st.button("Classify Sample Data"):
            classification_dict = create_classification_dictionary()
            classifications = classify_dataframe(df, "Statement", classification_dict)
            
            st.subheader("Sample Classification Results")
            for i, (idx, row) in enumerate(df.head().iterrows()):
                st.write(f"**Text:** {row['Statement']}")
                if classifications[i]:
                    for category, info in classifications[i].items():
                        st.write(f"- {category.replace('_', ' ').title()}: {', '.join(info['matched_keywords'])}")
                else:
                    st.write("- No matches found")
                st.write("---")
                
    except FileNotFoundError:
        st.info("Upload a CSV file to begin classification, or add sample_data.csv to the directory.")
