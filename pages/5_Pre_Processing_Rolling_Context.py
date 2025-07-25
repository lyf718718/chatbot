import streamlit as st
import pandas as pd
import io

def find_best_match(columns, keywords):
    for keyword in keywords:
        for col in columns:
            if keyword.lower() in col.lower():
                return col
    return columns[0] if columns else None

def get_default_columns(df):
    columns = df.columns.tolist()
    
    id_keywords = ['id', 'chatid', 'call_id', 'callid', 'conversation_id', 'conversationid']
    turn_keywords = ['turn', 'step', 'sequence', 'order', 'number']
    speaker_keywords = ['speaker', 'role', 'person', 'user', 'participant']
    transcript_keywords = ['transcript', 'text', 'message', 'statement', 'content', 'utterance']
    
    return {
        'id': find_best_match(columns, id_keywords),
        'turn': find_best_match(columns, turn_keywords),
        'speaker': find_best_match(columns, speaker_keywords),
        'transcript': find_best_match(columns, transcript_keywords)
    }

def create_rolling_context(df, id_col, turn_col, speaker_col, transcript_col):
    result = []
    
    for call_id in df[id_col].unique():
        call_data = df[df[id_col] == call_id].sort_values(turn_col)
        
        for i, row in call_data.iterrows():
            if row[speaker_col] == 'salesperson':
                context_lines = []
                for j, prev_row in call_data[call_data[turn_col] < row[turn_col]].iterrows():
                    context_lines.append(f"Turn {prev_row[turn_col]} ({prev_row[speaker_col]}): {prev_row[transcript_col]}")
                
                context_lines.append(f"Turn {row[turn_col]} ({row[speaker_col]}): {row[transcript_col]}")
                context = '\n '.join(context_lines)
                
                result.append({
                    'ID': call_id,
                    'Turn': row[turn_col],
                    'Speaker': row[speaker_col],
                    'Context': context,
                    'Statement': row[transcript_col]
                })
    
    return pd.DataFrame(result)

st.title("Rolling Context Window Processor")
st.write("Upload your conversation dataset and process it with rolling context windows")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("### Data Preview")
    st.dataframe(df.head())
    
    st.write("### Column Mapping")
    st.write("Map your columns to the required fields (auto-detected defaults shown):")
    
    defaults = get_default_columns(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        id_col = st.selectbox("ID Column (Call/Conversation ID)", df.columns, 
                             index=df.columns.tolist().index(defaults['id']) if defaults['id'] else 0)
        turn_col = st.selectbox("Turn Column", df.columns,
                               index=df.columns.tolist().index(defaults['turn']) if defaults['turn'] else 0)
    
    with col2:
        speaker_col = st.selectbox("Speaker Column", df.columns,
                                  index=df.columns.tolist().index(defaults['speaker']) if defaults['speaker'] else 0)
        transcript_col = st.selectbox("Statement/Transcript Column", df.columns,
                                     index=df.columns.tolist().index(defaults['transcript']) if defaults['transcript'] else 0)
    
    if st.button("Process Data"):
        with st.spinner("Processing..."):
            processed_df = create_rolling_context(df, id_col, turn_col, speaker_col, transcript_col)
            
            st.success(f"Processed {len(processed_df)} salesperson statements")
            
            st.write("### Processed Data Preview")
            st.dataframe(processed_df.head())
            
            csv_buffer = io.StringIO()
            processed_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="Download Processed CSV",
                data=csv_data,
                file_name="processed_rolling_window.csv",
                mime="text/csv"
            )
