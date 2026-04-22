"""
streamlit_app.py
A simple interactive UI for Question Answering using Streamlit.
"""
import streamlit as st
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath("src"))
from qa_model import load_qa_pipeline, answer_question

# Set page config
st.set_page_config(page_title="QA System", page_icon="🤖", layout="centered")

st.title("🤖 Question Answering System")
st.markdown("Enter a context passage and ask a question to get an answer using an NLP transformer model.")

@st.cache_resource
def load_model():
    return load_qa_pipeline()

try:
    qa_pipeline = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input fields
context = st.text_area("Context / Passage", height=200, placeholder="Paste the text passage here...")
question = st.text_input("Question", placeholder="What do you want to know?")

if st.button("Get Answer"):
    if not context.strip():
        st.warning("Please enter a context passage.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Extracting answer..."):
            result = answer_question(qa_pipeline, context, question)
            
            st.markdown("### Answer")
            st.info(f"**{result['answer']}**")
            
            st.markdown(f"*Confidence Score: {round(result['score'] * 100, 2)}%*")
            
            # Highlight answer in context
            start = result['start']
            end = result['end']
            highlighted_context = (
                context[:start] + 
                f"<mark style='background-color: #ffeb3b; color: black; font-weight: bold;'>{context[start:end]}</mark>" + 
                context[end:]
            )
            
            st.markdown("### Highlighted Context")
            st.markdown(highlighted_context, unsafe_allow_html=True)
