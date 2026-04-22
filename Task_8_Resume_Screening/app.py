import streamlit as st
import sys
import os

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import process_file, clean_text
from ner import extract_entities
from matcher import ResumeMatcher

st.set_page_config(page_title="Resume Screener", page_icon="📄", layout="wide")

@st.cache_resource
def load_matcher():
    return ResumeMatcher()

def main():
    st.title("📄 Resume Screening System using NLP")
    st.markdown("Upload a candidate's resume and provide a job description to see the semantic matching score and extracted skills.")
    
    matcher = load_matcher()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Job Description")
        job_desc = st.text_area("Paste the job description here:", height=300)
        
    with col2:
        st.subheader("2. Candidate Resume")
        uploaded_file = st.file_uploader("Upload Resume (.txt, .pdf)", type=['txt', 'pdf'])
        
    if st.button("Screen Resume", type="primary"):
        if not job_desc.strip():
            st.error("Please provide a job description.")
            return
            
        if uploaded_file is None:
            st.error("Please upload a resume.")
            return
            
        with st.spinner("Processing..."):
            # Determine file type
            file_type = "pdf" if uploaded_file.name.endswith('.pdf') else "txt"
            
            # Extract and clean text from resume
            resume_text = process_file(uploaded_file, file_type)
            clean_job_desc = clean_text(job_desc)
            
            # Extract Entities (Skills)
            entities = extract_entities(resume_text)
            skills = entities.get("skills", [])
            experience = entities.get("experience", [])
            
            # Compute match score
            score = matcher.compute_similarity(clean_job_desc, [resume_text])[0]
            
            st.success("Screening Complete!")
            
            st.markdown("---")
            st.subheader("📊 Match Results")
            
            # Display Score
            st.metric(label="Match Score", value=f"{score * 100:.1f}%")
            
            # Determine match strength
            if score >= 0.75:
                st.info("🔥 Strong Match for this role!")
            elif score >= 0.5:
                st.warning("👍 Average Match. Worth reviewing.")
            else:
                st.error("❌ Weak Match.")
                
            st.markdown("### 💡 Extracted Details")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Identified Skills:**")
                if skills:
                    for skill in skills:
                        st.markdown(f"- {skill}")
                else:
                    st.write("No specific skills identified.")
                    
            with col_b:
                st.markdown("**Organizations / Dates found:**")
                if experience:
                    for exp in experience:
                        st.markdown(f"- {exp}")
                else:
                    st.write("No specific experience entities identified.")

if __name__ == "__main__":
    main()
