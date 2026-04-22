import os
import pandas as pd

from preprocess import clean_text
from matcher import ResumeMatcher
from ner import extract_entities

def main():
    print("Starting Resume Screening Pipeline...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    resumes_path = os.path.join(data_dir, 'resumes.csv')
    jobs_path = os.path.join(data_dir, 'jobs.csv')
    
    if not os.path.exists(resumes_path) or not os.path.exists(jobs_path):
        print("Data files not found. Please run 'python data/generate_mock_data.py' first.")
        return
        
    print("Loading datasets...")
    resumes_df = pd.read_csv(resumes_path)
    jobs_df = pd.read_csv(jobs_path)
    
    print("Preprocessing texts...")
    resumes_df['clean_text'] = resumes_df['text'].apply(clean_text)
    jobs_df['clean_description'] = jobs_df['description'].apply(clean_text)
    
    print("Extracting entities (skills) from resumes...")
    resumes_df['extracted_skills'] = resumes_df['text'].apply(lambda t: extract_entities(t)['skills'])
    
    matcher = ResumeMatcher()
    
    # Evaluate against the first job description as an example
    target_job = jobs_df.iloc[0]
    print(f"\n--- Target Job: {target_job['title']} ---")
    print(f"Description: {target_job['description']}\n")
    
    print("Ranking resumes...")
    ranked_results = matcher.rank_resumes(
        job_desc=target_job['clean_description'],
        resumes=resumes_df['clean_text'].tolist(),
        resume_ids=resumes_df['id'].tolist()
    )
    
    print("\n--- Top Ranked Resumes ---")
    for rank, result in enumerate(ranked_results, 1):
        resume_id = result['id']
        # Find original text and extracted skills for display
        original_data = resumes_df[resumes_df['id'] == resume_id].iloc[0]
        
        print(f"Rank {rank}: Resume ID {resume_id} (Score: {result['score']:.4f})")
        print(f"  Skills: {', '.join(original_data['extracted_skills'])}")
        print(f"  Snippet: {original_data['text'][:100]}...\n")
        
    print("Pipeline complete!")

if __name__ == "__main__":
    main()
