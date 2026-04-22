import os
import json
import pandas as pd
from datetime import datetime
from matcher import ResumeMatcher
from preprocess import clean_text

def generate_results():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'data')
    os.makedirs(results_dir, exist_ok=True)
    
    resumes_path = os.path.join(data_dir, 'resumes.csv')
    jobs_path = os.path.join(data_dir, 'jobs.csv')
    
    if not os.path.exists(resumes_path) or not os.path.exists(jobs_path):
        print("Data files not found.")
        return

    print("Loading data...")
    resumes_df = pd.read_csv(resumes_path)
    jobs_df = pd.read_csv(jobs_path)
    
    resumes_df['clean_text'] = resumes_df['text'].apply(clean_text)
    jobs_df['clean_description'] = jobs_df['description'].apply(clean_text)
    
    matcher = ResumeMatcher()
    
    results_data = []
    
    for _, job in jobs_df.iterrows():
        job_id = job['job_id']
        job_desc = job['clean_description']
        
        ranked_resumes = matcher.rank_resumes(
            job_desc=job_desc,
            resumes=resumes_df['clean_text'].tolist(),
            resume_ids=resumes_df['id'].tolist()
        )
        
        for rank, r in enumerate(ranked_resumes, 1):
            results_data.append({
                "job_id": job_id,
                "resume_id": r['id'],
                "rank": rank,
                "score": r['score']
            })
            
    results_df = pd.DataFrame(results_data)
    csv_path = os.path.join(results_dir, 'screening_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Save a summary JSON
    summary = {
        "task": "Resume Screening",
        "timestamp": datetime.now().isoformat(),
        "total_jobs_evaluated": len(jobs_df),
        "total_resumes_evaluated": len(resumes_df),
        "model_used": "all-mpnet-base-v2",
        "average_top_1_score": results_df[results_df['rank'] == 1]['score'].mean()
    }
    
    json_path = os.path.join(results_dir, 'results_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to {json_path}")

if __name__ == "__main__":
    generate_results()
