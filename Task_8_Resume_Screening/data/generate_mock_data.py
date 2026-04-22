import os
import pandas as pd

def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Mock Resumes
    resumes = [
        {"id": "R1", "text": "Experienced Data Scientist with 5 years in machine learning, Python, pandas, and scikit-learn. Familiar with deep learning and NLP."},
        {"id": "R2", "text": "Frontend Developer with 3 years experience. Skilled in HTML, CSS, JavaScript, React, and building responsive web apps."},
        {"id": "R3", "text": "Recent graduate looking for a Junior Data Analyst role. Know SQL, Python, and data visualization using Matplotlib and Seaborn."},
        {"id": "R4", "text": "Senior Backend Engineer. Expertise in Node.js, Python, PostgreSQL, REST APIs, and AWS infrastructure."},
        {"id": "R5", "text": "NLP Engineer with experience in HuggingFace Transformers, PyTorch, spaCy, and building chatbot applications."}
    ]
    resumes_df = pd.DataFrame(resumes)
    resumes_df.to_csv(os.path.join(data_dir, "resumes.csv"), index=False)
    
    # Mock Job Descriptions
    jobs = [
        {"job_id": "J1", "title": "Machine Learning Engineer", "description": "Looking for an ML engineer with strong Python skills, experience with scikit-learn, and NLP or deep learning knowledge."},
        {"job_id": "J2", "title": "Web Developer", "description": "Seeking a frontend specialist proficient in React, HTML, CSS, and modern web development."},
        {"job_id": "J3", "title": "Data Analyst", "description": "We need someone to analyze data using SQL and Python. Experience with visualization libraries is a big plus."}
    ]
    jobs_df = pd.DataFrame(jobs)
    jobs_df.to_csv(os.path.join(data_dir, "jobs.csv"), index=False)
    
    print("Mock data generated successfully in", data_dir)

if __name__ == "__main__":
    main()
