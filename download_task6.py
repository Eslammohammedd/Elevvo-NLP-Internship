import pandas as pd
from datasets import load_dataset
import os

print("Downloading dataset for Task 6: Question Answering...")
squad = load_dataset('squad', split='validation')
t6_sample = pd.DataFrame(squad).sample(100, random_state=42).reset_index(drop=True)
os.makedirs('Task_6_Question_Answering/data', exist_ok=True)
t6_sample.to_csv('Task_6_Question_Answering/data/sample_squad.csv', index=False)
