import os
import pandas as pd
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

print("Downloading dataset for Task 1: Sentiment Analysis...")
imdb = load_dataset('imdb')
t1_train = pd.DataFrame(imdb['train']).sample(5000, random_state=42).reset_index(drop=True)
t1_test = pd.DataFrame(imdb['test']).sample(2000, random_state=42).reset_index(drop=True)
t1_train.to_csv('Task_1_Sentiment_Analysis/data/train.csv', index=False)
t1_test.to_csv('Task_1_Sentiment_Analysis/data/test.csv', index=False)

print("Downloading dataset for Task 2: News Classification...")
ag_news = load_dataset('ag_news')
t2_train = pd.DataFrame(ag_news['train'])
t2_test = pd.DataFrame(ag_news['test'])
t2_train = t2_train.groupby('label', group_keys=False).sample(n=2000, random_state=42).reset_index(drop=True)
t2_test = t2_test.groupby('label', group_keys=False).sample(n=500, random_state=42).reset_index(drop=True)
t2_train.to_csv('Task_2_News_Classification/data/train.csv', index=False)
t2_test.to_csv('Task_2_News_Classification/data/test.csv', index=False)

print("Downloading dataset for Task 3: Fake News Detection...")
liar = load_dataset('liar', trust_remote_code=True)
t3_train_df = pd.DataFrame(liar['train'])
t3_val_df = pd.DataFrame(liar['validation'])
t3_train = pd.concat([t3_train_df, t3_val_df], ignore_index=True)
t3_test = pd.DataFrame(liar['test'])
t3_train.to_csv('Task_3_Fake_News/data/train.csv', index=False)
t3_test.to_csv('Task_3_Fake_News/data/test.csv', index=False)

print("Downloading dataset for Task 7: Text Summarization...")
cnn = load_dataset('cnn_dailymail', '3.0.0', split='test')
t7_sample = pd.DataFrame(cnn.select(range(50)))
t7_sample = t7_sample.rename(columns={'article': 'article', 'highlights': 'reference_summary'})
t7_sample.to_csv('Task_7_Text_Summarization/data/sample_articles.csv', index=False)

print("Downloading dataset for Task 4: Named Entity Recognition...")
conll003 = load_dataset('conll2003')

t4_train = pd.DataFrame(conll003['train'])
t4_test = pd.DataFrame(conll003['test'])
t4_train.to_csv('Task_4_NER/data/train.csv', index=False)
t4_test.to_csv('Task_4_NER/data/test.csv', index=False)

print("Downloading dataset for Task 6: Question Answering...")
squad = load_dataset('squad', split='validation')
t6_sample = pd.DataFrame(squad).sample(100, random_state=42).reset_index(drop=True)
import os
os.makedirs('Task_6_Question_Answering/data', exist_ok=True)
t6_sample.to_csv('Task_6_Question_Answering/data/sample_squad.csv', index=False)

print("Data successfully saved to all tasks!")
