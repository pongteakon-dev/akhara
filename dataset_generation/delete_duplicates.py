import pandas as pd

df = pd.read_csv('dataset_generation/generated_nouns.csv')
df = df.drop_duplicates(subset=['noun'])
df.to_csv("dataset_generation/generated_nouns.csv", index=False)