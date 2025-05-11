from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

df = pd.read_csv("Home Remedies (1).csv")

health_issues = df['Health Issue'].astype(str).tolist()

embedder = SentenceTransformer('sentence-transformers/gtr-t5-base')


embeddings = embedder.encode(health_issues, convert_to_numpy=True)
np.save("C:/Users/manas/OneDrive/Desktop/New folder/health_issues_embeddings.npy", embeddings)