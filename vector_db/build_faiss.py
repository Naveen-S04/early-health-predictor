import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Get current script's directory (absolute)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correct absolute paths
desc_path = os.path.join(BASE_DIR, "disease_description.csv")
index_path = os.path.join(BASE_DIR, "faiss.index")

# Load disease description file
df = pd.read_csv(desc_path)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Get embeddings
descriptions = df["Symptom_Description"].tolist()
embeddings = embedding_model.encode(descriptions)
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index at correct location
faiss.write_index(index, index_path)

print("✅ FAISS index saved at:", index_path)
