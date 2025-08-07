import streamlit as st
import faiss
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Load your trained disease prediction model
with open("model/predictor.pkl", "rb") as f:
    predictor = pickle.load(f)

# Load symptom list
with open("model/symptom_list.pkl", "rb") as f:
    symptoms = pickle.load(f)

# Load FAISS index
index = faiss.read_index("vector_db/faiss.index")

# Load disease descriptions
df_desc = pd.read_csv("vector_db/disease_description.csv")

# Load sentence-transformer model (Hugging Face)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI
st.title("🧠 Early Health Predictor - GenAI")

# Symptom selection
st.markdown("### 🤒 Select your symptoms:")
selected_symptoms = st.multiselect("Choose from the list", symptoms)

if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        # Vectorize input symptoms for classifier
        input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]

        # Predict disease using classifier
        predicted_disease = predictor.predict([input_vector])[0]
        st.success(f"🩺 Predicted Disease: **{predicted_disease}**")

        # FAISS vector search for matching description
        query = " ".join(selected_symptoms)
        query_embedding = embedding_model.encode([query])
        distances, indices = index.search(np.array(query_embedding).astype("float32"), 1)

        top_index = indices[0][0]

        # Defensive check
        if top_index == -1 or top_index >= len(df_desc):
            st.warning("⚠️ No matching disease information found.")
        else:
            matched_info = df_desc.iloc[top_index]
            st.markdown("### 🧾 Disease Information")
            st.write(f"**Disease:** {matched_info['Disease']}")
            st.write(f"**Description:** {matched_info['Symptom_Description']}")
