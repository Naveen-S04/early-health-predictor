import pandas as pd
import pickle
import os

# Load the dataset
df = pd.read_csv("dataset/Training.csv")  # Make sure this file exists in the same directory

# Extract all symptom column names (excluding last column)
symptom_list = df.columns[:-1].tolist()

# Ensure 'model' folder exists
os.makedirs("model", exist_ok=True)

# Save as pickle file
with open("model/symptom_list.pkl", "wb") as f:
    pickle.dump(symptom_list, f)

print("✅ symptom_list.pkl has been saved correctly.")
 