import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os


csv_path = "dataset/Training.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ File not found: {csv_path}")
df = pd.read_csv(csv_path)
# Load dataset
df = pd.read_csv("dataset/Training.csv")  # ensure the path is correct

# Features are all columns except the last one
X = df.iloc[:, :-1]

# Target is the last column
y = df.iloc[:, -1]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained. Accuracy: {acc:.2f}")

# Save model and label encoder
os.makedirs("saved_model", exist_ok=True)
joblib.dump(model, "saved_model/naive_bayes_model.pkl")
joblib.dump(le, "saved_model/label_encoder.pkl")
