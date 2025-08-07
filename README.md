# 🧠 Early Health Predictor - GenAI + RAG + FAISS

This project is an end-to-end **Early Health Prediction** web app built using:

- ✅ **Machine Learning (Random Forest)** for disease classification
- ✅ **OpenAI Embeddings + FAISS** for RAG-based retrieval of medical data
- ✅ **Flask Web App** interface

---

## 🚀 Features
- Symptom-based disease prediction
- AI-generated medical explanation (OpenAI GPT-4o)
- Vector search over disease descriptions & precautions
- RAG (Retrieval-Augmented Generation) with FAISS

---

## 🗂️ Project Structure
```
early-health-predictor/
├── app.py                 # Flask backend
├── model/
│   └── train_model.py     # ML model training
├── vector_db/
│   ├── build_faiss.py     # Build FAISS vector DB
│   ├── faiss.index        # FAISS index (generated)
│   └── disease_map.pkl    # Index-to-disease mapping
├── templates/
│   └── index.html         # Web interface
├── static/
│   └── style.css          # Styling
├── dataset/               # Uploaded .csv files
├── requirements.txt
└── README.md
```

---

## 🛠️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Naveen-S04/early-health-predictor.git
cd early-health-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your OpenAI API key
Create a `.env` file:
```
OPENAI_API_KEY=your-key-here
```

### 4. Run FAISS vector index builder
```bash
python vector_db/build_faiss.py
```

### 5. Train the ML model
```bash
python model/train_model.py
```

### 6. Start the Flask app
```bash
python app.py
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) to use the app.

---

## 📊 Dataset
Ensure the following files are available in the root or `/dataset` folder:
- `Training.csv`
- `Testing.csv`
- `symptom_severity.csv`
- `disease_description.csv`
- `disease_precaution.csv`

---

## 📌 Credits
Based on [Ernest Bonat's article](https://ernest-bonat.medium.com/from-data-to-diagnostics-advanced-genai-solutions-for-disease-prediction-5aefa726f499)

---

## 🧪 License
MIT
