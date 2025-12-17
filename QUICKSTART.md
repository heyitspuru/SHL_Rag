# SHL RAG System - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
# Activate virtual environment (already created)
.venv\Scripts\activate

# Install required packages (if not already installed)
pip install -r requirements.txt
```

### Step 2: Build Vector Store

```bash
# This creates embeddings and builds the ChromaDB database
python src\build_vectorstore.py
```

Expected output:
```
✅ Loaded 90 assessments
✅ Created embeddings for 90 documents
✅ Vector store created and persisted
```

### Step 3: Test the System

**Option A: API Server**
```bash
# Start the FastAPI server
uvicorn src.api.main:app --reload --port 8000
```

Visit: http://localhost:8000/docs

Test with:
```bash
curl -X POST "http://localhost:8000/recommend" -H "Content-Type: application/json" -d "{\"query\": \"Java developer\", \"top_k\": 5}"
```

**Option B: Web Interface**
```bash
# Start Streamlit app
streamlit run src\frontend\app.py
```

Visit: http://localhost:8501

**Option C: Generate Test Predictions**
```bash
# Generate predictions for the test set
python src\generate_predictions.py
```

Output saved to: `data/predictions/test_predictions.csv`

---

## 📁 Important Files

| File | Purpose |
|------|---------|
| `data/raw/shl_assessments.csv` | Assessment catalog (90 items) |
| `data/raw/train.csv` | Training data (65 queries) |
| `data/raw/test.csv` | Test data (9 queries) |
| `data/predictions/test_predictions.csv` | Generated predictions |
| `src/build_vectorstore.py` | Build embeddings database |
| `src/generate_predictions.py` | Create test predictions |

---

## 🔧 Common Commands

```bash
# Rebuild vector store
python src\build_vectorstore.py

# Run API
uvicorn src.api.main:app --reload

# Run Frontend
streamlit run src\frontend\app.py

# Generate predictions
python src\generate_predictions.py

# Evaluate on training set
python src\evaluation\metrics.py
```

---

## 🐛 Troubleshooting

**Vector store not found?**
```bash
python src\build_vectorstore.py
```

**Import errors?**
```bash
# Make sure you're in the right directory
cd c:\Users\bhara\Desktop\shl_RAG

# Activate virtual environment
.venv\Scripts\activate
```

**Port already in use?**
```bash
# Use a different port
uvicorn src.api.main:app --port 8080
```

---

## 📊 Test Queries

Try these sample queries:

1. "I am hiring for Java developers who can also collaborate well in teams"
2. "Looking for entry-level sales representatives with customer service skills"
3. "Need data analysts with strong numerical and analytical abilities"
4. "Hiring senior manager with leadership and decision-making experience"
5. "Python developer with machine learning expertise"

---

## 🎯 Next Steps

1. ✅ Vector store built
2. ✅ Test API locally
3. ✅ Test frontend locally
4. ✅ Generate test predictions
5. ⏭️ Review `approach_document.md`
6. ⏭️ Deploy to cloud (optional)

---

## 📞 Support

- See [README.md](README.md) for full documentation
- See [approach_document.md](approach_document.md) for technical details
- Check logs in terminal for errors

**Happy recommending! 🎯**
