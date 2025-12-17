# SHL Assessment Recommendation System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Intelligent RAG-based system for recommending relevant SHL assessments based on job requirements using Retrieval-Augmented Generation, semantic search, and intelligent test type balancing.

## 🚀 Quick Links

- **Live Demo**: [Streamlit App](https://your-app.streamlit.app) _(Deploy after setup)_
- **API Endpoint**: [FastAPI Docs](http://localhost:8000/docs) _(Run locally)_
- **Documentation**: [Approach Document](approach_document.md)

## 📋 Project Overview

This system uses Retrieval-Augmented Generation (RAG) to recommend SHL Individual Test Solutions from a catalog of 90+ assessments. It intelligently analyzes job descriptions and requirements to provide balanced recommendations across different test types.

### Key Features

- ✅ **Semantic Search**: Uses sentence transformers for intelligent matching
- ✅ **Intelligent Balancing**: Distributes recommendations across test types (K, C, P, S)
- ✅ **MMR Retrieval**: Maximum Marginal Relevance for diverse results
- ✅ **REST API**: FastAPI backend with health check and recommendation endpoints
- ✅ **Web Interface**: Interactive Streamlit frontend
- ✅ **Batch Processing**: Handle multiple queries efficiently

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Vector DB** | ChromaDB |
| **Framework** | LangChain |
| **LLM** | Google Gemini 1.5 Flash (optional) |
| **API** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Data Processing** | pandas, numpy |

## 📂 Project Structure

```
shl_RAG/
├── data/
│   ├── raw/                          # Raw data files
│   │   ├── shl_assessments.csv       # Scraped catalog (90 assessments)
│   │   ├── shl_assessments.json      # JSON format
│   │   ├── train.csv                 # Training data (65 queries)
│   │   └── test.csv                  # Test data (9 queries)
│   ├── processed/
│   │   └── vectorstore/              # ChromaDB vector store
│   └── predictions/
│       └── test_predictions.csv      # Generated predictions
├── src/
│   ├── scraper/
│   │   └── shl_scraper.py           # Web scraper for SHL catalog
│   ├── rag/
│   │   ├── embeddings.py            # Embedding generation
│   │   ├── vectorstore.py           # Vector store management
│   │   ├── retriever.py             # Retrieval strategies
│   │   └── recommender.py           # Main recommendation engine
│   ├── evaluation/
│   │   └── metrics.py               # Recall@K and evaluation
│   ├── api/
│   │   ├── main.py                  # FastAPI application
│   │   └── models.py                # Pydantic models
│   ├── frontend/
│   │   └── app.py                   # Streamlit interface
│   ├── build_vectorstore.py         # Build vector DB
│   └── generate_predictions.py      # Generate test predictions
├── config/
│   └── config.yaml                  # Configuration
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variables template
└── README.md                        # This file
```

## 🚀 Setup & Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Google Gemini API key for LLM features

### Installation Steps

```bash
# 1. Clone or download the repository
cd shl_RAG

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables (optional)
copy .env.example .env
# Edit .env and add your GOOGLE_API_KEY if using LLM features
```

### Build Vector Store

```bash
# Generate embeddings and build ChromaDB vector store
python src/build_vectorstore.py
```

This will:
- Load the 90 assessments from `data/raw/shl_assessments.csv`
- Generate embeddings using sentence-transformers
- Create and persist ChromaDB vector store
- Takes ~2-3 minutes on first run

## 💻 Usage

### 1. Run the API Server

```bash
# Start FastAPI server
uvicorn src.api.main:app --reload --port 8000
```

Visit http://localhost:8000/docs for interactive API documentation.

**API Endpoints:**

- `GET /health` - Health check
- `POST /recommend` - Get recommendations
- `GET /stats` - Catalog statistics

**Example API Request:**

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Java developer with collaboration skills",
    "top_k": 10
  }'
```

### 2. Run the Web Interface

```bash
# Start Streamlit app
streamlit run src/frontend/app.py
```

Visit http://localhost:8501 to access the web interface.

**Features:**
- **Single Query**: Enter job description and get recommendations
- **Batch Process**: Upload CSV with multiple queries
- **Catalog Browser**: Explore all assessments with filters

### 3. Generate Test Predictions

```bash
# Generate predictions for test set
python src/generate_predictions.py --test-file data/raw/test.csv --output data/predictions/test_predictions.csv --top-k 10
```

**Options:**
- `--test-file`: Path to test CSV
- `--output`: Output path for predictions
- `--top-k`: Number of recommendations per query (default: 10)
- `--no-balancing`: Disable test type balancing

## 📊 System Architecture

### RAG Pipeline

```
User Query
    ↓
[Query Analysis]
    ↓
[Semantic Search] → Vector Store (ChromaDB)
    ↓
[MMR Retrieval] → Diverse candidate set
    ↓
[LLM Re-ranking] (optional)
    ↓
[Test Type Balancing]
    ↓
Recommendations (K, C, P, S types)
```

### Test Type Classification

- **K (Knowledge)**: Technical skills, programming, domain expertise
- **C (Cognitive)**: Numerical reasoning, verbal reasoning, logical aptitude
- **P (Personality)**: Behavioral traits, work style, motivations
- **S (Situational)**: Judgment, decision-making, scenario-based

### Intelligent Balancing Algorithm

The system analyzes queries to determine the optimal mix of test types:

```python
# Example: "Java developer with collaboration skills"
Requirements: {
    'K': 0.50,  # Technical skills (Java)
    'C': 0.20,  # Problem-solving
    'P': 0.30,  # Collaboration/teamwork
    'S': 0.00   # Not required
}

# Recommendations balanced to match these proportions
```

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Recall@10** | 0.72 (estimated) |
| **Recall@5** | 0.58 (estimated) |
| **Avg Response Time** | ~1.5s |
| **Catalog Size** | 90 assessments |
| **Embedding Dimension** | 384 |

### Evaluation

```bash
# Evaluate on training set
python -c "
from src.evaluation.metrics import evaluate_model
from src.rag.recommender import SHLRecommendationEngine
import pandas as pd

# Load data and generate predictions
# ... (see src/evaluation/metrics.py for full example)

results = evaluate_model(predictions_df, train_df, catalog_df, k_values=[5, 10])
print(results)
"
```

## 🔧 Configuration

Edit `config/config.yaml` to customize:

```yaml
# Embedding configuration
embeddings:
  model_name: "all-MiniLM-L6-v2"  # or "all-mpnet-base-v2" for better quality

# Retrieval configuration
retrieval:
  base_k: 40
  top_k: 20
  final_k: 10
  mmr_lambda: 0.7  # 0=diversity, 1=relevance

# Recommendation configuration
recommendation:
  enable_balancing: true
  enable_reranking: false  # Set true if using LLM
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Test specific component
pytest tests/test_retriever.py -v
```

## 📝 Development Workflow

### Adding New Assessments

1. Update `data/raw/shl_assessments.csv`
2. Rebuild vector store:
   ```bash
   python src/build_vectorstore.py
   ```

### Improving Retrieval Quality

1. **Try different embedding models** in `config/config.yaml`
2. **Adjust MMR lambda** for relevance vs. diversity trade-off
3. **Tune balancing logic** in `src/rag/recommender.py`
4. **Enable LLM re-ranking** for better relevance (requires API key)

### Dataset Format

**Training Data (train.csv):**
```csv
Query,Assessment_url
"I am hiring for Java developers...",https://www.shl.com/...
```

**Test Data (test.csv):**
```csv
Query
"Looking to hire mid-level professionals..."
```

**Predictions Output (test_predictions.csv):**
```csv
Query,Assessment_url
"Looking to hire mid-level professionals...",https://www.shl.com/assessment1
"Looking to hire mid-level professionals...",https://www.shl.com/assessment2
...
```

## 🐛 Troubleshooting

### Vector Store Not Found

```bash
# Rebuild the vector store
python src/build_vectorstore.py
```

### Import Errors

```bash
# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows
```

### API Not Starting

```bash
# Check if port 8000 is available
# Try a different port
uvicorn src.api.main:app --port 8080
```

### Streamlit Not Loading

```bash
# Clear cache and restart
streamlit cache clear
streamlit run src/frontend/app.py
```

## 🚢 Deployment

### Deploy API (Render.com)

1. Create `render.yaml`:
```yaml
services:
  - type: web
    name: shl-recommender-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
```

2. Connect GitHub repo to Render
3. Deploy

### Deploy Frontend (Streamlit Cloud)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy `src/frontend/app.py`

## 📚 Key Files Explained

| File | Purpose |
|------|---------|
| `src/rag/recommender.py` | Core recommendation engine with balancing logic |
| `src/rag/embeddings.py` | Document formatting and embedding generation |
| `src/rag/vectorstore.py` | ChromaDB management and persistence |
| `src/rag/retriever.py` | MMR retrieval and query enhancement |
| `src/evaluation/metrics.py` | Recall@K calculation and evaluation |
| `src/api/main.py` | FastAPI REST API |
| `src/frontend/app.py` | Streamlit web interface |
| `src/generate_predictions.py` | Batch prediction for test set |

## 🎯 Future Improvements

1. **Fine-tuned Embeddings**: Train domain-specific embeddings on SHL data
2. **User Feedback Loop**: Collect clicks/selections to improve ranking
3. **Query Expansion**: Use assessment taxonomy for better matching
4. **Caching**: Cache common queries for faster response
5. **A/B Testing**: Compare different retrieval strategies
6. **Multi-language Support**: Support queries in multiple languages
7. **Real-time Scraping**: Keep catalog updated automatically

## 📄 License

This project was created as part of SHL's hiring assessment.

## 👤 Author

Your Name - [email@example.com](mailto:email@example.com)

## 🙏 Acknowledgments

- SHL for the opportunity and dataset
- LangChain team for the excellent RAG framework
- ChromaDB for the vector database
- sentence-transformers for embeddings

## 📞 Support

For issues or questions:
1. Check this README
2. Review [approach_document.md](approach_document.md)
3. Check logs in terminal output
4. Open an issue on GitHub

---

**Built with ❤️ using LangChain, ChromaDB, and Streamlit**
