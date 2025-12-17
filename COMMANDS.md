# 🚀 SHL RAG - Quick Command Reference

## Essential Commands

### 🏃 Run Full Pipeline
```bash
python run_pipeline.py
```
Executes: Data verification → Build vector store → Generate predictions → Validate → System check

### 🧪 Test System
```bash
python test_system.py
```
Tests: Health endpoint → Recommendations → Statistics

### 🔧 Start Services

#### API Server
```bash
# Development
python -m uvicorn src.api.main:app --reload

# Production
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Frontend
```bash
streamlit run src/frontend/app.py
```

#### Both (Docker)
```bash
docker-compose up -d
```

---

## 📊 Key Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Get Recommendations
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Java programming assessment",
    "top_k": 10,
    "enable_balancing": true
  }'
```

### System Stats
```bash
curl http://localhost:8000/stats
```

### Interactive API Docs
```
http://localhost:8000/docs
```

---

## 📁 Important Files

| File | Purpose | Location |
|------|---------|----------|
| **Test Predictions** | Final submission | `data/predictions/test_predictions.csv` |
| **Vector Store** | ChromaDB | `data/processed/vectorstore/` |
| **Assessment Catalog** | 90 assessments | `data/raw/shl_assessments.csv` |
| **Train Data** | 65 queries | `data/raw/train.csv` |
| **Test Data** | 9 queries | `data/raw/test.csv` |

---

## 🐳 Docker Commands

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild
docker-compose up -d --build

# Check status
docker-compose ps
```

---

## 🔍 Troubleshooting

### Port Already in Use
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### Rebuild Vector Store
```bash
python src/build_vectorstore.py
```

### Regenerate Predictions
```bash
python src/generate_predictions.py --top-k 10
```

### Check Python Environment
```bash
# Verify virtual environment
.venv\Scripts\python --version

# List installed packages
pip list

# Install requirements
pip install -r requirements.txt
```

---

## 📦 Deployment Quick Start

### Render.com
```bash
git push origin main
# Then connect repository in Render dashboard
```

### Heroku
```bash
heroku login
heroku create shl-rag
git push heroku main
```

### AWS EC2
```bash
ssh -i key.pem ubuntu@<ec2-ip>
git clone <repo-url>
cd shl_RAG
docker-compose up -d
```

---

## 📈 Monitoring

### Check Logs
```bash
# API logs
tail -f logs/api.log

# Error logs
tail -f logs/error.log

# Docker logs
docker-compose logs -f shl-rag-api
```

### Resource Usage
```bash
# Docker stats
docker stats

# System resources
htop  # Linux
Get-Process | Sort-Object CPU -Descending | Select-Object -First 10  # Windows
```

---

## 🎯 Evaluation

### Run Metrics
```python
from evaluation.metrics import recall_at_k, mean_recall_at_k

# Calculate Recall@10
score = recall_at_k(predictions, ground_truth, k=10)
```

### Validate Predictions
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/predictions/test_predictions.csv')
print(f'Total: {len(df)} rows')
print(f'Unique queries: {df.Query.nunique()}')
print(f'Per query: {df.groupby(\"Query\").size().tolist()}')
"
```

---

## 🔑 Environment Variables

Create `.env` file:
```bash
ENV=production
LOG_LEVEL=info
API_PORT=8000
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K_DEFAULT=10
MMR_LAMBDA=0.7
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Main project documentation |
| [QUICKSTART.md](QUICKSTART.md) | Fast setup guide |
| [approach_document.md](approach_document.md) | Technical approach |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Deployment guide (5 platforms) |
| [PRODUCTION_STATUS.md](PRODUCTION_STATUS.md) | Production readiness report |
| [shl-project-guide.md](shl-project-guide.md) | Original requirements |

---

## 🆘 Quick Help

### API Not Responding
1. Check if running: `curl http://localhost:8000/health`
2. Check logs: `docker-compose logs shl-rag-api`
3. Restart: `docker-compose restart shl-rag-api`

### Vector Store Issues
1. Check existence: `ls data/processed/vectorstore/`
2. Rebuild: `python src/build_vectorstore.py`
3. Check permissions

### Import Errors
1. Activate venv: `.venv\Scripts\activate`
2. Reinstall: `pip install -r requirements.txt`
3. Check Python version: `python --version` (should be 3.11+)

---

## ✅ Verification Checklist

Before deployment:
- [ ] Pipeline runs successfully: `python run_pipeline.py`
- [ ] Tests pass: `python test_system.py`
- [ ] API health check: `curl localhost:8000/health`
- [ ] Predictions generated: Check `data/predictions/test_predictions.csv`
- [ ] Vector store exists: Check `data/processed/vectorstore/`
- [ ] Documentation complete: All .md files present
- [ ] Environment configured: `.env` file created
- [ ] Dependencies installed: `pip list` shows all packages

---

**Quick Access URLs**
- API: http://localhost:8000
- Frontend: http://localhost:8501
- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Stats: http://localhost:8000/stats

---

**Last Updated**: December 17, 2025  
**Status**: ✅ Production Ready
