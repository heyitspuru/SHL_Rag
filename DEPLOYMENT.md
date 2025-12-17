# SHL RAG System - Deployment Guide

## 🚀 Deployment Options

### Option 1: Docker Deployment (Recommended)

#### Prerequisites
- Docker and Docker Compose installed
- 4GB+ RAM available

#### Steps
```bash
# Build and run all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Access services
- API: http://localhost:8000
- Frontend: http://localhost:8501
- API Docs: http://localhost:8000/docs

# Stop services
docker-compose down
```

---

### Option 2: Render.com Deployment

#### Prerequisites
- Render account (free tier available)
- GitHub repository

#### Steps

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy via Render**
   - Go to [render.com/dashboard](https://dashboard.render.com/)
   - Click "New" → "Blueprint"
   - Connect your GitHub repository
   - Render will auto-detect `render.yaml`
   - Click "Apply" to deploy both services

3. **Access Your Application**
   - Backend API: `https://shl-rag-api.onrender.com`
   - Frontend: `https://shl-rag-frontend.onrender.com`

#### Environment Variables (Optional)
Set in Render dashboard:
- `LOG_LEVEL`: info, debug, warning
- `MAX_RESULTS`: Maximum recommendations per query

---

### Option 3: Streamlit Cloud (Frontend Only)

#### Prerequisites
- Streamlit Cloud account (free)
- GitHub repository

#### Steps

1. **Push to GitHub** (same as above)

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io/)
   - Click "New app"
   - Select repository, branch, and file: `src/frontend/app.py`
   - Click "Deploy"

3. **Note**: You'll need to deploy the API separately (Render, Railway, etc.)

---

### Option 4: Heroku Deployment

#### Prerequisites
- Heroku account
- Heroku CLI installed

#### Steps

1. **Login and Create App**
   ```bash
   heroku login
   heroku create shl-rag-api
   ```

2. **Deploy API**
   ```bash
   git push heroku main
   ```

3. **Scale Dyno**
   ```bash
   heroku ps:scale web=1
   ```

4. **View Logs**
   ```bash
   heroku logs --tail
   ```

---

### Option 5: AWS EC2 Deployment

#### Prerequisites
- AWS account
- EC2 instance (t3.medium recommended)
- Security group allowing ports 8000, 8501

#### Steps

1. **SSH into EC2**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

2. **Install Docker**
   ```bash
   sudo apt update
   sudo apt install docker.io docker-compose -y
   sudo usermod -aG docker $USER
   ```

3. **Clone Repository**
   ```bash
   git clone <your-repo-url>
   cd shl_RAG
   ```

4. **Deploy**
   ```bash
   docker-compose up -d
   ```

5. **Access**
   - API: `http://your-ec2-ip:8000`
   - Frontend: `http://your-ec2-ip:8501`

---

## 🔧 Production Configuration

### Environment Variables

Create `.env` file (not committed to Git):
```bash
# Application
ENV=production
LOG_LEVEL=info

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
MAX_WORKERS=4

# RAG Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTORSTORE_PATH=data/processed/vectorstore
TOP_K_DEFAULT=10
MMR_LAMBDA=0.7

# Frontend
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Performance Tuning

#### API Server
- **Workers**: Set based on CPU cores
  ```bash
  uvicorn src.api.main:app --workers 4
  ```

- **Gunicorn (production)**:
  ```bash
  gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
  ```

#### Memory Optimization
- Limit ChromaDB cache size
- Use lighter embedding models if needed
- Implement request rate limiting

---

## 📊 Monitoring

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Get system stats
curl http://localhost:8000/stats
```

### Logging

Logs are saved to `logs/` directory:
- `api.log`: API access logs
- `rag.log`: RAG pipeline logs
- `error.log`: Error traces

### Metrics to Monitor
- Request latency (target: <500ms)
- Memory usage (vector store + models)
- Cache hit rate
- Error rate

---

## 🔒 Security Best Practices

1. **API Security**
   - Add API key authentication
   - Implement rate limiting
   - Use HTTPS in production

2. **Data Security**
   - Don't commit `.env` files
   - Encrypt sensitive data
   - Regular backups of vector store

3. **Network Security**
   - Use firewall rules
   - Enable CORS properly
   - Keep dependencies updated

---

## 🐛 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port
lsof -ti:8000 | xargs kill -9
```

#### Out of Memory
- Reduce batch size in predictions
- Use smaller embedding model
- Increase instance RAM

#### Vector Store Not Found
```bash
# Rebuild vector store
python src/build_vectorstore.py
```

#### Slow API Response
- Check embedding model size
- Optimize retrieval parameters (k, fetch_k)
- Add caching layer (Redis)

---

## 📈 Scaling Strategies

### Vertical Scaling
- Upgrade instance size (more RAM/CPU)
- Use GPU for embeddings (optional)

### Horizontal Scaling
- Load balancer (nginx, HAProxy)
- Multiple API replicas
- Shared vector store (network storage)

### Caching
```python
# Add Redis for query caching
from redis import Redis
cache = Redis(host='localhost', port=6379)
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions Example

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Render
        run: |
          curl -X POST ${{ secrets.RENDER_DEPLOY_HOOK }}
```

---

## 📞 Support

For deployment issues:
1. Check logs: `docker-compose logs`
2. Verify environment variables
3. Test locally first
4. Review [README.md](README.md) for system requirements

---

## 🎯 Production Checklist

- [ ] Environment variables configured
- [ ] Vector store built and tested
- [ ] API health check passes
- [ ] Frontend loads correctly
- [ ] SSL certificate installed (production)
- [ ] Monitoring configured
- [ ] Backups scheduled
- [ ] Documentation updated
- [ ] Load testing completed
- [ ] Security audit passed
