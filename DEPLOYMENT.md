# Deployment Guide

This document provides instructions for deploying the Legal Sanctions RAG application in various environments.

## Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- Access to environment variables (API keys, etc.)

## Quick Start with Docker

1. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. Build and run with Docker Compose:
   ```bash
   docker-compose up -d
   ```

3. The application will be available at http://localhost:8080

## Production Deployment

### 1. Docker Deployment

1. Build the optimized Docker image:
   ```bash
   docker build -t legal-sanctions-rag .
   ```

2. Run the container:
   ```bash
   docker run -d \
     --name legal-sanctions-rag \
     -p 8080:8080 \
     --env-file .env \
     -v $(pwd)/data:/app/data \
     legal-sanctions-rag
   ```

### 2. Cloud Deployment

#### Google Cloud Run

1. Build and push the image:
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/legal-sanctions-rag
   ```

2. Deploy to Cloud Run:
   ```bash
   gcloud run deploy legal-sanctions-rag \
     --image gcr.io/YOUR_PROJECT_ID/legal-sanctions-rag \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars="$(cat .env | tr '\n' ',')"
   ```

#### AWS Elastic Container Service (ECS)

1. Create an ECR repository:
   ```bash
   aws ecr create-repository --repository-name legal-sanctions-rag
   ```

2. Build and push the image:
   ```bash
   aws ecr get-login-password | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com
   docker build -t legal-sanctions-rag .
   docker tag legal-sanctions-rag:latest YOUR_ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/legal-sanctions-rag:latest
   docker push YOUR_ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/legal-sanctions-rag:latest
   ```

3. Create an ECS task definition and service using the AWS Console or CLI.

## Performance Optimization

1. **Memory Management**:
   - Set appropriate worker count based on available CPU cores
   - Monitor memory usage and adjust worker count accordingly

2. **Database Optimization**:
   - Use persistent storage for ChromaDB
   - Consider using a managed vector database service for production

3. **Caching**:
   - Enable Redis caching for frequently accessed data
   - Use browser caching for static assets

4. **Load Balancing**:
   - Use a load balancer for high availability
   - Configure health checks and auto-scaling

## Monitoring and Maintenance

1. **Logging**:
   - Monitor application logs
   - Set up log aggregation (e.g., CloudWatch, Stackdriver)

2. **Health Checks**:
   - Implement health check endpoints
   - Set up monitoring alerts

3. **Backup**:
   - Regular backups of ChromaDB data
   - Document encryption key backup

## Security Considerations

1. **API Keys**:
   - Store API keys in a secure vault
   - Rotate keys regularly

2. **Network Security**:
   - Use HTTPS
   - Configure firewall rules
   - Implement rate limiting

3. **Data Security**:
   - Encrypt sensitive data
   - Implement proper access controls
   - Regular security audits