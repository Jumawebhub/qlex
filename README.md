# QueryLex v2 – Secure Legal‑RAG Marketplace ⚖️🤖

A secure, EU-hosted marketplace where legal professionals can buy, sell, and chat with Retrieval‑Augmented‑Generation (RAG) models. Built with enterprise-grade security and compliance in mind.

## ✨ Core Features

### Marketplace & Creator Tools
- **Marketplace / Leaderboard** – Sort by usage and ratings (OpenRouter style)
- **Creator Wizard** – Simple upload ▶ tag ▶ price ▶ publish workflow
- **Immutable Versioning** – Document bundles are versioned with secure hashes
- **Revenue Sharing** – Creators earn 70% of revenue (30% platform fee)

### User Experience
- **Free Tier** – 3 free requests per month (rate-limited to 1 req/min)
- **Credit System** – Top-up credits (1 € = 1 000 tokens)
- **Real-time Streaming** – Chat responses stream tokens in real-time
- **Source Citations** – All responses include source chunk hashes
- **Organized Workspace** – Drag-and-drop folder organization with React-Arborist

### Security & Compliance
- **EU Data Locality** – All data stored in EU ChromaDB cluster
- **Document Encryption** – Fernet-encrypted documents and embeddings
- **PII Protection** – ≥ 95% recall/precision PII scrubber (Piiranha + regex)
- **Comprehensive Audit Logging** – Track all system activities
- **GDPR Compliance** – Built-in data protection features

## 🏗 Project Structure

```
backend/     # Flask API, Stripe webhooks, cron jobs
frontend/    # React + Vite + Tailwind UI
services/    # extraction, PII scrubber, embeddings
infra/       # Docker-compose, Terraform, GitHub Actions
docs/        # architecture diagrams, this README
```

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/aerafal19898/QueryLexv2.git
cd QueryLexv2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```bash
# API Keys
OPENROUTER_API_KEY=your_key
STRIPE_SECRET_KEY=your_key
STRIPE_WEBHOOK_SECRET=your_key

# Security
SECRET_KEY=generate_secure_key
JWT_SECRET_KEY=generate_secure_key
DOCUMENT_ENCRYPTION_KEY=generate_32_byte_key

# Database
CHROMA_HOST=localhost
CHROMA_PORT=8000

# Feature Flags
DEBUG=false
ENCRYPTION_ENABLED=true
SECURE_PROCESSING=true
```

4. Start the development server:
```bash
python run.py
```

## 💳 Billing & Payouts

- **Stripe Connect** integration for secure payments
- **Automatic Payouts** to creators
- **Full Refund** capability for administrators
- **Usage-based Billing** with credit system

## 🔒 Security Features

- **End-to-end Encryption** for all documents
- **Role-based Access Control** (RBAC)
- **Multi-factor Authentication**
- **Secure Document Processing**
- **Private API Access** with JWT
- **Encrypted Document Search**

## 🤝 Contributing

We welcome contributions! Please read our contributing guidelines before submitting pull requests.

## 📄 License

MIT License - See LICENSE file for details