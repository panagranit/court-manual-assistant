# Court Clerk Manual Assistant

A Retrieval-Augmented Generation (RAG) web application that answers questions about court clerk procedures using a 67-page court clerk manual as its knowledge base. Built with Python, OpenAI, FAISS, and deployed on Vercel.

## Features

- **RAG-Powered Answers**: Uses vector embeddings and semantic search to find relevant manual sections
- **GPT-4o-mini Integration**: Generates accurate answers strictly from manual content
- **Serverless Deployment**: Runs on Vercel with automatic scaling
- **Public Access**: Shareable URL accessible to anyone
- **Secure API Usage**: OpenAI API key stored securely on server only

## Project Structure

```
court-manual-assistant/
├── api/
│   └── chat.py              # Serverless API endpoint for chat requests
├── rag/
│   ├── build_index.py       # Script to build FAISS index from manual
│   ├── manual.txt           # Court clerk manual (knowledge base)
│   ├── manual.index         # FAISS vector index (generated)
│   └── chunks.json          # Text chunks (generated)
├── public/
│   └── index.html           # Frontend web interface
├── requirements.txt         # Python dependencies
├── vercel.json             # Vercel deployment configuration
└── README.md               # This file
```

## Setup Instructions

### Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Vercel account (for deployment)

### Local Development

#### 1. Create Virtual Environment

```bash
# Navigate to project directory
cd court-manual-assistant

# Create virtual environment
python -m venv .venv

# Activate virtual environment