# JURIS – Hybrid Legal Document Text Mining and Consumer Law Decision Support System

**JURIS** is a hybrid legal-document text mining and consumer-law decision-support system designed to help users understand consumer-related legal information through document processing, text mining, machine learning, semantic search, and AI-assisted analysis.

The system processes legal documents in different formats, extracts meaningful textual information, organizes documents into legal categories, and helps users identify relevant legal provisions based on their consumer-related situations.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Key Features](#key-features)
- [Text Mining Techniques](#text-mining-techniques)
- [System Architecture](#system-architecture)
- [Document Categories](#document-categories)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Environment Configuration](#environment-configuration)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Dataset and Models](#dataset-and-models)
- [Future Enhancements](#future-enhancements)

---

## Project Overview

Legal documents often contain complex language, lengthy provisions, and technical terminology that can make it difficult for consumers to identify the information relevant to their problems.

JURIS addresses this challenge through an integrated text-mining pipeline that combines traditional machine-learning techniques with semantic retrieval and AI-assisted legal information analysis.

The system supports both machine-readable and scanned PDF documents through text extraction and OCR processing.

It applies preprocessing, feature extraction, document classification, semantic indexing, and information retrieval to organize and retrieve relevant legal content.

The system also provides situation-based analysis to help users understand potentially applicable consumer-law provisions and relevant information.

## Objectives

- Automate the extraction of textual information from legal PDF documents.
- Support scanned and image-based legal documents through OCR.
- Preprocess legal text to improve its quality and consistency.
- Categorize legal documents using machine-learning techniques.
- Identify important keywords and entities from legal content.
- Generate document summaries.
- Retrieve relevant legal passages using keyword and semantic search.
- Provide AI-assisted analysis of consumer-related situations.
- Improve accessibility and understanding of consumer-law information.

## Key Features

### 1. Legal Document Processing
- Upload and process PDF documents.
- Extract text using PyMuPDF.
- Apply OCR using Tesseract for scanned or image-based content.
- Store extracted and preprocessed textual information.

### 2. Text Preprocessing
- Text normalization and cleaning.
- Tokenization.
- Stop-word removal.
- Lemmatization.
- Preparation of text for downstream machine-learning tasks.

### 3. Automatic Document Classification
- Organizes legal documents into predefined categories.
- Uses TF-IDF for textual feature extraction.
- Supports machine-learning classification using Logistic Regression and Multinomial Naive Bayes.
- Includes feature selection using Chi-square.

### 4. Keyword-Based Search
- Searches legal documents using relevant keywords.
- Uses TF-IDF-based textual representations.
- Helps identify relevant legal content.

### 5. Semantic Search
- Generates document embeddings using Sentence Transformers.
- Divides legal documents into searchable chunks.
- Uses FAISS for vector similarity search.
- Retrieves semantically related passages even when the exact search terms are not present.

### 6. Legal Entity Extraction
- Identifies relevant entities and information from legal documents.
- Organizes extracted information for further analysis.

### 7. Document Summarization
- Generates concise summaries of legal documents.
- Helps users understand lengthy legal content more efficiently.

### 8. Consumer Situation Analysis
- Accepts consumer-related situations as input.
- Retrieves relevant legal passages.
- Uses an integrated LLM service to assist in generating contextual analysis.
- Presents relevant legal information to support user understanding.

**Note:** JURIS is an informational and decision-support system, not a substitute for professional legal advice.

---

## Text Mining Techniques

JURIS incorporates several text-mining concepts throughout its processing pipeline.

| Technique | Application in JURIS |
|---|---|
| Text Extraction | Extracts text from PDF documents |
| OCR | Processes scanned and image-based documents |
| Tokenization | Converts text into individual tokens |
| Stop-word Removal | Removes common words with limited analytical value |
| Lemmatization | Converts words into their base forms |
| TF-IDF | Converts textual content into numerical feature vectors |
| Chi-square Feature Selection | Selects informative textual features |
| Machine Learning Classification | Categorizes legal documents |
| Named Entity Extraction | Identifies meaningful entities |
| Text Summarization | Produces concise document summaries |
| Sentence Embeddings | Represents legal text as dense numerical vectors |
| Semantic Similarity | Retrieves contextually related legal passages |
| FAISS Indexing | Enables efficient vector similarity retrieval |

### Processing Pipeline

1. Legal document upload.
2. PDF text extraction or OCR.
3. Text cleaning and normalization.
4. Tokenization and stop-word removal.
5. Lemmatization.
6. Feature extraction and document classification.
7. Entity extraction and summarization.
8. Embedding generation and vector indexing.
9. Keyword and semantic retrieval.
10. Consumer situation analysis.

---

## System Architecture

JURIS follows a frontend-backend architecture.

**Frontend**
- Provides the user interface.
- Allows users to interact with document processing, search, classification, summarization, and situation-analysis features.

**Backend**
- Provides REST API endpoints using FastAPI.
- Handles document processing and NLP operations.
- Manages machine-learning models and semantic retrieval.
- Integrates the LLM service for situation-based analysis.

**Data and Model Layer**
- Stores processed document text and metadata.
- Stores trained classification models.
- Stores document embeddings and vector indexes.
- Supports document retrieval and analysis.

---

## Document Categories

JURIS organizes consumer-law documents into six predefined categories.

| Category | Description |
|---|---|
| A | Primary Consumer-Related Legislation |
| B | Consumer Protection Rules |
| C | Consumer Protection Regulations |
| D | Special Consumer Rules |
| E | Consumer Protection Guidelines |
| F | Consumer Law Case Decisions |

These categories provide a structured organization for legal documents and support document retrieval and classification.

---

## Technology Stack

### Frontend
- React
- TypeScript
- Vite
- CSS

### Backend
- Python
- FastAPI
- Uvicorn
- Pydantic

### Text Mining and Machine Learning
- NLTK
- spaCy
- Scikit-learn
- NumPy
- Joblib
- NetworkX

### Document Processing
- PyMuPDF
- Tesseract OCR
- Pytesseract
- Pillow

### Semantic Search
- Sentence Transformers
- PyTorch
- FAISS

### Database and Storage
- SQLAlchemy
- SQLite / PostgreSQL (depending on configuration)
- JSON-based processed document storage

### AI Integration
- Configurable LLM service
- Gemini or other supported provider, depending on configuration

---

## Project Structure

```text
JURIS/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── analysis.py
│   │   │   ├── classification.py
│   │   │   ├── documents.py
│   │   │   ├── embeddings.py
│   │   │   ├── entities.py
│   │   │   ├── search.py
│   │   │   ├── situation.py
│   │   │   └── summarization.py
│   │   │
│   │   ├── core/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── main.py
│   │
│   ├── scripts/
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
│
├── data/
│   ├── processed/
│   ├── raw/
│   └── metadata/
│
├── models/
│   ├── classification/
│   ├── embeddings/
│   └── vector_store/
│
├── storage/
│
├── .env.example
├── .gitignore
└── README.md
```

---

## Installation and Setup

### Prerequisites

Install the following before running JURIS:

- Python 3.10 or later (use a version compatible with the pinned dependencies).
- Node.js and npm.
- Git.
- Tesseract OCR.
- A configured LLM API key if using AI-assisted analysis.

### 1. Clone the Repository

```bash
git clone https://github.com/RachelDenny10/JURIS.git
```

Navigate into the project directory:

```bash
cd JURIS
```

### 2. Backend Setup

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install backend dependencies:

```bash
pip install -r backend/requirements.txt
```

### 3. Environment Configuration

Create your local environment file using the example:

```powershell
Copy-Item .env.example .env
```

Open `.env` and configure the required database settings, model configuration, and your own API credentials.

**Never commit your `.env` file or publish API keys.**

### 4. Frontend Setup

Open another terminal and navigate to the frontend:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

---

## Running the Application

### 1. Start the Backend

From the project root, activate the virtual environment and run:

```bash
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8001 --reload
```

Backend URL:

http://127.0.0.1:8001

### 2. Start the Frontend

In another terminal:

```bash
cd frontend
npm run dev
```

Open the local URL displayed by Vite, usually:

http://localhost:5173

### 3. Access API Documentation

FastAPI provides interactive API documentation:

http://127.0.0.1:8001/docs

---

## API Documentation

The backend exposes REST API endpoints for the different JURIS modules.

| Endpoint | Purpose |
|---|---|
| `/health` | Backend health check |
| `/docs` | Interactive API documentation |
| `/search/semantic` | Semantic legal-document search |
| `/analysis/tfidf/fit` | Fit or refit the TF-IDF representation |
| `/analysis/tfidf/status` | Check TF-IDF model status |
| `/search/index/build` | Build the semantic search index |
| `/situation/analyze` | Analyze a consumer-related situation |

Refer to the interactive Swagger documentation at `/docs` for available request formats and responses.

---

## Dataset and Models

JURIS uses a collection of consumer-law documents organized into six categories.

The repository includes processed textual data and pretrained model artifacts to support the existing text-mining and retrieval workflow.

The shared model artifacts include:

- TF-IDF vectorizer and feature matrix.
- Logistic Regression classification model.
- Multinomial Naive Bayes classification model.
- Chi-square feature selector.
- Sentence embedding vectors.
- Document chunk metadata.
- FAISS semantic search index and metadata.

**Dataset note:** Original PDF documents and local database files may not be included in the public repository because of redistribution permissions and local configuration. The shared processed data and model artifacts are provided separately from those original files. Rebuilding the database or processing additional documents may require the original documents and appropriate configuration.

---

## Future Enhancements

- Expand the legal-document dataset.
- Improve classification accuracy with additional training data.
- Enhance legal entity recognition.
- Improve semantic retrieval and ranking.
- Support additional document formats.
- Add multilingual legal-document processing.
- Improve explainability and traceability of retrieved legal provisions.
- Expand consumer-law decision-support capabilities.

---

## Disclaimer

JURIS is an academic and research-oriented project developed for legal-document text mining and consumer-law information support.

The information generated by the system is intended for educational and informational purposes. It does not constitute legal advice, and users should consult qualified legal professionals for specific legal matters.

---

## Author

**Developed as an academic project**

**Project Name:** JURIS – Hybrid Legal Document Text Mining and Consumer Law Decision Support System

**Repository:** https://github.com/RachelDenny10/JURIS
