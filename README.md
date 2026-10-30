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
8. Emb