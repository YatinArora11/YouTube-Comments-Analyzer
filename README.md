# Enhanced YouTube Comment Analyzer

This project analyzes YouTube comments using three specialized NLP-based machine learning models:

1. Sentiment Analysis Model
2. Community Health Detection Model
3. Constructive Feedback Detection Model

The system is deployed using a Streamlit dashboard and integrates with the YouTube Data API to fetch real-time comments.

## Project Objective

YouTube creators receive large volumes of comments, making it difficult to manually identify audience sentiment, toxic/spam comments, and useful feedback. This project solves that using a three-model ML pipeline.

## Models Used

### 1. Sentiment Analysis
Classifies comments into:
- Positive
- Negative
- Neutral
- Irrelevant

### 2. Community Health Detection
Classifies comments into:
- Healthy
- Toxic
- Spam

### 3. Constructive Feedback Detection
Classifies comments into:
- Constructive
- Non-Constructive

## Datasets Used

| Dataset | Approximate Rows |
|---|---:|
| Twitter Training Dataset | ~74,682 |
| YouTube Spam Collection Dataset | ~1,956 |
| Jigsaw Toxic Comment Dataset | ~159,571 |
| C3 Constructive Comments Dataset | ~11,000–12,000 |

## Key Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Sentiment Analysis | 92.13% | 92.17% | 92.13% | 92.11% |
| Community Health | 89.67% | 89.91% | 89.67% | 89.48% |
| Constructive Feedback | 91.67% | 91.68% | 91.67% | 91.66% |

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- NLTK
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Altair
- YouTube Data API
- Joblib

## System Architecture

```text
YouTube Video URL
        ↓
Extract Video ID
        ↓
Fetch Comments using YouTube Data API
        ↓
Preprocess Text
        ↓
TF-IDF Vectorization
        ↓
Three ML Models
        ↓
Streamlit Dashboard
