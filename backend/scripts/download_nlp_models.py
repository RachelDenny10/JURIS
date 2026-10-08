"""
One-time setup script: downloads the NLTK data and spaCy model that
JURIS's preprocessing service depends on.

Run this once after installing requirements.txt, and again if you
ever recreate your virtual environment.
"""

import subprocess
import sys

import nltk


def download_nltk_data():
    print("Downloading NLTK data (punkt, punkt_tab, stopwords)...")
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")


def download_spacy_model():
    print("Downloading spaCy model 'en_core_web_sm'...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])


if __name__ == "__main__":
    download_nltk_data()
    download_spacy_model()
    print("Done. NLP models are ready.")