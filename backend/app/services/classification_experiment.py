
"""Reproducible, group-aware Logistic Regression experiment for JURIS."""

import json
from pathlib import Path

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from app.core.config import get_settings
from app.services import document_service, tfidf_service


def run_experiment() -> dict:
    """Evaluate Logistic Regression without saving a production model."""

    # Load document IDs from the fitted TF-IDF state.
    _, _, document_ids = tfidf_service._load_fitted_state()

    # Load document metadata.
    records = {
        doc.document_id: doc
        for doc in document_service.list_documents()
    }

    root = Path(get_settings().project_root)

    # Documents belonging to the same legal-document family.
    document_groups = {
        # Consumer Protection Act versions
        "3c77f5085a944b4ea6db0964a1193d96": "act_2019",
        "22f5078b6a9c4555bfeb005c0026946d": "act_2019",

        # E-Commerce Rules and amendment
        "60d2668aff3146edb63068274e2a0d72": "ecommerce_rules",
        "1f40f10f85954ca7823082b7ca0c3c00": "ecommerce_rules",

        # CDRC Rules and amendment
        "4b5def66ea314aadb287351497f3b8d6": "cdrc_rules",
        "d401e01d79be493384842da5916f7829": "cdrc_rules",

        # Dark Patterns Guidelines versions
        "f61428a60bff432b85411703e0c1bf90": "dark_patterns",
        "735371ef5ca54398a807325a28bf41d4": "dark_patterns",
    }

    texts = []
    labels = []
    groups = []

    # Load bilingual preprocessed tokens for every document.
    for document_id in document_ids:

        doc = records.get(document_id)

        if doc is None:
            raise ValueError(
                f"Missing document metadata: {document_id}"
            )

        path = (
            root
            / "data"
            / "processed"
            / f"{document_id}_preprocessed.json"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Preprocessed document not found: {path}"
            )

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        tokens = data.get("lemmatized_tokens", [])

        if not isinstance(tokens, list) or not tokens:
            raise ValueError(
                f"No lemmatized tokens found for document: {document_id}"
            )

        texts.append(" ".join(tokens))
        labels.append(doc.category.value)

        # Related documents share a group.
        # Independent documents receive their own group.
        groups.append(
            document_groups.get(document_id, document_id)
        )

    labels = np.asarray(labels)
    groups = np.asarray(groups)

    categories, counts = np.unique(
        labels,
        return_counts=True,
    )

    print("\nJURIS DOCUMENT SUMMARY")
    print("----------------------")
    print("Total documents:", len(texts))
    print("Category counts:", dict(zip(categories, counts)))
    print("Independent document groups:", len(set(groups)))

    if len(texts) < 4 or counts.min() < 2:
        raise ValueError(
            "Not enough documents for stratified cross-validation."
        )

    # Two folds because the smallest category has only two documents.
    folds = 2

    cv = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42,
    )

    # Inspect group-aware folds before model evaluation.
    print("\nGROUP-AWARE FOLD DIAGNOSTICS")
    print("----------------------------")

    splits = list(
        cv.split(texts, labels, groups)
    )

    valid_training_folds = True

    for fold_number, (train_idx, test_idx) in enumerate(
        splits,
        start=1,
    ):
        train_categories = sorted(
            set(labels[train_idx])
        )

        test_categories = sorted(
            set(labels[test_idx])
        )

        missing_training_categories = sorted(
            set(categories) - set(train_categories)
        )

        print(f"\nFold {fold_number}")
        print("Training categories:", train_categories)
        print("Testing categories:", test_categories)
        print("Training documents:", len(train_idx))
        print("Testing documents:", len(test_idx))
        print(
            "Missing training categories:",
            missing_training_categories,
        )

        if missing_training_categories:
            valid_training_folds = False

    # Stop rather than report misleading evaluation metrics.
    if not valid_training_folds:
        return {
            "status": "not_evaluable",
            "documents": len(texts),
            "category_counts": {
                str(category): int(count)
                for category, count in zip(categories, counts)
            },
            "folds": folds,
            "reason": (
                "At least one training fold is missing a category. "
                "The current document families do not support reliable "
                "group-aware evaluation across all categories."
            ),
            "recommendation": (
                "Collect additional independent legal-document families "
                "for the affected categories before calculating a "
                "group-aware performance score."
            ),
        }

    # Leakage-safe pipeline:
    # TF-IDF and feature selection are fitted separately within each fold.
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                strip_accents=None,
                token_pattern=r"(?u)\b\w+\b",
                ngram_range=(1, 2),
                sublinear_tf=True,
            ),
        ),
        (
            "selector",
            SelectKBest(chi2, k=500),
        ),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=3000,
                random_state=42,
            ),
        ),
    ])

    # Pass groups explicitly to the group-aware cross-validation.
    predictions = cross_val_predict(
        pipeline,
        texts,
        labels,
        cv=cv,
        groups=groups,
    )

    return {
        "status": "completed",
        "documents": len(texts),
        "category_counts": {
            str(category): int(count)
            for category, count in zip(categories, counts)
        },
        "folds": folds,
        "accuracy": round(
            float(accuracy_score(labels, predictions)),
            4,
        ),
        "macro_f1": round(
            float(
                f1_score(
                    labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        ),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=categories.tolist(),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": {
            "labels": categories.tolist(),
            "matrix": confusion_matrix(
                labels,
                predictions,
                labels=categories,
            ).tolist(),
        },
        "warning": (
            "Exploratory estimate only. The corpus is small and "
            "the results should not be interpreted as a robust "
            "estimate of real-world classification performance."
        ),
    }


if __name__ == "__main__":
    result = run_experiment()
    print("\nEXPERIMENT RESULT")
    print("-----------------")
    print(json.dumps(result, indent=2))
