##################################################################################
# SENTIMENT ANALYSIS - FINAL VERSION (Bagging + Boosting + TF-IDF + Clean Labels)
##################################################################################

import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

line = '___' * 30


##################################################################################
# 1️⃣ CLEAN TEXT FUNCTION
##################################################################################

def clean_text(text):
    """Text cleaning: remove URLs, mentions, punctuations, digits, etc."""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)  # remove mentions
    text = re.sub(r"#\w+", "", text)  # remove hashtags
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuations
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text


##################################################################################
# 2️⃣ LOAD + PREPROCESS DATA
##################################################################################

def load_and_preprocess(dataset):
    df = pd.read_csv(dataset, encoding="latin1")

    # Ensure expected columns exist
    if 'text' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'sentiment' columns")

    # Drop missing values
    df = df[['text', 'sentiment']].dropna()

    # Clean text
    df['clean_text'] = df['text'].apply(clean_text)

    print("✅ Data cleaned successfully")
    print(df.head())
    print(line)
    return df


##################################################################################
# 3️⃣ FEATURE EXTRACTION (TF-IDF)
##################################################################################

def vectorize_text(df):
    """Convert text to TF-IDF features."""
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = tfidf.fit_transform(df['clean_text'])
    print("✅ TF-IDF vectorization complete")
    print(f"Shape of TF-IDF matrix: {X.shape}")
    print(line)
    return X, tfidf


##################################################################################
# 4️⃣ TRAIN & EVALUATE MODELS
##################################################################################

def train_models(X, y, le):
    """Train multiple models and print evaluation reports."""
    results = {}

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42
        ),
    }

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    for name, model in models.items():
        print(line)
        print(f"🚀 Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print("___" * 30)
        print(f"🎯 {name} Results:")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        results[name] = acc

    return results


##################################################################################
# 5️⃣ MAIN FUNCTION
##################################################################################

def main():
    print(line)
    print("🧠 Sentiment Analysis Case Study (Final Version - Bagging + Boosting + TF-IDF)")
    print(line)

    dataset = "test.csv"

    # Load & preprocess
    df = load_and_preprocess(dataset)

    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(df['sentiment'])

    # Vectorize text
    X, tfidf = vectorize_text(df)

    # Train and evaluate models
    results = train_models(X, y, le)

    print(line)
    print("🏆 FINAL MODEL ACCURACIES:")
    for model, acc in results.items():
        print(f"{model}: {acc:.4f}")


##################################################################################
# 6️⃣ RUN SCRIPT
##################################################################################

if __name__ == "__main__":
    main()
