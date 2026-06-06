import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import glob
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for plots if it doesn't exist
os.makedirs("model_plots", exist_ok=True)

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.stem import WordNetLemmatizer

tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

negation_words = {
    'no', 'not', 'never', 'none', 'neither', 'nor',
    "don't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't",
    "can't", "cannot", "couldn't",
    "won't", "wouldn't",
    "shouldn't", "mustn't"
}
final_stop_words = stop_words - negation_words

# --- HELPER: LEARNING CURVE ONLY (To save space) ---
def save_learning_curve(estimator, X, y, title, filename):
    print(f"Generating Learning Curve for {title}...")
    plt.figure(figsize=(10, 6))
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=3, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='accuracy'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")

    plt.title(f"Learning Curve: {title}")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.grid(True)
    
    plt.savefig(f"model_plots/{filename}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Learning Curve: model_plots/{filename}")


# --- PREPROCESSING ---
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word not in final_stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


# --- 1. SENTIMENTAL ANALYSIS MODEL ---

def load_data(file_path):
    col_names = ['ID', 'Category', 'Sentiment', 'Text']
    try:
        data = pd.read_csv(file_path, header=None, names=col_names)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None
    data['Text'] = data['Text'].fillna("")
    data['cleaned_text'] = data['Text'].apply(preprocess_text)
    return data

def train_sentiment_model():
    print("\n--- Training Sentimental Analysis Model ---")
    data = load_data("twitter_training.csv")
    if data is None: return

    X_train, X_test, y_train, y_test = train_test_split(
        data['cleaned_text'], data['Sentiment'], test_size = 0.2, random_state = 400
    )

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    svm_classifier = LinearSVC(class_weight='balanced', dual='auto')
    svm_classifier.fit(X_train_tfidf, y_train)

    y_pred = svm_classifier.predict(X_test_tfidf)
    print(f"Accuracy: {accuracy_score(y_test, y_pred): .4f}")
    print("\nReport:\n", classification_report(y_test, y_pred))

    # Learning Curve
    save_learning_curve(svm_classifier, X_train_tfidf, y_train, 
                        "Sentiment Analysis", "sentiment_learning_curve.png")

    dump((svm_classifier, tfidf_vectorizer), 'sentiment_model.joblib')


# --- 2. COMMUNITY HEALTH MODEL ---

def load_community_data(spam_folder_path, toxic_file_path):
    print("Loading community datasets...")
    spam_files = glob.glob(f"{spam_folder_path}/*.csv")
    if not spam_files:
        print(f"Error: No .csv files found in {spam_folder_path}")
        return None

    all_spam_dfs = []
    for file in spam_files:
        try:
            df = pd.read_csv(file, encoding='latin-1')
            all_spam_dfs.append(df)
        except Exception: pass
            
    if not all_spam_dfs: return None
    df_spam = pd.concat(all_spam_dfs, ignore_index=True)
    
    if 'CONTENT' in df_spam.columns and 'CLASS' in df_spam.columns:
        df_spam = df_spam[['CONTENT', 'CLASS']]
        df_spam.rename(columns={'CONTENT': 'text', 'CLASS': 'label'}, inplace=True)
        df_spam['label'] = df_spam['label'].map({1: 'spam', 0: 'healthy'})
    else: return None

    try:
        df_toxic = pd.read_csv(toxic_file_path)
    except Exception: return None
        
    toxic_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df_toxic['is_toxic'] = df_toxic[toxic_labels].sum(axis=1) > 0
    df_toxic['label'] = df_toxic['is_toxic'].map({True: 'toxic', False: 'healthy'})
    df_toxic = df_toxic[['comment_text', 'label']]
    df_toxic.rename(columns={'comment_text': 'text'}, inplace=True)

    df_spam_final = df_spam[df_spam['label'] == 'spam']
    df_toxic_final = df_toxic[df_toxic['label'] == 'toxic']
    df_healthy_sample = pd.concat([
        df_spam[df_spam['label'] == 'healthy'].sample(n=min(len(df_spam[df_spam['label']=='healthy']), 1000), random_state=42),
        df_toxic[df_toxic['label'] == 'healthy'].sample(n=min(len(df_spam_final)+len(df_toxic_final), len(df_toxic[df_toxic['label']=='healthy'])), random_state=42)
    ])
    
    final_df = pd.concat([df_spam_final, df_toxic_final, df_healthy_sample])
    return final_df.sample(frac=1, random_state=42).reset_index(drop=True)

def train_community_model():
    print("\n--- Training Community Health Model ---")
    data = load_community_data("youtube-spam-collection-v1", "train.csv")     
    if data is None: return

    data['cleaned_text'] = data['text'].apply(preprocess_text)
    X_train, X_test, y_train, y_test = train_test_split(
        data['cleaned_text'], data['label'], test_size = 0.2, random_state = 400, stratify = data['label']
    )

    tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    svm_classifier = LinearSVC(class_weight='balanced', dual='auto')
    svm_classifier.fit(X_train_tfidf, y_train)
    y_pred = svm_classifier.predict(X_test_tfidf)

    print(f"Accuracy: {accuracy_score(y_test, y_pred): .4f}")
    print("\nReport:\n", classification_report(y_test, y_pred))
     
    # Learning Curve
    save_learning_curve(svm_classifier, X_train_tfidf, y_train, 
                        "Community Health", "community_learning_curve.png")

    dump((svm_classifier, tfidf_vectorizer), 'community_model.joblib')


# --- 3. CONSTRUCTIVE FEEDBACK MODEL ---

def preprocess_constructive(text):
    if not isinstance(text, str): text = str(text)
    return text.lower()

def load_constructive_data(file_path):
    print(f"Loading constructive dataset from: {file_path}")
    try: data = pd.read_csv(file_path)
    except Exception: return None
    
    if 'pp_comment_text' in data.columns and 'constructive_binary' in data.columns:
        data = data[['pp_comment_text', 'constructive_binary']]
        data.rename(columns={'pp_comment_text': 'text', 'constructive_binary': 'label'}, inplace=True)
    else: return None

    data['label'] = data['label'].map({1: 'Constructive', 0: 'Non-Constructive'})
    data = data.dropna(subset=['text', 'label'])
    data['cleaned_text'] = data['text'].apply(preprocess_constructive)
    data['text_len'] = data['text'].apply(len)
    return data

def train_constructive_model():
    print("\n--- Training Constructive Feedback Model ---")
    data = load_constructive_data("C3_anonymized.csv")
    if data is None: return 
    
    X_train, X_test, y_train, y_test, len_train, len_test = train_test_split(
        data['cleaned_text'], data['label'], data['text_len'], test_size = 0.2, random_state=400, stratify = data['label']
    )

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=5, max_df=0.9)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    scaler = MinMaxScaler()
    len_train_scaled = scaler.fit_transform(len_train.values.reshape(-1,1))
    len_test_scaled = scaler.fit_transform(len_test.values.reshape(-1,1))

    X_train_final = hstack([X_train_tfidf, len_train_scaled])
    X_test_final = hstack([X_test_tfidf, len_test_scaled])

    svm_classifier = LinearSVC(class_weight='balanced', C=1.0, max_iter=2000, dual='auto')
    svm_classifier.fit(X_train_final, y_train)
    y_pred = svm_classifier.predict(X_test_final)

    print(f"Accuracy: {accuracy_score(y_test, y_pred): .4f}")
    print("\nReport:\n", classification_report(y_test, y_pred))

    save_learning_curve(svm_classifier, X_train_final, y_train, 
                        "Constructive Feedback", "constructive_learning_curve.png")

    dump((svm_classifier, tfidf_vectorizer, scaler), 'constructive_model.joblib')
    print("Model saved as 'constructive_model.joblib'")

if __name__ == "__main__":
    train_sentiment_model()
    train_community_model()
    train_constructive_model()