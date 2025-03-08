# Standard library imports
import os
import re
import email
import hashlib
from datetime import datetime
from email import policy
from email.utils import parsedate_tz
import tarfile
from urllib.request import urlretrieve

# Data handling and analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, roc_curve, auc)

# Configuration
plt.style.use('ggplot')
sns.set_palette("husl")

# Constants
DATA_DIR = 'data'
HAM_URL = 'https://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2'
SPAM_URL = 'https://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2'

def download_and_extract_data():
    """Download and extract datasets if they don't exist"""
    if not os.path.exists(DATA_DIR):
        print("Downloading and extracting datasets...")
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Download files
        ham_path, _ = urlretrieve(HAM_URL, 'easy_ham.tar.bz2')
        spam_path, _ = urlretrieve(SPAM_URL, 'spam.tar.bz2')
        
        # Extract files
        with tarfile.open(ham_path, 'r:bz2') as tar:
            tar.extractall(DATA_DIR)
        with tarfile.open(spam_path, 'r:bz2') as tar:
            tar.extractall(DATA_DIR)
        print("Download and extraction complete.")
    else:
        print("Data directory already exists, skipping download.")

def load_email_files():
    """Load email filenames, handling hidden files"""
    ham_dir = os.path.join(DATA_DIR, 'easy_ham')
    spam_dir = os.path.join(DATA_DIR, 'spam')
    
    ham_files = [f for f in os.listdir(ham_dir) 
                if not f.startswith('.') and (f.endswith('.txt') or len(f) > 20)]
    spam_files = [f for f in os.listdir(spam_dir) 
                 if not f.startswith('.') and (f.endswith('.txt') or len(f) > 20)]
    
    print(f"Ham files: {len(ham_files)}, Spam files: {len(spam_files)}")
    return ham_files, spam_files

def create_dataframe(ham_files, spam_files):
    """Create DataFrame with UUIDs and spam labels"""
    df = pd.DataFrame({
        'filename': ham_files + spam_files,
        'is_spam': [0]*len(ham_files) + [1]*len(spam_files)
    })
    
    # Generate UUIDs using SHA-256 hashing
    df['email_id'] = df['filename'].apply(
        lambda x: hashlib.sha256(x.encode()).hexdigest()[:16]
    )
    return df[['email_id', 'is_spam', 'filename']]

def extract_header(header_name, content):
    """Extract email headers using email module"""
    try:
        msg = email.message_from_string(content, policy=policy.default)
        return msg.get(header_name, None)
    except Exception as e:
        print(f"Error parsing header: {e}")
        return None

def extract_email_content(row):
    """Extract content and headers from raw email text"""
    try:
        content = row['raw_content']
        
        return {
            'sender': extract_header('From', content),
            'subject': extract_header('Subject', content),
            'date': extract_header('Date', content),
            'content_type': extract_header('Content-Type', content),
            'content': '\n'.join(
                part.get_payload() for part in email.message_from_string(content).walk()
                if part.get_content_type() == 'text/plain'
            )
        }
    except Exception as e:
        print(f"Error processing {row['filename']}: {e}")
        return None

def extract_time_of_day(date_str):
    """Extract hour from date string with multiple fallbacks"""
    if pd.isna(date_str):
        return None
    try:
        # Try email module parsing first
        parsed = parsedate_tz(date_str)
        if parsed:
            return parsed[3]  # Return hour
        
        # Regex fallback patterns
        patterns = [
            r'\d{1,2}:\d{1,2}:\d{1,2}',  # HH:MM:SS
            r'\d{1,2}:\d{1,2}',          # HH:MM
            r'\d{1,2}[hH]',              # 12h format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, date_str)
            if match:
                return int(re.search(r'\d{1,2}', match.group()).group())
        
        return None
    except Exception as e:
        print(f"Error parsing date {date_str}: {e}")
        return None

def clean_text(text):
    """Clean and tokenize text while preserving numbers"""
    if pd.isna(text):
        return []
    try:
        # Remove special chars but keep numbers
        text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower())
        return [word for word in text.split() if len(word) > 2]
    except:
        return []

def generate_word_clouds(df):
    """Generate word clouds and frequency plots"""
    spam_text = ' '.join(df[df['is_spam'] == 1]['combined_text'].astype(str))
    ham_text = ' '.join(df[df['is_spam'] == 0]['combined_text'].astype(str))

    # Enhanced stop words
    custom_stop_words = list(ENGLISH_STOP_WORDS) + [
        'com', 're', 'edu', 'net', 'org', 'http', 'www', 'html'
    ]
    
    # Process texts
    spam_words = clean_text(spam_text)
    ham_words = clean_text(ham_text)
    
    # Remove stop words
    spam_words = [w for w in spam_words if w not in custom_stop_words]
    ham_words = [w for w in ham_words if w not in custom_stop_words]
    
    # Create figures
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Plot frequency distributions
    for i, (words, color, title) in enumerate(zip(
        [spam_words, ham_words],
        ['red', 'green'],
        ['Spam', 'Ham']
    )):
        word_counts = Counter(words)
        common_words = word_counts.most_common(15)
        words, counts = zip(*common_words)
        
        axes[0, i].barh(words, counts, color=color, alpha=0.7)
        axes[0, i].set_title(f'Top 15 Words in {title} Emails')
        axes[0, i].invert_yaxis()
    
    # Generate word clouds
    wc_params = {
        'width': 800,
        'height': 400,
        'background_color': 'white',
        'max_words': 100,
        'stop_words': custom_stop_words
    }
    
    spam_wc = WordCloud(**wc_params).generate(' '.join(spam_words))
    ham_wc = WordCloud(**wc_params).generate(' '.join(ham_words))
    
    axes[1, 0].imshow(spam_wc, interpolation='bilinear')
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Spam Word Cloud')
    
    axes[1, 1].imshow(ham_wc, interpolation='bilinear')
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Ham Word Cloud')
    
    plt.tight_layout()
    plt.show()

def main():
    # Data pipeline
    download_and_extract_data()
    ham_files, spam_files = load_email_files()
    df = create_dataframe(ham_files, spam_files)
    
    # Read file contents
    contents = []
    for _, row in df.iterrows():
        dir_type = 'easy_ham' if row['is_spam'] == 0 else 'spam'
        path = os.path.join(DATA_DIR, dir_type, row['filename'])
        try:
            with open(path, 'rb') as f:
                contents.append(f.read().decode('latin-1'))
        except Exception as e:
            print(f"Error reading {path}: {e}")
            contents.append('')
    df['raw_content'] = contents
    
    # Extract structured features
    features = df.apply(extract_email_content, axis=1).apply(pd.Series)
    df = pd.concat([df, features], axis=1)
    
    # Feature engineering
    df['time_of_day'] = df['date'].apply(extract_time_of_day)
    df['content_length'] = df['content'].str.len().clip(upper=df['content'].str.len().quantile(0.99))
    df['subject_length'] = df['subject'].str.len().clip(upper=df['subject'].str.len().quantile(0.99))
    df['combined_text'] = df[['content', 'subject', 'sender', 'content_type']].fillna('').astype(str).agg(' '.join, axis=1)
    
    # Handle missing values
    df['time_of_day'] = df['time_of_day'].fillna(df['time_of_day'].median())
    df = df.dropna(subset=['content'])
    
    # Visualization
    generate_word_clouds(df)
    
    # Prepare features
    X = df[['combined_text', 'time_of_day', 'content_length', 'subject_length']]
    y = df['is_spam']
    
    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=19, stratify=y
    )
    
    # Model pipeline
    preprocessor = ColumnTransformer([
        ('text', TfidfVectorizer(
            tokenizer=clean_text,
            stop_words=list(ENGLISH_STOP_WORDS),
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=5
        ), 'combined_text'),
        ('numerical', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ], ['time_of_day', 'content_length', 'subject_length'])
    ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=19,
            solver='liblinear'
        ))
    ])
    
    # Training
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix')
    plt.show()
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Logistic Regression')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()