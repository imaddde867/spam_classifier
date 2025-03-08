# 📧 Spam Classifier  

A machine learning-based spam classifier using Apache SpamAssassin’s public datasets. This project preprocesses email data, converts it into feature vectors, and trains various classifiers to distinguish between spam and ham with high accuracy.  

## 📌 Project Overview  
Spam emails are a common problem in digital communication. This project aims to build a spam classifier using machine learning techniques, transforming raw email text into structured data and training different classifiers to achieve high precision and recall.  

## 📂 Dataset  
We use **Apache SpamAssassin’s public datasets**, which contain labeled spam and ham (non-spam) emails. The dataset needs to be downloaded and extracted before processing.  

- **Spam emails** 📩 - Unsolicited, bulk messages.  
- **Ham emails** 📬 - Legitimate, non-spam messages.  

## ⚙️ Data Preprocessing  
To convert raw emails into meaningful input for machine learning models, we apply the following transformations:  

✅ **Tokenization** - Splitting emails into words.  
✅ **Lowercasing** - Converting all words to lowercase.  
✅ **Removing headers** (optional) - Stripping email metadata.  
✅ **Removing punctuation** - Cleaning unnecessary characters.  
✅ **Replacing URLs & numbers** - Converting them to placeholders like `URL` and `NUMBER`.  
✅ **Stemming** - Reducing words to their base form (e.g., "running" → "run").  
✅ **Stopword removal** - Removing common words like "the", "and", "is".  

## 🔢 Feature Engineering  
Each email is converted into a **sparse feature vector**, representing the presence or frequency of words. For example:  

📩 **Email:** "Hello you Hello Hello you"  
➡️ **Vector (binary):** `[1, 0, 0, 1]` (word presence)  
➡️ **Vector (frequency):** `[3, 0, 0, 2]` (word count)  

## 🏗️ Model Training  
We experiment with multiple classifiers:  

🔹 **Naïve Bayes** (good for text classification)  
🔹 **Logistic Regression**  
🔹 **Support Vector Machines (SVM)**  
🔹 **Random Forest**  
🔹 **Deep Learning** (optional for advanced experiments)  

## 📊 Evaluation  
The classifier is evaluated using **precision, recall, F1-score**, and **confusion matrix** to ensure both high accuracy and minimal false positives/negatives.  

## 🚀 Getting Started  

### 1️⃣ Install Dependencies  
```bash
pip install numpy pandas scikit-learn nltk
```

### 2️⃣ Download and Extract Dataset  
Download the Apache SpamAssassin dataset and unzip it into the project folder.  

### 3️⃣ Run the Classifier  
```bash
python train.py
```

### 4️⃣ Evaluate Performance  
```bash
python evaluate.py
```

## 🛠️ Future Improvements  
✅ Improve feature extraction with TF-IDF  
✅ Implement deep learning models (LSTM, Transformers)  
✅ Enhance preprocessing with named entity recognition (NER)  

## 📜 License  
This project is for educational purposes and follows an open-source approach.  
