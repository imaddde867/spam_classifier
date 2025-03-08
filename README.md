# Email Spam Detection using Logistic Regression

This project demonstrates how to build a spam detection system using logistic regression. It includes downloading and preprocessing email data from the SpamAssassin public corpus, extracting features such as text content, subject, sender, and metadata like send time and content length. The text is vectorized using TF-IDF with bigrams, and numerical features are scaled. A logistic regression model is trained with balanced class weights and evaluated using classification metrics and visualizations.

## Table of Contents

- [Requirements](#requirements)
- [Dataset](#dataset)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Requirements

To run this notebook, you need the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `wordcloud`
- `tarfile`
- `urllib`

You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn wordcloud
```

**Note:** `tarfile` and `urllib` are part of the Python standard library, so no additional installation is needed for them.

## Dataset

The dataset used in this project is sourced from the SpamAssassin public corpus, specifically the `easy_ham` and `spam` datasets from 2002. The notebook automatically downloads and extracts these datasets:

- **Ham (non-spam) emails**: [20021010_easy_ham.tar.bz2](https://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2)
- **Spam emails**: [20021010_spam.tar.bz2](https://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2)

## Usage

Follow these steps to set up and run the project:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd your-repo-name
   ```

3. **Open the Jupyter notebook**:
   ```bash
   jupyter notebook Email_Spam_Detection.ipynb
   ```

4. **Run the notebook**:
   - Execute the cells in sequence to download the data, preprocess it, train the model, and evaluate the results.

**Note**: An internet connection is required to download the datasets when running the notebook for the first time.

## Project Structure

The Jupyter notebook is organized into the following sections:

1. **Setup and Data Download**
   - Installs necessary libraries (if not already installed).
   - Downloads and extracts the `easy_ham` and `spam` datasets into a `data` directory.

2. **Data Loading and Preparation**
   - Loads email filenames from the extracted datasets.
   - Creates a DataFrame with labels (0 for ham, 1 for spam) and generates unique IDs for each email.

3. **Feature Extraction**
   - Extracts email headers (e.g., sender, subject, date, content-type) and body content.
   - Handles exceptions during extraction to ensure robustness.

4. **Feature Engineering**
   - Extracts the time of day from email dates.
   - Calculates the length of the email content and subject (clipped at the 99th percentile to handle outliers).
   - Combines text features (content, subject, sender) into a single column for analysis.

5. **Exploratory Data Analysis (EDA)**
   - Visualizes the distribution of spam vs. ham emails using a count plot.
   - Plots the distribution of email send times by class using a kernel density estimate (KDE).
   - Generates word clouds to highlight common words in spam and ham emails.

6. **Model Training**
   - Splits the data into training and testing sets (80-20 split, stratified by class).
   - Builds a pipeline with:
     - TF-IDF vectorization for text features (with bigrams, stop words, and frequency filtering).
     - MinMax scaling for numerical features (time of day, content length, subject length).
   - Trains a logistic regression model with balanced class weights.

7. **Model Evaluation**
   - Generates a classification report with precision, recall, and F1-score for both classes.
   - Plots a confusion matrix to visualize prediction performance.
   - Displays the ROC curve and calculates the Area Under the Curve (AUC).
   - Shows the precision-recall curve for further evaluation.

## Results

The logistic regression model effectively distinguishes between spam and ham emails. Key performance metrics, including precision, recall, F1-score, and AUC, are provided in the notebook. Visualizations such as the confusion matrix, ROC curve, and precision-recall curve offer insights into the model's performance. Users can run the notebook to see the exact results based on the processed dataset.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please feel free to:
- Submit a pull request with your changes.
- Open an issue to discuss ideas or report bugs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
