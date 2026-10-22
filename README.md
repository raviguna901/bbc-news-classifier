# BBC News Article Classification using Machine Learning

## 1. Project Overview

This project is a **News Article Classification System** that automatically predicts the category of a news article using Natural Language Processing (NLP) and Machine Learning.

The application takes a news article as input, processes its text, converts the text into numerical features using TF-IDF vectorization, and predicts the most likely category using a trained Multinomial Naive Bayes classifier.

The project also includes an interactive web application built using Streamlit, allowing users to test news articles without writing any code.

**Live Application:** https://bbc-news-classifier-on5tnhwpflwm8tbadmqkwz.streamlit.app/

## 2. Project Objectives

- Understand and preprocess textual data using NLP techniques.
- Explore and prepare a news article dataset.
- Convert text into numerical features using TF-IDF.
- Train a Multinomial Naive Bayes classification model.
- Evaluate the model using test data.
- Build and deploy an interactive Streamlit application.
- Predict the category of previously unseen news articles.

## 3. Dataset Information

**Dataset:** BBC News Text Classification Dataset

**Dataset URL:**  
https://raw.githubusercontent.com/ronaldleonardo/Data_Transformer_Datasets/refs/heads/main/bbc_text_cls.csv

The dataset contains news articles belonging to five categories.

| Category | Number of articles |
|---|---:|
| Business | 510 |
| Sport | 510 |
| Politics | 417 |
| Tech | 401 |
| Entertainment | 386 |
| **Total** | **2,224** |

**Dataset columns:**

- `text`: The news article content used as the input feature.
- `labels`: The category assigned to the article and used as the target variable.

## 4. Technologies and Libraries Used

- **Python:** Core programming language.
- **Pandas:** Data loading and manipulation.
- **NumPy:** Numerical operations.
- **NLTK:** Text preprocessing, stopwords, POS tagging and lemmatization.
- **Regular Expressions (`re`):** Text cleaning.
- **Scikit-learn:** Train-test splitting, TF-IDF vectorization, model training and evaluation.
- **Multinomial Naive Bayes:** Text classification algorithm.
- **Joblib:** Saving and loading the trained model and vectorizer.
- **Streamlit:** Building the interactive web application.
- **Git and GitHub:** Version control and source code hosting.
- **Streamlit Community Cloud:** Application deployment.

## 5. Project Workflow

The project follows this machine learning pipeline:

1. Load the BBC news dataset.
2. Inspect the dataset and target categories.
3. Clean and preprocess the text.
4. Split the data into training and testing sets.
5. Convert text into numerical features using TF-IDF.
6. Train the Multinomial Naive Bayes model.
7. Evaluate model performance on unseen test data.
8. Save the trained model and TF-IDF vectorizer.
9. Build a Streamlit application.
10. Deploy the application online.

## 6. Step-by-Step Implementation

### Step 1: Load and Understand the Dataset

The dataset is loaded into a Pandas DataFrame.

Initial exploration includes:

- Checking the number of rows and columns.
- Inspecting sample news articles.
- Checking for missing values.
- Examining the distribution of news categories.

The dataset contains 2,224 articles across five target categories.

### Step 2: Text Preprocessing

Raw text contains punctuation, inconsistent capitalization, and words that may not contribute much to classification. Preprocessing prepares the articles for feature extraction.

The text-cleaning pipeline includes:

1. **Lowercasing:** Converts text to lowercase for consistent processing.
2. **Removing punctuation and unwanted characters:** Uses regular expressions.
3. **Tokenization:** Splits text into individual words.
4. **Stopword removal:** Removes common English words using NLTK, while preserving selected negation words where applicable.
5. **Part-of-speech tagging:** Identifies grammatical roles of words.
6. **Lemmatization:** Converts words to their base forms using WordNet.

These steps help standardize the text before it is converted into numerical features.

### Step 3: Separate Features and Target

The input feature is the article text, and the target variable is the article category.

- **X:** News article text.
- **y:** News category labels.

This is a supervised multiclass classification problem because each article has a known category during training.

### Step 4: Split the Dataset

The dataset is divided into training and testing sets using an 80:20 split.

The split uses `random_state=42` and stratification to preserve the category distribution across both sets.

| Dataset | Number of articles |
|---|---:|
| Training set | 1,779 |
| Testing set | 445 |
| **Total** | **2,224** |

The training data is used to learn patterns, while the testing data evaluates performance on articles not used to fit the model.

### Step 5: Apply TF-IDF Vectorization

Machine learning algorithms require numerical input, so the cleaned text is transformed into numerical feature vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

TF-IDF gives greater importance to words that are informative within an article while reducing the relative importance of words that occur frequently across many documents.

The implementation follows these rules:

- Fit the TF-IDF vectorizer on the training text only.
- Transform the training text into feature vectors.
- Transform the test text using the same fitted vectorizer.

This prevents information from the test set from leaking into the feature extraction process.

### Step 6: Train the Multinomial Naive Bayes Model

The classification algorithm used is **Multinomial Naive Bayes**, a probabilistic algorithm commonly used for text classification.

The model learns relationships between TF-IDF features and the five news categories. When a new article is supplied, it estimates the most likely category based on the learned patterns.

The five possible output categories are:

- Business
- Entertainment
- Politics
- Sport
- Tech

### Step 7: Evaluate Model Performance

The trained model is evaluated using the held-out test set.

**Reported test accuracy: 98.43%**

The model correctly classified 438 out of 445 test articles.

Accuracy is calculated as:

Accuracy = Correct Predictions / Total Predictions

The reported accuracy is based on the test split used during model development. Results on new articles may differ.

### Step 8: Save the Model and Vectorizer

After training, the model and fitted TF-IDF vectorizer are saved using Joblib.

The project includes these saved files:

- `news_model.pkl` — The trained Multinomial Naive Bayes classifier.
- `tfidf_vectorizer.pkl` — The fitted TF-IDF vectorizer.

Saving these artifacts allows the deployed application to make predictions without retraining the model every time it starts.

### Step 9: Build the Streamlit Application

The Streamlit application provides a user-friendly interface for text classification.

The application's workflow is:

1. The user enters or pastes a news article.
2. The application loads the saved model and vectorizer.
3. The article undergoes the same text preprocessing used during training.
4. The fitted TF-IDF vectorizer transforms the article into numerical features.
5. The trained model predicts the news category.
6. The application displays the predicted category and a probability chart when available.

The probability chart displays the model's estimated class probabilities. These estimates should not be interpreted as guaranteed or calibrated confidence scores.

### Step 10: Deploy the Application

The project files are stored in a GitHub repository and the Streamlit application is deployed through Streamlit Community Cloud.

**GitHub Repository:**  
https://github.com/raviguna901/bbc-news-classifier

**Live Application:**  
https://bbc-news-classifier-on5tnhwpflwm8tbadmqkwz.streamlit.app/

The deployed application allows evaluators to enter sample news articles and observe the model's predictions directly in a browser.

## 7. Project Structure

```text
bbc-news-classifier/
│
├── app.py
├── news_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── Text Cleaning and Model Building.ipynb
├── .gitignore
└── README.md
```

**File descriptions:**

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface and prediction logic |
| `news_model.pkl` | Saved trained classification model |
| `tfidf_vectorizer.pkl` | Saved TF-IDF feature extractor |
| `requirements.txt` | Python dependency list |
| `Text Cleaning and Model Building.ipynb` | Notebook containing preprocessing, feature extraction, training and evaluation |
| `.gitignore` | Excludes virtual environments and unnecessary files |
| `README.md` | Project documentation |

## 8. How to Run the Project Locally

### Prerequisites

Install Python 3.10 or a compatible Python version supported by the dependencies.

### Step 1: Clone the repository

```bash
git clone https://github.com/raviguna901/bbc-news-classifier.git
```

### Step 2: Navigate to the project directory

```bash
cd bbc-news-classifier
```

### Step 3: Create a virtual environment (recommended)

```bash
python3 -m venv .venv
```

### Step 4: Activate the virtual environment on macOS or Linux

```bash
source .venv/bin/activate
```

### Step 5: Install the dependencies

```bash
pip install -r requirements.txt
```

If the NLTK resources required by the preprocessing code are not available, download the resources used by your notebook and application. For example:

```python
import nltk

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("averaged_perceptron_tagger")
nltk.download("omw-1.4")
```

The exact NLTK resources required depend on the functions and versions used by the application.

### Step 6: Start the Streamlit application

```bash
streamlit run app.py
```

Streamlit will provide a local URL, typically:

```text
http://localhost:8501
```

Open that address in your browser to use the application.

**Note:** The saved model and vectorizer files must be present in the project directory. You do not need to retrain the model to run the application.

## 9. Example Test Cases

Try the following types of articles in the application.

| Sample input topic | Expected category |
|---|---|
| A company reports increased profits and plans business expansion. | Business |
| A football team wins a match and qualifies for a tournament final. | Sport |
| Parliament debates a new government policy. | Politics |
| A technology company releases a smartphone with new AI features. | Tech |
| A film receives positive reviews and performs well at the box office. | Entertainment |

These are illustrative examples. The model's actual prediction may differ depending on the article's wording and content.

## 10. Key Learnings

Through this project, I gained practical experience in:

- Preparing and cleaning textual datasets.
- Applying NLP preprocessing techniques.
- Understanding TF-IDF feature extraction.
- Splitting datasets into training and testing sets.
- Training a Multinomial Naive Bayes classifier.
- Evaluating classification accuracy on unseen data.
- Saving and loading machine learning artifacts with Joblib.
- Building an interactive application with Streamlit.
- Deploying a machine learning application using GitHub and Streamlit Community Cloud.

## 11. Limitations and Future Improvements

- The classifier is trained on five predefined news categories.
- It always predicts one of the trained categories, so unrelated text may receive an incorrect news category.
- Performance may differ for articles from other sources or domains.
- The displayed class probabilities may not be calibrated confidence estimates.

Possible future improvements include adding an unknown or non-news category, expanding the training dataset, evaluating precision, recall and F1-score for each class, and calibrating predicted probabilities.

## 12. Conclusion

This project demonstrates an end-to-end text classification workflow, from data preprocessing and TF-IDF feature extraction to Multinomial Naive Bayes training, evaluation, and deployment.

The final application provides a simple way to classify news articles into five categories through a browser-based interface.

**Project status:** Completed and deployed.
