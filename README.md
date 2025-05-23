# Research Topic Prediction Project Analysis

## Project Overview

This project implements a multi-label text classification system to predict research topics for academic papers based on their titles and abstracts. The system analyzes the content of research papers and classifies them into one or more of six academic disciplines:

1. Computer Science
2. Physics
3. Mathematics
4. Statistics
5. Quantitative Biology
6. Quantitative Finance

## Process & Pipeline

1. Data Acquisition & Exploration
   Loading training and test data from CSV files
   Initial data exploration to understand dataset dimensions
   Identification of features (title and abstract) and target labels (research topics)
2. Text Preprocessing
   Cleaning text data:
   Removing punctuation using regex patterns [^a-zA-Z]
   Converting all text to lowercase for consistency
   Removing single-letter words that add no semantic value
   Eliminating redundant whitespaces
   Feature combination: Merging title and abstract into a single "combined" text field to create a unified representation of each document
3. Feature Engineering
   Text vectorization: Converting text documents to numerical feature vectors
   N-gram modeling: Using both unigrams and bigrams (ngram_range=(1,2)) to capture word combinations and context
   Bag-of-words representation: Using CountVectorizer to create a term-document matrix
   TF-IDF transformation: Applying term frequency-inverse document frequency weighting to emphasize important words while reducing the impact of common terms
4. Model Training
   Train-test split: Dividing data (90%-10%) to enable model evaluation
   Multi-label classification: Using MultiOutputClassifier to handle the multi-label nature of the problem
   Model selection: Employing LinearSVC (Support Vector Classification) with balanced class weights as the base classifier
5. Evaluation & Prediction
   Performance metrics: Examining precision, recall, F1-score, and overall accuracy
   Final prediction: Making predictions on the test dataset
   Output generation: Creating a submission file with prediction probabilities for each topic

## Tech Stack

Programming Language
Python: Primary language for implementation
Core Libraries
NumPy: Numerical operations and array handling
Pandas: Data manipulation and analysis
Matplotlib: Visualization (though minimally used)
Machine Learning & NLP
scikit-learn:
LinearSVC: Support Vector Classification algorithm
MultiOutputClassifier: Extension for multi-label classification
CountVectorizer: Text tokenization and vectorization
TfidfTransformer: TF-IDF feature weighting
train_test_split: Data partitioning for evaluation
Various metrics modules for model evaluation
NLTK (Natural Language Toolkit):
Text tokenization tools
Stopwords corpus for filtering common words
WordNet for lemmatization capabilities

## Key Concepts Applied

Natural Language Processing
Text preprocessing: Standard NLP pipeline for cleaning and normalizing text
Feature extraction: Converting unstructured text to structured numerical features
Semantic representation: Capturing document meaning through statistical word patterns
Machine Learning
Multi-label classification: Handling papers that belong to multiple research areas simultaneously
Linear Support Vector Machines: Efficient algorithm for text classification that works well in high-dimensional spaces
Class balancing: Using class_weight='balanced' to handle potential imbalance in topic distribution
Information Retrieval
TF-IDF weighting: Prioritizing terms that are important to a document but not overly common in the corpus
N-gram modeling: Capturing word combinations that may have special meaning

## Output & Results

The final output is a CSV file (submission2.csv) containing predicted topics for each research paper in the test set
Each row represents one paper with its ID and six binary indicators (0 or 1) for each research topic
The model achieves moderate to high performance across different topics, with the overall accuracy around 66%

## Significance

This system demonstrates how modern NLP techniques can be applied to automate the categorization of research papers, which has practical applications in:

Automatic tagging of research repositories
Recommendation systems for academic papers
Trend analysis in scientific research
Knowledge management in academic databases
The multi-label approach accurately reflects the interdisciplinary nature of modern research, where papers often span multiple fields of study.
