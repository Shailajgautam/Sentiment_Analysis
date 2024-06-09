---

# Sentiment Analysis of Tweets using Bidirectional GRU

This project implements a sentiment analysis model using Bidirectional GRU to classify tweets as positive or negative. The dataset consists of tweets labeled with sentiment scores, and the model is trained to predict these scores.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Data Preparation](#data-preparation)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Tweet Classification](#tweet-classification)
- [Visualizations](#visualizations)
- [How to Run](#how-to-run)

## Introduction

Sentiment analysis is a natural language processing (NLP) technique used to determine whether data is positive, negative, or neutral. This project focuses on classifying tweets into positive or negative sentiment using a Bidirectional GRU model implemented in TensorFlow.

## Dependencies

Ensure you have the following dependencies installed:
- pandas
- numpy
- tensorflow
- matplotlib
- seaborn
- re
- wordcloud
- sklearn

You can install these dependencies using pip:

```bash
pip install pandas numpy tensorflow matplotlib seaborn wordcloud scikit-learn
```

## Data Preparation

### Loading and Cleaning the Data

The datasets are loaded and cleaned by removing unnecessary columns and mapping target values to binary classes (0 for negative, 1 for positive).

### Cleaning the Text

A custom function `clean_text` is defined to preprocess the text data by removing URLs, mentions, hashtags, numbers, and punctuation.

### Tokenizing and Padding Sequences

Tokenization and padding are performed to ensure uniform sequence length for the input to the model.

## Model Building

A Bidirectional GRU model is built with embedding, dropout, and batch normalization layers.

## Model Evaluation

The model is evaluated using test data and metrics like accuracy, precision, recall, and F1-score are calculated.

## Tweet Classification

A function is defined to classify new tweets as positive or negative.


## Visualizations

### Word Clouds

Word clouds are generated to visualize the most frequent words in positive and negative tweets.

### Training History

The training and validation accuracy and loss values are plotted to visualize the model's performance over epochs.

### Confusion Matrix

The confusion matrix is plotted to visualize the performance of the model in classifying tweets.

## How to Run

1. Ensure you have all dependencies installed.
2. Clone this Repo.
3. Download the dataset files and place them in the same directory as the script.
4. Run the script to preprocess the data, build the model, and evaluate its performance.
5. Use the `classify_tweet` function to classify new tweets.
