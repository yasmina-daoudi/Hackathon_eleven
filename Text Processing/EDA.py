import numpy as np
import pandas as pd
import json
import spacy
import matplotlib.pyplot as plt
import datetime
from spacy_langdetect import LanguageDetector
import string
import re

# Special Libraries
import wordcloud
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import wordcloud

# Loading of the reviews file
file_path = "reviews_output.csv"

# Initialiazing a Language Detector Pipeline in case reviews not in English
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)

# Initialiazing the StopWords
stop_words = stopwords.words('english')

# Definition of Functions used in our pipeline


def extract_mostprobablelanguage(text):

    """This function is made to compute the language probability scores on
    a text, before returning the number 1 ranked language.
    We will use it on each raw of our column Reviews

    Args:
        text ([text]): [Raw text]

    Returns:
        [Dict]: [{'language': 'Number One Language', 'score': %}]
    """
    document = nlp(text)
    return document._.language['language']


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts):
    return [bigram_model[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_model[bigram_model[doc]] for doc in texts]


def lemmatization(bagofwords):
    # Use of Spacy for Lemmatization
    lemma_docu = nlp(" ".join(bagofwords))
    lemma_text = [token.text if '_' in token.text else token.lemma_ for token in lemma_docu if token.lemma_ != '-PRON-']
    return lemma_text


def nostopwordsandlemma(text):
    text = remove_stopwords(text)
    text = lemmatization(text)
    return text


if __name__ == '__main__':
    df_preprocessing = pd.read_csv(file_path)
    # First we compute the length of the reviews
    df_preprocessing['review_length'] = df_preprocessing['review_text'].map(lambda x: len(x.split()))

    # We will remove all the short reviews, most are spam
    df_valuablereviews = df_preprocessing[df_preprocessing['review_length'] >= np.percentile(df_preprocessing['review_length'], 0)].reset_index(drop=True)

    # Now we can remove the duplicates, we can consider them as not necessary either
    df_noduplicates = df_valuablereviews.drop_duplicates(subset=['review_text'])

    # We now want to know the language for each review. Might drop the quality of the date if
    # some reviews have multiple languages inside
    df_noduplicates['review_language'] = [extract_mostprobablelanguage(x) for x in df_noduplicates['review_text']]

    # Now we keep only the 'En' reviews and drop the multilingual ones
    df_english = df_noduplicates[df_noduplicates['review_language'] == 'en'].reset_index(drop=True)

    # We create a list that will store all the words of the reviews
    list_of_words = []
    list_of_words.extend([i for i in df_english['review_text']]

    # Taking inspiration from https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
    # We create bigram/trigram models
    bigram = gensim.models.Phrases(list(df_english['review_text']), min_count=5, threshold=10)
    trigram = gensim.models.Phrases(bigram[list(df_english['review_text'])], threshold=10)
    bigram_model = gensim.models.phrases.Phraser(bigram)
    trigram_model = gensim.models.phrases.Phraser(trigram)

    # We run these models through all our text
    df_english['2grams'] = make_bigrams(df_english['review_text'])
    df_english['3grams'] = make_trigrams(df_english['review_text'])

    # Now that we have bigrams and trigrams,
    # we remove the stopwords and we lemmatize
    df_english['2grams'] = df_english['2grams'].map(lambda x: nostopwordsandlemma(x))
    df_english['3grams'] = df_english['3grams'].map(lambda x: nostopwordsandlemma(x))

    df_english.to_csv('data_ready_LDA.csv')


# We can generate our results in the form of WordClouds
list_of_2grams = []
list_of_2grams.extend([i for i in df_english['2grams']])
list_of_3grams = []
list_of_3grams.extend([i for i in df_english['3grams']])

strings_of_2grams = ' '.join(g for g in list_of_2grams)
wordcloudimage_2grams = wordcloud.WordCloud(width = 1500, height=900, max_words = 80).generate(strings_of_2grams)
plt.figure(figsize=(27,20))
plt.imshow(wordcloudimage_2grams, interpolation='bilinear')
plt.axis("off")
plt.show()

strings_of_3grams = ' '.join(g for g in list_of_3grams)
wordcloudimage_3grams = wordcloud.WordCloud(width = 1500, height=900, max_words = 80).generate(strings_of_3grams)
plt.figure(figsize=(27,20))
plt.imshow(wordcloudimage_3grams, interpolation='bilinear')
plt.axis("off")
plt.show()

frequency_2grams = nltk.FreqDist(list_of_2grams)
plt.figure(figsize=(29,10))
frequency_2grams.plot(150,cumulative=False)

frequency_3grams = nltk.FreqDist(list_of_3grams)
plt.figure(figsize=(29,10))
frequency_3grams.plot(150,cumulative=False)