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
import contractions
import en_core_web_sm


# Loading of the reviews file
file_path = "../Hackathon_eleven/Web Scrapping/All_reviews_merged.csv"


# Initialiazing a Language Detector Pipeline in case reviews not in English
nlp = en_core_web_sm.load()
nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)

# Initialiazing the StopWords
stop_stopwords = [s for s in set(STOPWORDS)]
stop_stopwords.extend(['i', 'am', 'we', 'like', 'the','they', 'good', 'were', 'airline', '_', 'th', 'aircraft', 'swiss',
        'did', 'aeroflot', 'emirates', 'hop', 'iberia', 'klm', 'lufthansa', 'ryanair', "air", "airfrance", "france", "british", 
        "airways", "cdg", 'intl', 'virgin', 'atlantic', 'transavia', 'easyjet','vueling', 'southwest', 'united', 'airport',
        'paris', 'orly', 'london', 'heathrow', 'mexico', 'airline', 'qatar', 'airways', 'tel', 'aviv', 'jet', 'honk', 'kong', 
        'flight', 'plane', 'flights', 'airlines', 'fly', 'way', 'flew', 'ba', 'passengers', 'it', 'use', 'singapore', 'dubai', 'me', 'you', ''])
stop_stopwords = [s for s in set(stop_stopwords)]

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
    return [word for word in simple_preprocess(str(texts)) if word not in stop_stopwords]


def make_bigrams(texts):
    return [bigram_model[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_model[bigram_model[doc]] for doc in texts]


def lemmatization(bagofwords, allowed_postags=['NOUN']):
    # Use of Spacy for Lemmatization
    lemma_docu = nlp(" ".join(i for i in bagofwords))
    lemma_text = [token.text if '_' in token.text else token.lemma_ if token.pos_ in allowed_postags else '' for token in lemma_docu]
    return lemma_text


def nostopwordsandlemma(texts):
    texts = remove_stopwords(texts)
    texts = lemmatization(texts)
    texts = [text for text in texts if len(text)>0]
    return texts


if __name__ == '__main__':
    df_preprocessing = pd.read_csv(file_path, sep='|', index_col=0, encoding="ISO-8859-1").reset_index()

    # First we compute the length of the reviews
    df_preprocessing['review_length'] = df_preprocessing['review'].map(lambda x: len(x.split()))

    # We will remove all the short reviews, most are spam
    df_valuablereviews = df_preprocessing[df_preprocessing['review_length'] >= np.percentile(df_preprocessing['review_length'], 0)].reset_index(drop=True)

    # Now we can remove the duplicates, we can consider them as not necessary either
    df_noduplicates = df_valuablereviews.drop_duplicates(subset=['review'])

    # We now want to know the language for each review. Might drop the quality of the date if
    # some reviews have multiple languages inside
    df_noduplicates['review_language'] = [extract_mostprobablelanguage(x) for x in df_noduplicates['review']]

    # Now we keep only the 'En' reviews and drop the multilingual ones, and expand the contracted words like "I'm"
    df_english = df_noduplicates[df_noduplicates['review_language'] == 'en'].reset_index(drop=True)
    df_english['review'] = df_english['review'].map(lambda x: contractions.fix(x))

    # We create a list that will store all the words of the reviews
    beg_list_of_words = []
    beg_list_of_words.extend([text.lower().split() for text in df_english['review']])

    #remove punctuations from words and strip leading & trailing whitespaces
    list_of_words_nopunc = [[re.sub(r'[^\w\s]', ' ', word).strip() for word in sublist] for sublist in beg_list_of_words]

    #lemmatise words
    #list_of_words_nopunc_lemma = [[word.text if '_' in word.text else word.lemma_ if word.pos_ == 'NOUN' else '' for word in nlp(" ".join(i for i in sublist))] for sublist in list_of_words_nopunc]
    list_of_words_nopunc_lemma = [[word.text if '_' in word.text or 'PRON' in word.lemma_  else word.lemma_ for word in nlp(" ".join(i for i in sublist))] for sublist in list_of_words_nopunc]


    #remove stopwords
    list_of_words = [[word for word in sublist if word not in stop_stopwords] for sublist in list_of_words_nopunc_lemma]
    #print(list_of_words[4930])
    # Taking inspiration from https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
    # We create bigram/trigram models

    bigram = gensim.models.Phrases(list_of_words, min_count=5, threshold=10)
    trigram = gensim.models.Phrases(bigram[list_of_words], threshold=10)
    bigram_model = gensim.models.phrases.Phraser(bigram)
    trigram_model = gensim.models.phrases.Phraser(trigram)

    # We run these models through all our text
    df_english['2grams'] = [bigram_model[list_of_words[i]] for i in range(len(list_of_words))]
    df_english['3grams'] = [trigram_model[bigram_model[list_of_words[i]]] for i in range(len(list_of_words))]

    df_english.to_csv('../Hackathon_eleven/Text Processing/data_ready_LDA_final.csv')


# We can generate our results in the form of WordClouds
list_of_2grams = []
list_of_2grams.extend([i for i in df_english['2grams']])
flat_list_of_2grams = [item for sublist in list_of_2grams for item in sublist]

list_of_3grams = []
list_of_3grams.extend([i for i in df_english['3grams']])
flat_list_of_3grams = [item for sublist in list_of_3grams for item in sublist]

strings_of_2grams = ' '.join(g for g in flat_list_of_2grams)
wordcloudimage_2grams = wordcloud.WordCloud(width = 1500, height=900, max_words = 80).generate(strings_of_2grams)
plt.figure(figsize=(27,20))
plt.imshow(wordcloudimage_2grams, interpolation='bilinear')
plt.axis("off")
plt.show()

strings_of_3grams = ' '.join(g for g in flat_list_of_3grams)
wordcloudimage_3grams = wordcloud.WordCloud(width = 1500, height=900, max_words = 80).generate(strings_of_3grams)
plt.figure(figsize=(27,20))
plt.imshow(wordcloudimage_3grams, interpolation='bilinear')
plt.axis("off")
plt.show()

frequency_2grams = nltk.FreqDist(flat_list_of_2grams)
plt.figure(figsize=(29,10))
frequency_2grams.plot(150,cumulative=False)

frequency_3grams = nltk.FreqDist(flat_list_of_3grams)
plt.figure(figsize=(29,10))
frequency_3grams.plot(150,cumulative=False)
