import re
import string
import spacy
from spacy.tokenizer import Tokenizer
import nltk
from nltk import WordNetLemmatizer

import gensim
from gensim.models import LdaMulticore
from gensim.parsing.preprocessing import STOPWORDS
from gensim.test.utils import datapath
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel, LdaModel, LdaMulticore


# We use the Vader Library which returns 4 values :
# pos: The probability of the sentiment to be positive
# neu: The probability of the sentiment to be neutral
# neg: The probability of the sentiment to be negative
# compound: The normalized compound score which calculates the sum of all lexicon ratings and takes values from -1 to 1
# The compound score is very useful : we want a single measure of sentiment.
# Typical threshold values are the following:
# compound score>=0.05, positive
# compound score between -0.05 and 0.05, neutral
# compound score<=-0.05


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
sns.set()

# Reviews are broken into sentences with SpaCy
# Extraction of main topic per sentence
# Sentiment scoring per sentence
# Aggregate senteces and scores by topic, for EACH review
# Then aggregate scores of all reviews by topic >> by type of sentiment


def format_topics_sentences(dataframe, ldamodel=lda_model, corpus=corpus, documents=documents):
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):

            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic, 4),  topic_keywords]), ignore_index=True)
            else:
                break

    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    raw_contents = pd.Series(dataframe[['sentences', 'sentences_as_tokens'])
    docus = pd.Series(documents)
    sent_topics_df = pd.concat([sent_topics_df, docus, raw_contents], axis=1)
    return sent_topics_df


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def parsingsentencestokens(review):
    sent_tokens = []
    if type(review)!=str :
        print(type(review))
        return ""
    text = expandContractions(review)
    text = [word for word in text.split() if len(word)>1 and len(word) <= 45]
    parsed_text = ' '.join(word for word in text)
    doc = nlp(parsed_text)
    for i,sent in enumerate(doc.sents):
        sent_tokens.append(sent.text)
    return sentences_tokens

def remove_punc(sentences):
    clean_text = [s.translate(str.maketrans('', '', string.punctuation)) for s in sentences]
    return cleaned_text



if __name__ == '__main__':
    ultimate_df = pd.read_csv('../Hackathon_eleven/Modelling/Models/topics_tokens.csv',index_col=0)
    LDA_model = LdaMulticore.load('../Hackathon_eleven/Modelling/Models/model_saved.model')
    # We store all the sentences and an index number to reconciliate scores by topic and review later, and facilitate aggregation for scores
    sentences_list = []
    reviews_indices = []
    df_sentences = pd.DataFrame()
    review_count = 0
    for review in ultimate_df['text']:
        sentences_tokens = parse_sent_tokenize(review)
        sentences_tokens = remove_punc(sentences_tokens)
        big_sent_lst2.extend(sentences_tokens)
        review_count+=1
        review_num_lst = [str(review_count)] * len(sentences_tokens)
        review_index_lst.extend(review_num_lst)
    df_sentences['sentences'] = sentences_list
    df_sentences['index_review'] = reviews_indices
    # We proceed to tokenization
    df_sentences['sentences_as_tokens'] = [x.split() for x in sentences_list]
    en_stopwords = list(set(STOPWORDS))
    en_stopwords.extend(['I','i', 'like', 'good', 'great', 'love'])
    df_sentences['sentokens_no_stopwords'] = df_sentences['sentences_as_tokens'].map(lambda x: remove_stopwords(x))
    bigram = gensim.models.Phrases(list(df_sentences['sentokens_no_stopwords']), min_count=5, threshold=10)
    trigram = gensim.models.Phrases(bigram[list(df_sentences['sentokens_no_stopwords'])], threshold=10)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    df_sentences['3grams'] = make_trigrams(df_sentences['sentokens_no_stopwords'])
    df_sentences['3grams'] = df_sentences['3grams'].map(lambda x: spacy_lemma(x))
    df_sentences['number_of_tokens'] = df_sentences['3grams'].map(lambda x: len(x))
    # We remove tokens smaller than 1
    last_df = df_sentences[df_sentences['number_of_tokens']>1].reset_index(drop=True)
    # We load the parameters to run again on this datafram the format_topic_sentences function built upon our LDA
    docu3grams = list(last_df['3grams'])
    dict3grams = gensim.corpora.Dictionary(docu3grams)
    dict3grams.filter_extremes(no_below = 5,no_above=0.5)
    corpus3grams = [dict3grams.doc2bow(text) for text in docu3grams]
    df_topics_sentences_tokens_in = format_topics_sentences(last_df,ldamodel=LDA_model, corpus=corpus3grams, documents=docu3grams)
    df_topics_sentences_tokens = df_topics_sentences_tokens_in.reset_index()
    # We clean and add some infos on the tokens
    df_topics_sentences_tokens.columns = ['index', 'main_topic', 'percent_contribution_of_topic', 'keywords', 'tokens','text','sentence_tokens']
    df_topics_sentences_tokens['number_tokens']=last_df['number_of_tokens']
    df_topics_sentences_tokens['index_review']=last_df['index_review'].map(lambda x: int(x))
    df_topics_sentences_tokens = df_topics_sentences_tokens[['index_review','main_topic','topic_perc_contrib','tokens','sentence_tokens','number_tokens','keywords','text']]
    # Finallyyyyy some sentiment analysis !
    df_topics_sentences_tokens['compound_score'] = df_topics_sentences_tokens['text'].map(lambda x: anakin.polarity_scores(x)['compound'])
    # Saving this
    df_topics_sentences_tokens.to_csv('../Hackathon_eleven/Recommendations/aggregation_sentences_compound_scores.csv')
