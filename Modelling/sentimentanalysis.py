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


def format_topics_sentences(df='dataframe', ldamodel='lda_model', corpus='corpus', documents='documents'):
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
    raw_contents = pd.DataFrame(df[['sentences', 'sentences_as_tokens']])
    docus = pd.Series(documents)
    sent_topics_df = pd.concat([sent_topics_df, docus, raw_contents], axis=1)
    return sent_topics_df


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def emphazizes_scores(listofscores):
    integerscores = [(-1 if x <-0.08 else (0 if -0.08 <= x <= 0.08 else 1)) for x in listofscores]
    return integerscores

def get_mode(lyst):
    our_counter = Counter(lyst)
    _,val = our_counter.most_common(1)[0]
    return [x for x,y in our_counter.items() if y == val]

nlp = spacy.load("en_core_web_sm")

def parsingsentencestokens(review):
    sentences_tokens = []
    if type(review)!=str :
        print(type(review))
        return ""
    text = [word for word in review.split() if len(word)>1 and len(word) <= 45]
    parsed_text = ' '.join(word for word in text)
    doc = nlp(parsed_text)
    for i,sent in enumerate(doc.sents):
        sentences_tokens.append(sent.text)
    return sentences_tokens

def remove_punc(sentences):
    cleaned_text = [s.translate(str.maketrans('', '', string.punctuation)) for s in sentences]
    return cleaned_text

def remove_stopwords(doc):
    words = [w for w in doc if w not in en_stopwords]
    return words

def lemmatizationspacy(bow,allowed_postags=['NOUN']):
    lemmat_docu = nlp(" ".join(bow)) 
    lemmat_text = [token.text if '_' in token.text else token.lemma_ if token.pos_ in allowed_postags else '' for token in lemmat_docu]
    return lemmat_text
    

Dictionary_of_our_topics = {'0.0' : 'Topic1', '1.0' : 'Topic2', '2.0' : 'Topic3', '3.0' : 'Topic4', '4.0' : 'Topic5',
                            '5.0' : 'Topic6', '6.0' : 'Topic7', '7.0' : 'Topic8', '8.0' : 'Topic9', '9.0' : 'Topic10', 
                            '10.0' : 'Topic11', '11.0' : 'Topic12', '12.0' : 'Topic13', '13.0' : 'Topic14', '14.0' : 'Topic15',
                            '15.0' : 'Topic16', '16.0' : 'Topic17', '17.0' : 'Topic18', '18.0' : 'Topic19', '19.0' : 'Topic20'}

def scoring_all_reviews(list_of_all_reviews,dataframe):
    score_topics = {Dictionary_of_our_topics['0.0']:[], Dictionary_of_our_topics['1.0']:[],
                    Dictionary_of_our_topics['2.0']:[], Dictionary_of_our_topics['3.0']:[],
                    Dictionary_of_our_topics['5.0']:[], Dictionary_of_our_topics['6.0']:[],
                    Dictionary_of_our_topics['7.0']:[], Dictionary_of_our_topics['8.0']:[],
                    Dictionary_of_our_topics['9.0']:[], Dictionary_of_our_topics['10.0']:[],
                    Dictionary_of_our_topics['11.0']:[], Dictionary_of_our_topics['12.0']:[],
                    Dictionary_of_our_topics['13.0']:[], Dictionary_of_our_topics['14.0']:[],
                    Dictionary_of_our_topics['15.0']:[], Dictionary_of_our_topics['15.0']:[],
                    Dictionary_of_our_topics['16.0']:[], Dictionary_of_our_topics['17.0']:[],
                    Dictionary_of_our_topics['18.0']:[], Dictionary_of_our_topics['19.0']:[]}
    ultimate_scores = {}

    for i in list_of_all_reviews:
        review_identifier_bool = dataframe['index_review'] == i
        review_considerated = dataframe[review_identifier_bool]

        for key in Dictionary_of_our_topics:
            dataframe_topics = review_considerated[review_considerated['main_topic'] == Dictionary_of_our_topics[key]] 

            if len(dataframe_topics) == 0:
                pass

            else:
                list_of_the_scores = list(dataframe_topics['integer_scores'])
                list_of_modes = get_mode(list_of_the_scores)

                if len(list_of_modes) ==1:
                    sentiment = list_of_modes[0]

                elif len(list_of_modes) >1:
                    sentiment = 0
                score_topics[Dictionary_of_our_topics[key]].append(sentiment)

    for key in score_topics:
        list_all_scores = score_topics[key]
        positive_topics = list_all_scores.count(1)/len(list_all_scores)
        neutral_topics = list_all_scores.count(0)/len(list_all_scores)
        negative_topics = list_all_scores.count(-1)/len(list_all_scores)
        aggregation_scores_topics = [round(positive_topics,3),round(neutral_topics,3),round(negative_topics,3)]
        ultimate_scores[key] = aggregation_scores_topics

    return ultimate_scores


if __name__ == '__main__':
    ultimate_df = pd.read_csv('../Hackathon_eleven/Modelling/Models/topics_tokens_sktrax.csv',index_col=0)
    LDA_model = LdaMulticore.load('../Hackathon_eleven/Modelling/Models/model_saved.model')
    # We store all the sentences and an index number to reconciliate scores by topic and review later, and facilitate aggregation for scores
    sentences_list = []
    reviews_indices = []
    df_sentences = pd.DataFrame()
    review_count = 0
    for review in ultimate_df['text']:
        sentences_tokens = parsingsentencestokens(review)
        sentences_tokens = remove_punc(sentences_tokens)
        sentences_list.extend(sentences_tokens)
        review_count+=1
        review_num_lst = [str(review_count)] * len(sentences_tokens)
        reviews_indices.extend(review_num_lst)
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
    df_sentences['3grams'] = df_sentences['3grams'].map(lambda x: lemmatizationspacy(x))
    df_sentences['number_of_tokens'] = df_sentences['3grams'].map(lambda x: len(x))
    # We remove tokens smaller than 1
    last_df = df_sentences[df_sentences['number_of_tokens']>1].reset_index(drop=True)
    # We load the parameters to run again on this datafram the format_topic_sentences function built upon our LDA
    docu3grams = list(last_df['3grams'])
    dict3grams = gensim.corpora.Dictionary(docu3grams)
    dict3grams.filter_extremes(no_below = 5,no_above=0.5)
    corpus3grams = [dict3grams.doc2bow(text) for text in docu3grams]
    df_topics_sentences_tokens_in = format_topics_sentences(df=last_df,ldamodel=LDA_model, corpus=corpus3grams, documents=docu3grams)
    df_topics_sentences_tokens = df_topics_sentences_tokens_in.reset_index()
    # We clean and add some infos on the tokens
    df_topics_sentences_tokens.columns = ['index', 'main_topic', 'percent_contribution_of_topic', 'keywords', 'tokens','text','sentence_tokens']
    df_topics_sentences_tokens['number_tokens']=last_df['number_of_tokens']
    df_topics_sentences_tokens['index_review']=last_df['index_review'].map(lambda x: int(x))
    df_topics_sentences_tokens = df_topics_sentences_tokens[['index_review','main_topic','percent_contribution_of_topic','tokens','sentence_tokens','number_tokens','keywords','text']]
    # Finallyyyyy some sentiment analysis !
    SentimentOhYeah = SentimentIntensityAnalyzer()
    df_topics_sentences_tokens['compound_score'] = df_topics_sentences_tokens['text'].map(lambda x: SentimentOhYeah.polarity_scores(x)['compound'])
    # Saving this
    df_topics_sentences_tokens.to_csv('../Hackathon_eleven/Recommendations/aggregation_sentences_compound_scores_SkyTrax.csv')

    # Some work on the scores (TO BE ADAPTED ACCORDING TO OUR DATASET)
    # If we want to avoid central measures such as mode or median to give too neutral reviews,
    # we have to emphazize extremes.
    # Because we don't care about the scores themselves : we need to know if a review is
    # positive, netural, or negative.
    df_topics_sentences_tokens['integer_scores'] = emphazizes_scores(df_topics_sentences_tokens['compound_score'])
    # VERY IMPORTANT : IF YOU CHANGED THE NUMBER OF TOPICS TO MORE THAN 5, PLEASE MODIFY
    # THE scoring_all_reviews and modify accordingly 'Dictionary_of_our_topics' and
    # 'score_topics'
    list_of_all_reviews = list(df_topics_sentences_tokens['index_review'].unique())
    # Passing on the function that will score all reviews
    dictionary_with_sentiment_scores_topics = scoring_all_reviews(list_of_all_reviews,df_topics_sentences_tokens)
    # Vizualisation
    final_result_sentiment_topics = pd.DataFrame(dictionary_with_sentiment_scores_topics).T
    final_result_sentiment_topics.columns = ['Positive Score','Neutral Score','Negative Score']
    print(final_result_sentiment_topics)
    # If someone could add plots
