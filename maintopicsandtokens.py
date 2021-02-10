# This must be run AFTER topicmodelinglda.py

import string
import pandas as pd
import gensim
from gensim.models import LdaMulticore
from gensim.parsing.preprocessing import STOPWORDS
from gensim.test.utils import datapath
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel, LdaModel


# Definition of function to get the main topics : source https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

def format_topics_sentences(ldamodel, corpus, documents):
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):

            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]),
                    ignore_index=True,
                )
            else:
                break

    sent_topics_df.columns = ["Dominant_Topic", "Perc_Contribution", "Topic_Keywords"]

    # Add original text to the end of the output
    raw_contents = pd.Series(df_for_sentiment["review_text"])
    docus = pd.Series(documents)
    sent_topics_df = pd.concat([sent_topics_df, docus, raw_contents], axis=1)
    return sent_topics_df


if __name__ == "__main__":
    df_for_sentiment = pd.read_csv(
        "../Hackathon_eleven/Modelling/Models/df_for_sentiment.csv", index_col=0
    )
    # We load the model
    LDA_model = LdaMulticore.load(
        "../Hackathon_eleven/Modelling/Models/model_saved.model"
    )
    # We only keep 2 columns
    df_for_sentiment = df_for_sentiment[["review_text", "3grams"]]
    # Some formatting : the strings of each cell into list again
    df_for_sentiment["review_text"] = df_for_sentiment["review_text"].map(
        lambda x: "".join(
            c for c in x if c == "_" or c not in string.punctuation
        ).split()
    )
    df_for_sentiment["3grams"] = df_for_sentiment["3grams"].map(
        lambda x: "".join(
            c for c in x if c == "_" or c not in string.punctuation
        ).split()
    )
    # Reloading parameters for LDA
    docs3grams = list(df_for_sentiment["3grams"])
    dict3grams = gensim.corpora.Dictionary(docs3grams)
    dict3_grams_filtered = gensim.corpora.Dictionary.load(
        "../Hackathon_eleven/Modelling/Models/model_saved.model.id2word"
    )
    corpus3grams = [dict3_grams_filtered.doc2bow(text) for text in docs3grams]
    # We call the function that will create the DF with the main topics
    df_topics = format_topics_sentences(
        ldamodel=LDA_model, corpus=corpus3grams, documents=docs3grams
    )
    df_topics_tokens = df_topics.reset_index()
    # We clean and add some infos on the tokens
    df_topics_tokens.columns = [
        "index",
        "main_topic",
        "percent_contribution_of_topic",
        "keywords",
        "tokens",
        "text",
    ]
    df_topics_tokens["number_tokens"] = df_topics_tokens["tokens"].map(
        lambda x: len(x) if hasattr(x, "__len__") else 0
    )
    df_topics_tokens["unique_tokens"] = df_topics_tokens["tokens"].map(
        lambda x: list(set(x)) if hasattr(x, "__iter__") else []
    )
    df_topics_tokens["number_unique_tokens"] = df_topics_tokens["unique_tokens"].map(
        lambda x: len(x) if hasattr(x, "__len__") else 0
    )
    df_topics_tokens = df_topics_tokens[
        [
            "main_topic",
            "text",
            "percent_contribution_of_topic",
            "tokens",
            "number_tokens",
            "unique_tokens",
            "number_unique_tokens",
            "keywords",
        ]
    ]
    # We save the Dataframe
    df_topics_tokens.to_csv("../Hackathon_eleven/Modelling/Models/topics_tokens.csv")
