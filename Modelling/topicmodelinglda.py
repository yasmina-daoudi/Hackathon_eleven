import pandas as pd
import string
import matplotlib.pyplot as plt

import gensim
from gensim.test.utils import datapath
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel, LdaModel, LdaMulticore

# For vizualisation of the topics-keywords
import pyLDAvis
import pyLDAvis.gensim

import os
import pathlib

# We won't use TF-IDF as "LDA is a probabilistic model that tries to estimate probability distributions for topics in documents and words in topics."

# Function for finding the optimal number of topics for LDA


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, workers=3)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


if __name__ == '__main__':
    df_LDA = pd.read_csv('../Hackathon_eleven/Text Processing/data_ready_LDA_skytrax.csv', index_col=0)
    # We only keep the 3 columns
    df_LDA = df_LDA[['review', '2grams', '3grams']]
    # Some formatting : the strings of each cell into list again
    df_LDA['review_text'] = df_LDA['review_text'].map(lambda x: ''.join(c for c in x if c == '_' or c not in string.punctuation).split())
    df_LDA['2grams'] = df_LDA['2grams'].map(lambda x: ''.join(c for c in x if c == '_' or c not in string.punctuation).split())
    df_LDA['3grams'] = df_LDA['3grams'].map(lambda x: ''.join(c for c in x if c == '_' or c not in string.punctuation).split())
    # LDA Model
    # First we build our 3gram data : a dict and the corpus
    docs3grams = list(df_LDA['3grams'])
    dict3grams = gensim.corpora.Dictionary(docs3grams)
    dict3grams.filter_extremes(no_below=4, no_above=0.4)
    corpus3grams = [dict3grams.doc2bow(w) for w in docs3grams]
    # Parameters of our model
    num_topics = 20
    passes = 100
    eval_every = None
    # Call to the model
    LDAmodel = gensim.models.ldamulticore.LdaMulticore(corpus3grams, num_topics=num_topics, id2word=dict3grams, passes=passes, alpha='asymmetric', eval_every=eval_every)
    # The topics are...
    listoftopics = LDAmodel.print_topics(num_topics=num_topics, num_words=15)
    for i, element in enumerate(listoftopics):
        first_string = str(element[1])
        for char in "0123456789+*\".":
            first_string = first_string.replace(char, "")
        first_string = first_string.replace("  ", " ")
        print(first_string)
    # We compute the Perplexity and Coherence of our Model
    # Compute Perplexity : lower = good model
    print('\nPerplexity: ', LDAmodel.log_perplexity(corpus3grams))
    # Compute Coherence Score : 1 is best
    coherence_model_LDA = CoherenceModel(model=LDAmodel, texts=docs3grams, dictionary=dict3grams, coherence='c_v')
    coherence_of_our_LDA = coherence_model_LDA.get_coherence()
    print('\nCoherence Score: ', coherence_of_our_LDA)
    # If we want to visualize the topics-keywords
    # pyLDAvis.enable_notebook() only if you run it inside a Jupyter
    vizualisation_topics = pyLDAvis.gensim.prepare(LDAmodel, corpus3grams, dict3grams)
    pathviz = '../Hackathon_eleven/Modelling/Vizualisation'
    if not os.path.exists(pathviz):
        os.makedirs(pathviz)
    pyLDAvis.save_html(vizualisation_topics, '../Hackathon_eleven/Modelling/Vizualisation/LDA_Model_Skytrax.html')

    # Tuning our LDA
    limit = 80
    start = 2
    step = 6
    model_list, coherence_values = compute_coherence_values(dictionary=dict3grams, corpus=corpus3grams, texts=list(df_LDA['3grams']), start=start, limit=limit, step=step)
    # Print the coherence scores & plot coherence against number of topics
    for m, cv in zip(range(start, limit, step), coherence_values):
        print("Number of Topics =", m, " has Coherence Value of", round(cv, 4))
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

    # Saving the model
    pathmodel = '../Hackathon_eleven/Modelling/Models'
    if not os.path.exists(pathmodel):
        os.makedirs(pathmodel)
    LDAmodel.save('../Hackathon_eleven/Modelling/Models/model_saved.model')

    # Saving the Dataframe
    df_LDA.to_csv('../Hackathon_eleven/Modelling/Models/df_for_sentiment_skytrax.csv')
