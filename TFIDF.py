#%% Importing packages
import pandas as pd
import numpy as np
import re
import nltk
import math

#%% Importing data
root_dir = "C:/Users/marti/OneDrive/Thesis/Datasets/"
dataframe = pd.read_csv(filepath_or_buffer= root_dir + "Intermediate_output/NLP_dataset_with_prefixes.csv")
dataframe = dataframe.iloc[0:50, :]
bow_dataframe = pd.read_csv(filepath_or_buffer= root_dir + "Intermediate_output/BOW_applied_to_mini_dataset.csv")
#%% Function for making BOW into TFIDF
def generate_tfidf(tokenized_df, bow_df):

    # Create dataframe with term frequencies (tf): occurrences of word in doc / words in doc
    tf_df = bow_df.copy()
    for row in range(len(bow_df)):  # loop over tokenized df
        # length of tokenized title and abstract is required (not of BOW version)
        nwordsindoc = len(tokenized_df.iloc[row, 1].split(', ') + tokenized_df.iloc[row, 2].split(', '))
        for column in range(len(tf_df.columns)):  # now loop over bow_df
            if tf_df.columns[column].startswith(tuple(["ab", "ti"])):
                tf_df.iloc[row, column] = tf_df.iloc[row, column].astype(float)/nwordsindoc  # term frequency of a word in article

    # Create dataframe with inverse document frequencies (idf): inverse of number of documents containing the word relative to total number of docs
    idf_df = tf_df.copy()  # generating the count of documents containing the word
    ndocs = len(idf_df)  # initialize number of docs to later transform doc freq to inverse doc freq
    for column in range(len(idf_df.columns)):  # loop over columns (words)
        count = 0  # initialize number of documents (doc freq) containing the word
        if tf_df.columns[column].startswith(tuple(["ab", "ti"])):
            for row in range(len(idf_df)):  # loop over rows (documents)
                if idf_df.iloc[row, column] != 0:  # if word occurs at least once in this document (term_freq =/= 0)
                    count += 1  # counts the number of documents (rows) containing the word (column)
        idf_df.iloc[:, column] = math.log(ndocs/(count+1))  # idf with smoothing parameter, same for all documents (rows)

    # Create dataframe with term frequency-inverse document frequencies (tfidf)
    tfidf_df = tf_df.copy()
    for row in range(len(tfidf_df)):
        for column in range(len(tfidf_df.columns)):
            if tfidf_df.columns[column].startswith(tuple(["ab", "ti"])):
                # multiply term frequencies with document frequencies
                tfidf_df.iloc[row, column] = tf_df.iloc[row, column] * idf_df.iloc[row,column]

    return tfidf_df


tfidf_dataframe2 = generate_tfidf(dataframe, bow_dataframe)


#%%
