# Written in Python version 3.9
#%% Importing packages
import pandas as pd

#%% Generating BOW representation
# Input: DF with 6 columns: PMID, preprocessed title, preprocessed abstract, MeSH terms, publication type and relevance
# Output: DF (still incl. PMID and relevance)where all words from all preprocessed titles and abstracts and
# all MeSH terms and publication types have become separate columns. The value for each word from the title and abstract
# is the number of times it appears in the article (often 0) and MeSH terms and publication types are binary features
# indicating their absence or presence
def generate_bow(df):
    vocab = []  # create vocabulary of all words in all articles
    for row in range(len(df)):
        for column in range(1, 5):  # loop over columns with title, abstract, mesh_terms and publication types
            vocab.extend(df.iloc[row, column].split(", "))  # make one list of all these words across all articles
    vocabulary = sorted(set(vocab))  # only keep unique words in this vocabulary

    bow_df = pd.DataFrame(df.iloc[:, [0, -1]])  # initializing DF with only PMID and relevance
    for word in vocabulary:  # creating an empty column for each word in the vocabulary
        word_df = pd.DataFrame(columns=[word])
        bow_df = pd.concat([bow_df, word_df], axis=1)
        bow_df = bow_df.fillna(0)

    for row in range(len(bow_df)):  # loop over all articles in bow_df
        words_in_article = df.iloc[row, 1].split(', ') + df.iloc[row, 2].split(', ') + df.iloc[row, 3].split(', ') + df.iloc[row, 4].split(', ')
        for column in range(2, len(bow_df.columns)):  # loop over all words in the bow vocabulary (skip relevance)
            for article_word in words_in_article:
                if bow_df.columns[column] == article_word:  # if the vocabulary word is in the article:
                    if bow_df.columns[column].startswith(tuple(["ab", "ti"])): # in case of title and abstract
                        bow_df.iloc[row, column] += 1  # add 1 to counter
                    else: # in case of MeSH terms and publication types
                        bow_df.iloc[row, column] = 1  # make binary value into 1
    return bow_df

