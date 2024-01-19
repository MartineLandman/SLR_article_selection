# Written in Python version 3.9
#%% Importing packages
import math

#%% Converting BOW representation to TFIDF representation
# Input: output from preprocess function (NLP_preprocessing) and from BOW function
# Output: DF with binary features for MeSH terms and publication types and weighted frequencies for words from the title
# and abstract (still incl. PMID and relevance)
def generate_tfidf(tokenized_df, bow_df):
    # Create DF with term frequencies (tf): occurrences of word in doc / words in doc
    tf_df = bow_df.copy()
    for row in range(len(bow_df)):  # loop over tokenized df
        # length of tokenized title and abstract is required (from preprocess function, not of BOW version)
        nwordsindoc = len(tokenized_df.iloc[row, 1].split(', ') + tokenized_df.iloc[row, 2].split(', '))
        for column in range(len(tf_df.columns)):  # now loop over bow_df
            if tf_df.columns[column].startswith(tuple(["ab", "ti"])):
                tf_df.iloc[row, column] = tf_df.iloc[row, column].astype(float)/nwordsindoc  # term frequency of a word in article

    # Create dataframe with inverse document frequencies (idf): inverse of number of documents containing the word
    # relative to total number of docs
    idf_df = tf_df.copy()
    ndocs = len(idf_df)  # initialize number of docs to transform doc freq to inverse doc freq in line 32
    for column in range(len(idf_df.columns)):  # loop over columns (words)
        if tf_df.columns[column].startswith(tuple(["ab", "ti"])): # we only apply tfidf to title and abstract
            count = 0  # initialize number of documents (doc freq) containing the word
            for row in range(len(idf_df)):  # loop over rows (documents)
                if idf_df.iloc[row, column] != 0:  # if word occurs at least once in this document (term_freq =/= 0)
                    count += 1  # doc freq: counts the number of documents (rows) containing the word (column)
            idf_df.iloc[:, column] = math.log(ndocs/count) # transform doc freq to idf

    # Create DF with term frequency-inverse document frequencies (tfidf) = tf * idf
    tfidf_df = tf_df.copy()
    for row in range(len(tfidf_df)):
        for column in range(len(tfidf_df.columns)):
            if tfidf_df.columns[column].startswith(tuple(["ab", "ti"])):
                # multiply term frequencies with document frequencies of t
                tfidf_df.iloc[row, column] = tf_df.iloc[row, column] * idf_df.iloc[row, column]

    return tfidf_df



