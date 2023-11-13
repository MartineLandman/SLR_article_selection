import gensim
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import csv
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
#%%
root_dir = "C:/Users/marti/OneDrive/Thesis/Datasets/"
df_no_preprocessing = pd.read_csv(filepath_or_buffer= root_dir + "Intermediate_output/final_dataset_right_MeSH_extended.csv")
df_no_preprocessing = df_no_preprocessing.iloc[0:50, :]
df_BOW = pd.read_csv(filepath_or_buffer= root_dir + "Intermediate_output/BOW_applied_to_mini_dataset.csv")
#%% The dataframe needs to be processed to train the word2vec model
# This differs from the BOW preprocessing, because for word2vec, sentences need to be lists of words
def preprocess_for_w2v (BOW_df, no_preprocessing_df):
    # selecting the title and abstract from the df that has not been preprocessed:
    tiabs_df = df_no_preprocessing[["pmid", "title", "abstract"]]
    # selecting all mesh and publication type columns (with binary values) from the BOW df:
    pubmesh_cols = [column for column in df_BOW.columns if column.startswith(tuple(['mesh', 'pt', 'pmid', 'relevance']))]
    pubmesh_df = BOW_df[pubmesh_cols]
    new_df = tiabs_df.merge(pubmesh_df, how="left", on='pmid')  # combining tiabs_df and pubmesh_df
    new_df["tiabs"] = new_df["title"] + " " + new_df["abstract"]  # combining title and abstract
    new_df.drop(["title", "abstract"], axis=1, inplace=True)

    stop_words = set(stopwords.words('english'))
    for row in range(len(new_df)):
        for column in range(len(new_df.columns)):
            if new_df.columns[column].startswith(tuple(['ti', 'abs'])):
                sentences = new_df.iloc[row, -1].split('. ')
                all_sentences = []  # to make a list of all tokenized sentences from one article
                for sentence in sentences:
                    tokenized_sentence = []  # to be able to later combine tokens into a sentence again
                    text = word_tokenize(sentence)  # tokenization
                    for token in text:  # removal of non-alphabetical characters from title
                        alphatoken = re.sub(r'[^a-zA-Z]', '', token)
                        if len(alphatoken) > 2:  # remove short and empty strings from title
                            lowertoken = alphatoken.lower()
                            if lowertoken not in stop_words:  # remove stopwords
                                tokenized_sentence.append(lowertoken)  # make tokens into a sentence again
                    all_sentences.append(tokenized_sentence)  # add to list of all tokenized sentences from one article
                new_df.iat[row, -1] = all_sentences  # overwrite 'tiabs' column with preprocessed tiabs
    return new_df

preprocessed_df = preprocess_for_w2v(df_BOW, df_no_preprocessing)
#%% Training of the word2vec model
def create_w2v_model (dataframe, vec_dimensions):
    # First, create the word2vec model, which needs a list of sentences (which are lists of words) to be trained
    sentences = []
    for row in range(len(dataframe)):
        for sentence in dataframe['tiabs'].iloc[row]:
            sentences.append(sentence)  # combine all sentences from the training data into one list of sentences
    w2v_model = Word2Vec(sentences, vector_size= vec_dimensions)
    return w2v_model


w2v_model = create_w2v_model(preprocessed_df, 100)

#%% Creating a df with vectorized tiabs (and binary mesh term and publication type values)
def vectorize(df, w2v_model, vec_dimensions):
    copy_df = df.copy()  # original df must be saved to indicate its number of columns in last line of this function
    empty_df = pd.DataFrame(columns=list(range(1,vec_dimensions+1)), index=list(range(len(copy_df)))).fillna(0)
    copy_df = pd.concat([copy_df, empty_df], axis=1)
    for row in range(len(copy_df)):  # loop over all pmid's
        all_vec = []  # create empty list to fill with vectors of all words in an article
        for sentence in df.iloc[row, -1]:  # for each sentence in an article:
            for word in sentence:  # for each word in a sentence:
                # (some words appear <5 times in corpus, so are not in w2v vocab):
                if word in w2v_model.wv.index_to_key:
                    new_vec = w2v_model.wv[word]  # create the word2vec vector of this word
                    all_vec.append(new_vec)  # add all vectors of all words in article to the list
        all_vec = np.array(all_vec)
        avg_vec = all_vec.mean(axis=0)  # create average vector of all words in article
        ncols_original_df = len(df.columns)
        for dimension_in_vec in range(len(avg_vec)):  # for dimension in average vector:
            copy_df.iloc[row, dimension_in_vec + len(df.columns)] = avg_vec[dimension_in_vec]  # add value to new_df2
    copy_df = copy_df.drop("tiabs", axis=1)
    return copy_df


df_vectorized = vectorize(preprocessed_df, w2v_model, 100)
#%%
print(w2v_model.wv.most_similar('glucose'))
print(w2v_model.wv.most_similar('insulin'))
#%%
