#%% Importing packages
import pandas as pd


#%% Importing data
root_dir = "C:/Users/marti/OneDrive/Thesis/Datasets/"
dataframe = pd.read_csv(filepath_or_buffer= root_dir + "Intermediate_output/NLP_dataset_with_prefixes.csv")
dataframe = dataframe.iloc[0:500, :]  # so there are also a few relevant ones
#%% Function to generate BOW
def generate_bow(df):
    vocab = []  # create vocabulary of all words in all articles
    for row in range(len(df)):
        for column in range(1, 5):  # loop over columns with title, abstract, mesh_terms and pubtype
            vocab.extend(df.iloc[row, column].split(", "))  # make one list of all these column titles
    vocabulary = sorted(set(vocab))  # only keep unique words in the vocabulary

    bow_df = pd.DataFrame(df.iloc[:, [0, -1]])  # initialize df with only pmid and relevance
    for word in vocabulary:  # creating a column for each word in the vocabulary
        word_df = pd.DataFrame(columns=[word])
        bow_df = pd.concat([bow_df, word_df], axis=1)
        bow_df = bow_df.fillna(0)

    for row in range(len(bow_df)):  # loop over all articles in bow_df
        words_in_article = df.iloc[row, 1].split(', ') + df.iloc[row, 2].split(', ') + df.iloc[row, 3].split(', ') + df.iloc[row, 4].split(', ')
        for column in range(2, len(bow_df.columns)):  # loop over all words in the bow vocabulary (skip relevance)
            for article_word in words_in_article:
                if bow_df.columns[column] == article_word:  # if the vocabulary word is in the article
                    if bow_df.columns[column].startswith(tuple(["ab", "ti"])):
                        bow_df.iloc[row, column] += 1  # add 1 to counter
                    else:
                        bow_df.iloc[row, column] = 1  # make binary value into 1
    return bow_df


bow_dataframe = generate_bow(dataframe)
#%%
#bow_dataframe.to_csv(path_or_buf= root_dit + "Intermediate_output/BOW_applied_to_mini_dataset.csv", index=False)
