# Written in Python version 3.9
#%% Importing packages
import pandas as pd
from sklearn.feature_selection import chi2
import numpy as np

#%% Generating DF where document frequency based feature selection has been applied
# Input: result of generate_bow or generate_tfidf function and the number of features you want to keep
# Output: DF with PMID and relevance and top n features that occur in the highest number of documents
def generate_docfreqFS(df, num_features):
    copy_df = df.copy()
    copy_df.loc[len(copy_df)]= np.count_nonzero(copy_df, axis =0)  # add row with count of documents that contain feature
    transposed_df = copy_df.T # transpose
    transposed_df.iloc[0,-1] = 10000000 # set counter for PMID to high number so it will always remain highest when sorting values
    transposed_df.iloc[1,-1] = 10000000 # same for relevance
    ordered_words_df = transposed_df.sort_values(by=transposed_df.columns[-1], axis=0, ascending= False) # sort based on count
    featselected_df = ordered_words_df.iloc[:num_features+2,:-1] # select PMID + relevance + top n features
    final_df = featselected_df.T # transpose back to original

    return final_df

#%% Generating DF where chi-squared based feature selection has been applied
# Input: result of generate_bow or generate_tfidf function and the number of features you want to keep
# Output: DF with PMID and relevance and top n features that have the highest chi2 score
def generate_chisq_FS(df, num_features):
    features= df.iloc[:, 2:]  # define features
    outcome = df['relevance']  # and outcome for chi-square test
    chisquared = chi2(features, outcome)  # perform chi2 test: produces array with chi2 values and array with p-values
    chi2_df = pd.DataFrame(chisquared).T
    featurenames = pd.Series(df.columns[2:])  # extract feature names from initial df
    chi2_features_df = pd.concat([featurenames, chi2_df.iloc[:, 1]], axis=1)  # combine feature names & p-values
    chi2_features_df = chi2_features_df.sort_values(by=1, axis=0)  # sort from smallest to biggest p-value
    chi2_features_df = chi2_features_df.iloc[0:num_features, :]  # select only n features with smallest p-value
    selected_features = ['pmid', 'relevance'] + chi2_features_df.iloc[:, 0].tolist()  # make relevant features into list
    applied_fs_df = df[selected_features] # create version of original DF with only selected features

    return applied_fs_df

