#%% importing packages
import pandas as pd
from sklearn.feature_selection import chi2
#%%
root_dir = "..."
initial_df = pd.read_csv(filepath_or_buffer= root_dir + "Intermediate_output/BOW_applied_to_mini_dataset.csv")

#%% TERM FREQUENCY BASED FEATURE SELECTION
# How many times does the word occur in entire corpus?
def generate_tf_based_FS(df, num_features):
    tf_based_fs_df = df.copy()  # create df to store total count of each word
    for column in range(2, len(df.columns)):
        count = 0
        for row in range(len(df)):
            count += df.iloc[row, column]  # sum occurrences of word across all articles in original df
        tf_based_fs_df.iloc[:, column] = count  # fill in this sum in the tf_based_fs_df
    only_words_df = tf_based_fs_df.iloc[:,2:].sort_values(by=0, axis=1, ascending=False)  # order the columns(counts)
    selected_words_df = only_words_df.iloc[:, :num_features+2]  # select n most often occurring words/columns (+ pmid & relevance)
    selected_features = ['pmid', 'relevance'] + [col for col in selected_words_df.columns]  # make list of all selected features (+ pmid & relevance)
    applied_fs_df = df[selected_features]  # only keep the selected features (+ pmid & relevance)
    return applied_fs_df


tf_based_FS_df = generate_tf_based_FS(initial_df, 500)
#%% DOCUMENT FREQUENCY BASED FEATURE SELECTION
# How many articles contain the word?
def generate_docfr_based_FS(df, num_features):
    docfr_based_fs_df = df.copy()  # create df to store total count of each word
    for column in range(2, len(docfr_based_fs_df.columns)):
        count = 0
        for row in range(len(docfr_based_fs_df)):
            if docfr_based_fs_df.iloc[row, column] != 0:  # this means word occurs at least once in this article
                count += 1  # counting number of articles containing word (at least once)
        docfr_based_fs_df.iloc[:, column] = count
    only_words_df = docfr_based_fs_df.iloc[:, 2:].sort_values(by=0, axis=1, ascending=False)  # order columns(counts)
    selected_words_df = only_words_df.iloc[:, :num_features]  # select n most often occurring words/columns
    selected_features = ['pmid', 'relevance'] + [col for col in selected_words_df.columns]  # make list of all selected features (+ pmid & relevance)
    applied_fs_df = df[selected_features]  # only keep the selected features (+ pmid & relevance)
    return applied_fs_df


df_based_FS_df = generate_docfr_based_FS(initial_df, 500)

#%% CHI SQUARE
def chisq_FS(df, num_features):
    features= initial_df.iloc[:, 2:]  # define features(x)
    outcome = initial_df['relevance']  # and outcome/label(y) for chi-square test
    chisquared = chi2(features, outcome)  # perform chi2 test: produces array with chi2 values and array with p-values
    chi2_df = pd.DataFrame(chisquared).T
    featurenames = pd.Series(initial_df.columns[2:])  # extract feature names from initial df
    chi2_features_df = pd.concat([featurenames, chi2_df.iloc[:, 1]], axis=1)  # combine feature names & p-values
    chi2_features_df = chi2_features_df.sort_values(by=1, axis=0)  # sort from smallest to biggest p-value
    chi2_features_df = chi2_features_df.iloc[0:num_features, :]  # select only n features with smallest p-value
    selected_features = ['pmid', 'relevance'] + chi2_features_df.iloc[:, 0].tolist()  # make relevant features into list
    applied_fs_df = df[selected_features]

    return applied_fs_df


chisq_based_FS_df = chisq_FS(initial_df, 500)
