# Written in Python version 3.9
#%% Importing packages
import pandas as pd
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

#%% RUS: Random undersampling
# Input: result of generate_docfreqFS or generate_chisq_FS and minority/majority ratio
# Output: DF (excl. PMID) where a number of observations of the majority class has been discarded
def generate_rus (df, ratio):
    features = df.iloc[:, 2:]
    outcome = df['relevance']
    undersampling = RandomUnderSampler(sampling_strategy=ratio, replacement= True)
    features_RUS, outcome_RUS = undersampling.fit_resample(features, outcome)
    # sampling strategy = n minority/ n majority -> 0.5 means minority has half the number of samples that majority has
    # so the majority class is reduced to 2x minority class by discarding random majority class instances
    rus_df = pd.concat([outcome_RUS, features_RUS], axis=1)


    return rus_df
#%% ROS: Random oversampling
# Input: result of generate_docfreqFS or generate_chisq_FS and minority/majority ratio
# Output: DF (excl. PMID) where a number of observations of the minority class has been copied
def generate_ros (df, ratio):
    features = df.iloc[:, 2:]
    outcome = df['relevance']
    oversampling = RandomOverSampler(sampling_strategy=ratio)
    features_ROS, outcome_ROS = oversampling.fit_resample(features, outcome)
    # 0.5 means that random instances from the minority class are copied to have half of the number in the majority class
    ros_df = pd.concat([outcome_ROS, features_ROS], axis=1)

    return ros_df
#%% SMOTE: Synthetic Minority Oversampling Technique
# Input: result of generate_docfreqFS or generate_chisq_FS and minority/majority ratio
# Output: DF (excl. PMID) where a number of synthetic observations of the majority class has been added
def generate_smote (df, ratio):
    features = df.iloc[:, 2:]
    outcome = df['relevance']
    smote = SMOTE(sampling_strategy=ratio,  k_neighbors=5)
    features_SMOTE, outcome_SMOTE = smote.fit_resample(features, outcome)
    # 0.5 means that synthetic samples from the minority class are created to have half of the number in the majority class
    SMOTE_df = pd.concat([outcome_SMOTE, features_SMOTE], axis=1)

    return SMOTE_df