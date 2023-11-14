import pandas as pd
from collections import Counter
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

#%% Importing data and setting features
root_dir = "..."
initial_df = pd.read_csv(filepath_or_buffer= root_dir + "Intermediate_output/BOW_applied_to_mini_dataset.csv")

#%% RUS: Random undersampling
def rus (df, seed, ratio):
    features = df.iloc[:, 2:]
    outcome = df['relevance']
    undersampling = RandomUnderSampler(sampling_strategy=ratio, random_state=seed, replacement= True)
    features_RUS, outcome_RUS = undersampling.fit_resample(features, outcome)
    # sampling strategy = n minority/ n majority -> 0.5 means minority has half the number of samples that majority has
    # so the majority class is reduced to 2x minority class by discarding random majority class instances
    rus_df = pd.concat([outcome_RUS, features_RUS], axis=1)

    return rus_df

df_rus = rus(initial_df, 567, 0.5)
#%% ROS: Random oversampling
def ros (df, seed, ratio):
    features = df.iloc[:, 2:]
    outcome = df['relevance']
    oversampling = RandomOverSampler(sampling_strategy=ratio, random_state=seed)
    features_ROS, outcome_ROS = oversampling.fit_resample(features, outcome)
    # 0.5 means that random instances from the minority class are copied to have half of the number in the majority class
    ros_df = pd.concat([outcome_ROS, features_ROS], axis=1)

    return ros_df

df_ros = ros(initial_df, 567, 0.5)

#%% SMOTE: Synthetic Minority Oversampling Technique
def smote (df, seed, ratio):
    features = df.iloc[:, 2:]
    outcome = df['relevance']
    smote = SMOTE(sampling_strategy=ratio, random_state=seed, k_neighbors =3) #AANPASSEN NAAR 5!!!!!!!!!!!!!!!!!!!!!!!!
    features_SMOTE, outcome_SMOTE = smote.fit_resample(features, outcome)
    # 0.5 means that synthetic samples from the minority class are created to have half of the number in the majority class
    SMOTE_df = pd.concat([outcome_SMOTE, features_SMOTE], axis=1)

    return SMOTE_df

df_SMOTE = smote(initial_df, 567, 0.5)

#%% RUS combined with SMOTE
def rus_smote (df, seed, rus_ratio, smote_ratio):
    features = df.iloc[:,2:]
    outcome = df['relevance']
    undersampling = RandomUnderSampler(sampling_strategy=rus_ratio, random_state=seed, replacement= True)
    features_RUS, outcome_RUS = undersampling.fit_resample(features, outcome)
    smote = SMOTE(sampling_strategy=smote_ratio, random_state=seed, k_neighbors =3)  # now apply smote to undersampled data
    features_SMOTE, outcome_SMOTE = smote.fit_resample(features_RUS, outcome_RUS)
    rus_smote_df = pd.concat([outcome_SMOTE, features_SMOTE], axis=1)

    return rus_smote_df

rus_smote_df = rus_smote(initial_df, 567, 0.5, 0.7)
#%%
