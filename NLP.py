#%% Importing packages

import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')


#%% Importing data
root_dir = "C:/Users/marti/OneDrive/Thesis/Datasets/"
df_articles = pd.read_csv(filepath_or_buffer= root_dir + "Intermediate_output/final_dataset_right_MeSH_extended.csv")
df_articles = df_articles.iloc[0:20, :]
#%% Tokenization, removal of non-alphabetic characters, changing to lowercase, stopword removal and stemming
def preprocess(dataframe):
    df = dataframe.copy()
    stop_words = set(stopwords.words('english'))
    for row in range(len(df)):
        for column in range(1, 3):  # preprocessing title and abstract
            processedtext = []
            text = word_tokenize(df.iloc[row, column])  # tokenization
            for token in text:  # removal of non-alphabetical characters from title
                alphatoken = re.sub(r'[^a-zA-Z]', '', token)
                if len(alphatoken) > 2:  # remove empty strings and short words from title
                    lowertoken = alphatoken.lower()
                    if lowertoken not in stop_words:  # remove stopwords from title
                        stemtoken = ps.stem(lowertoken)  # change title to lowercase & stem
                        processedtext.append(stemtoken)
            df.iat[row, column] = processedtext  # overwriting value in dataframe
        for column in range(3, 5):  # changing mesh terms and pubtypes into lists
            df.iat[row, column] = list(df.iat[row, column].split(', '))

    return df

tokenized_df = preprocess(df_articles)
#%% Function to precede title, abstract, mesh terms and publication type with prefix
def add_prefixes(dataframe):
    df = dataframe.copy()
    for row in range(len(df)):
        for column in range(1, 5):  # only select columns with titles, abstracts, pubtypes and mesh terms
            for word in range(len(df.iloc[row, column])):
                if column == 1:  # precede words in the title with ti_
                    df.iloc[row, column][word] = "ti_" + df.iloc[row, column][word]
                if column == 2:  # precede words in the abstract with ab_
                    df.iloc[row, column][word] = "ab_" + df.iloc[row, column][word]
                if column == 3:  # precede mesh terms with mesh_
                    df.iloc[row, column][word] = "mesh_" + df.iloc[row, column][word]
                if column == 4:  # precede publication types with pt_
                    df.iloc[row, column][word] = "pt_" + df.iloc[row, column][word]
            # making list into string to avoid impractical lay-out when exporting dataframe to csv:
            df.iloc[row, column] = ", ".join(str(element) for element in df.iloc[row, column])
    return df


df_prefixes = add_prefixes(tokenized_df)


#%% Saving table with preprocessed articles, both version with and without prefixes
#processed_df.to_csv(path_or_buf= root_dir + "Intermediate_output/NLP_dataset_no_prefixes.csv", index=False)
#df_prefixes.to_csv(path_or_buf= root_dir + "Intermediate_output/NLP_dataset_with_prefixes.csv", index=False)

