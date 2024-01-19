# Written in Python version 3.9
#%% Importing packages
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#%% Preprocessing of title and abstract (stopword removal, stemming, etc.)
# Input: DF with 6 columns: PMID, title, abstract, MeSH terms, publication type and relevance
# Output: DF with same 6 columns where title and abstract are preprocessed and each column has become a list
def preprocess(dataframe):
    df = dataframe.copy()
    stop_words = set(stopwords.words('english'))
    for row in range(len(df)):
        for column in range(1, 3):  # preprocessing title and abstract
            processedtext = [] # initiate list for all preprocessed words from title and abstract
            text = word_tokenize(df.iloc[row, column])  # tokenization
            for token in text:
                alphatoken = re.sub(r'[^a-zA-Z]', '', token) # removal of non-alphabetical characters
                if len(alphatoken) > 2:  # remove empty strings and short words from title
                    lowertoken = alphatoken.lower()  # change everything to lowercase
                    if lowertoken not in stop_words:  # remove stopwords from title
                        stemtoken = ps.stem(lowertoken)  # stemming
                        processedtext.append(stemtoken)
            df.iat[row, column] = processedtext  # overwriting value in dataframe
        # sometimes there is only a ',' between to mesh_terms instead of ', ' -> solution:
        df.iloc[row, 3] = re.sub(r'(?<=,)(?=[^\s])', r' ', str(df.iloc[row, 3]))
        for column in range(3, 5):  # mesh terms and publication types
            df.iat[row, column] = df.iloc[row, column].split(', ')  # changing into lists

    return df

#%% Preceding title, abstract, mesh terms and publication type with prefix
# Input: output from preprocess function
# Output: DF with same 6 columns where all words in title, abstract, MeSH and pubtypes have a prefix
# and are changed back from lists to strings
def add_prefixes(dataframe):
    df = dataframe.copy()
    for row in range(len(df)):
        for column in range(1, 5):  # don't select PMID and relevance columns
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





