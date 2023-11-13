Contents:

- Data cleaning performed in R:
  * Recreation_dataset.R:  Contains the steps (1) retrieval of articles based on queries (2) retrieval of relevant articles from EndNote library and (3) combining these dataframes.
    The result is a dataframe with the PMID, title, abstract, publication type, MeSH terms and relevance (0 or 1) of each article from the query results.
  * Query_results.R: Script used to apply step 2 to all 53 queries and finally combine these into one dataframe (result can be loaded into Recreation_dataset.R)
  * Add_articles_not_in_query_results:  Due to suboptimal reproduction of the queries, not all relevant articles from the EndNote library were in Query_results.R. This script is used to retrieve these remaining relevant articles and add it to the result of Recreation_dataset.R

- Machine learning pipeline applied in Python:
  * NLP.py:  Applies natural language processing steps (tokenization, stopword removal, etc.) to the result of Recreation_dataset.R and adds prefixes (e.g. 'ti_' for words in the title)
  * BOW.py: Changes the result of NLP.py to a bag-of-words representation, where each word (from title, abstract, publication type and MeSH terms) is a column. For each word from the title and abstract, each article has a count of how many times the word appears. Publication type and MeSH terms are binary variables.
  * TFIDF.py: Changes the result of BOW.py to a term-frequency inverse-document-frequency representation. Here, the number of occurrences of a word from the title or abstract (=BOW) is set off against the number of articles this word appears in.
  * Word2Vec.py: Changes the result of NLP.py and BOW.py into a Word2Vec representation. Publication types and MeSH terms remain binary as in the BOW, but the words from the title and abstract are translated to an vector representation in an n-dimensional vector-space.
  * Feature_selection.py: N features are selected based on term frequency (number of times word occurs across all articles), document frequency (number of documents containing word) and the chi-square test
  * Resampling.py: Class dictribution of dataset is changed through application of random undersampling (RUS) (randomly discarding instances from majority class), random oversampling (ROS) (random copying of instances from minority class), synthetic minority oversampling technique (SMOTE) (randomly creating synthetic, plausible instances of the minority class) and a combination of RUS and SMOTE
