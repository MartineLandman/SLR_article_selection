## Written in R version 4.2.0
library(easyPubMed)
library(readr)
library(stringr)
library(textcat)
library(dplyr)
library(tidyr)
my_api_key = "..." # see README file

##############################################################################

#### STEP 1: RETRIEVING COMPLETE DATASET (QUERY RESULTS)
## Use the Queries.R script to create a file with all query results
## and read it in as df_all_articles at the beginning of step 3

##############################################################################

#### STEP 2: RETRIEVING PMIDS OF RELEVANT ARTICLES FROM ENDNOTE LIBRARY 

    ### Step 2.1: Retrieving list of article titles from Endnote ###
endnote_articles <- read_file("examplelibrary.txt")
endnote_df <- as.data.frame(strsplit(endnote_articles, "Reference Type: ")) #split file into separate articles
colnames(endnote_df) <- "info"

## Only keeping journal articles, books, book sections, generic 
## and conference proceedings -> Then only keep the titles of these articles
titles_df <- endnote_df %>% 
  filter(str_starts(info,  " Journal Article| Book| Generic| Conference"))%>%
  mutate(info = str_extract(info, "Title: .+"))%>%
  mutate(info = sub("Title: ", "", info)) %>% #Remove title
  mutate(info = str_replace_all(info, "–|—|‐", "-")) #Change to correct '-'

english_df <- titles_df %>% # Filter for only English language articles
  filter(textcat(info) =='english')
non_english_df <- titles_df %>% #Filter for only NON-English language articles
  filter(textcat(info) !='english')

## The textcat() function works imperfectly
## Manually check if articles in non_english_df are actually non-English
## Fill in the indexes of the articles that are actually English in the vector
## Then combine this dataframe with english_df
list_actually_english <- c(...) 
actually_english_df <- as.data.frame(non_english_df[list_actually_english, ])
colnames(actually_english_df) <- "info"
all_titles <- rbind(english_df, actually_english_df)


    ### Step 2.2: Retrieving PMIDs belonging to titles of relevant articles ###
## The function to retrieve PMIDs returns one, zero or multiple possible PMIDs 
df_one_pmid <- data.frame(title = character(), pmid=character()) # Initiate three dataframes for three cases
df_no_pmid <- data.frame(title = character(), pmid=character())
df_multi_pmid <- data.frame(title = character(), pmid=character()) 

## Retrieve PMIDs for all articles
for (row in 1:nrow(all_titles)){
  article_data <- get_pubmed_ids_by_fulltitle(all_titles[row,1], field = "[Title]", api_key = my_api_key)
  if (lengths(article_data["IdList"]) == 1){ 
    df_one_pmid <- df_one_pmid %>% add_row(title = all_titles[row,1], pmid = toString(article_data[["IdList"]]))}
  if (lengths(article_data["IdList"]) == 0){
    df_no_pmid <- df_no_pmid %>% add_row(title = all_titles[row,1], pmid = 'None')}
  if (lengths(article_data["IdList"]) > 1){
    df_multi_pmid <- df_multi_pmid %>% add_row(title = all_titles[row,1], pmid = toString(article_data[["IdList"]]))}
}
## Export the three different dataframes to a separate folder (which only contains these three files)
## For df_no_pmid and df_multi_pmid: find the right pmid manually and fill in (e.g. in Excel) -> save file again
# Import the three dataframes that you have manually completed 
read_in_one_pmid <- read.table("...csv", header=TRUE)
read_in_no_pmid <- read.csv2("....csv")
read_in_multi_pmid <- read.csv2("....csv")

df_all_pmid <- rbind(read_in_one_pmid, read_in_no_pmid, read_in_multi_pmid)
df_all_pmid <- df_all_pmid[df_all_pmid$pmid !='None',] # Remove all articles that had no pmid
df_all_pmid <- df_all_pmid %>% distinct() # A few duplicates remained -> Remove duplicates
## Obtaining the PMID's takes long, so therefore we recommend storing the file
write.table(df_all_pmid, "....csv")


##############################################################################

#### STEP 3: LABELLING RELEVANT AND IRRELEVANT ARTICLES IN COMPLETE DATASET

df_all_articles <- read.table("...csv") # read in result from Queries.R
df_all_pmid <- read.table ("...csv") # if you did step 2 earlier and saved result, read it in 
df_all_articles$relevance <- NA #add column for indicating which articles are relevant
for (row in 1:nrow(df_all_articles)){
  if (df_all_articles[row, 1]%in% df_all_pmid[,1])
  {df_all_articles[row,6] <- 1} #article is relevant
  else {df_all_articles[row,6] <- 0} #article is irrelevant
}
write.table(df_all_articles, "....csv", row.names = FALSE) #save end result

## Check how many of the relevant articles from the EndNote Library are in your
## recreated dataset from the query results
print(length(which(df_all_articles$relevance==1))) 
## If not all relevant articles are in there and you want to apply machine learning,
## we recommend using Add_articles_not_in_query_results.R to add missing relevant
## articles to increase training data 

