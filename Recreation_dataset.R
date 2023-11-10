library(easyPubMed)
library(readr)
library(stringr)
library(textcat)
library(dplyr)
library(tidyr)
test_query_potato <- '("diabetes mellitus"[MeSH Terms] OR ("diabetes"[All Fields] AND "mellitus"[All Fields]) OR "diabetes mellitus"[All Fields] OR "diabetes"[All Fields]) AND (systematic[sb] OR Review[ptyp] OR Meta-Analysis[ptyp] OR Clinical Trial[ptyp] OR Randomized Controlled Trial[ptyp] OR Controlled Clinical Trial[ptyp] OR Observational Study[ptyp](=) AND ( "2014/10/01"[PDAT] : "2019/10/01"[PDAT] ) AND ("humans"[MeSH Terms]) AND ("solanum tuberosum"[MeSH Terms] OR ("solanum"[All Fields] AND "tuberosum"[All Fields]) OR "solanum tuberosum"[All Fields] OR "potato"[All Fields]))'
my_api_key = "da61fa20ee9ea3c1abf2ced18da6c4fa8909"
setwd("C:/Users/marti/OneDrive/Thesis/Datasets")

##############################################################################

#### STEP 1: RETRIEVING COMPLETE DATASET (QUERY RESULTS)

    ### Step 1.1: Retrieving title and abstract of each article ###
## Downloading articles from pubmed (API key required) and saving in 
## XML format as .txt file
batch_pubmed_download(test_query_potato, dest_dir = NULL, 
                      dest_file_prefix = "easyPubMed_data_", format = "xml", 
                      api_key = my_api_key	, 
                      batch_size = 400, res_cn = 1, encoding = "UTF8")
## function automatically parks output as file in your working directory-> need to read data in again
article_list <- articles_to_list("easyPubMed_data_01.txt", encoding = "UTF8")

## article_to_df parses an XML article and puts it into a dataframe
## doing this for each article and combining them into one dataframe
df_articles = data.frame() 
for (article in article_list) {     
  next_article <- article_to_df(article, autofill = FALSE,
                                max_chars = -1, getKeywords = TRUE,
                                getAuthors = FALSE)
  df_articles <- rbind(df_articles, next_article[,c(1,3,4)])} 
#only keep relevant columns (pmid, title, abstract)


    ### Step 1.2: Retrieving publication type and MeSH terms of each article ###
## (same function as step 1.1, but format = "medline" instead of "xml")
batch_pubmed_download(test_query_potato, dest_dir = NULL, 
                      dest_file_prefix = "medline_data_", format = "medline", 
                      api_key = my_api_key	, 
                      batch_size = 400, res_cn = 1, encoding = "UTF8")
## function automatically parks output as file in your working directory-> need to read data in again
medline_file <- read_file("medline_data_01.txt")

## Split medline txt file into separate articles and store as dataframe
separate_articles <- strsplit(medline_file, "\r\n\r\n")
separate_articles <- strsplit(separate_articles[[1]], "\r\n\r\n" )
sep_articles_df <- as.data.frame(do.call(rbind, separate_articles))
colnames(sep_articles_df) <- "medline_info"
  
## Create list with 169 existing publication types
all_pub_types <- c(read_lines("medline_PT.txt"))
for (i in 1:length(all_pub_types)){
  all_pub_types[[i]] <- str_replace(all_pub_types[[i]],",", ";")}
# replace ',' with ';' because ',' also separates items in a list
# and in later python scripts lists sometimes need to be converted 
# to strings and then split again based on ','

##creating a dataframe with publication type and MeSH terms
##article_to_df() does not work for "medline" output from 
##batch_pubmed_download() (only "XML") -> have to create dataframe from scratch

df_medline <- sep_articles_df %>% 
  mutate(pmid = str_extract(medline_info, "PMID- [0-9]+"))%>%
  mutate(pmid = str_extract(pmid, "[0-9]+")) %>%
  mutate(mesh = str_extract_all(medline_info, "MH  - .*"))%>%
  mutate(mesh = str_remove_all(mesh, "MH  - " ))%>% #Remove "MH - "
  mutate(mesh = str_remove_all(mesh, "\""))%>% #Remove " 
  mutate(mesh = str_sub(mesh,3,-2))%>% #Remove 'c(' at beginning and ')' at end
  mutate(pubtype = NA) #Create column to fill in pubtype in next step

for (row in 1:nrow(df_medline)){
  pt_one_article <- c()
  for (pt in all_pub_types){ # Retrieve publication types
    if (grepl(pt, df_medline[row,1])){
      pt_one_article <- append(pt_one_article, pt)# To list
      pt_one_article <- paste(pt_one_article, collapse = ", ")# To string
      df_medline[row, 4 ] <- pt_one_article}}} # New row containing pubtype
df_medline<-select(df_medline,-1) # Remove first column

## Combining dataframes with title & abstract and pubtype & MeSH
df_all_articles <- merge(df_articles, df_medline, by= "pmid")
## This was done for all queries and then all results were combined
## therefore, this step can also be skipped and 'all_queries_combined.csv' can
## be read as df_all_articles immediately at the beginning of step 3

##############################################################################

#### STEP 2: RETRIEVING PMIDS OF RELEVANT ARTICLES FROM ENDNOTE LIBRARY 

    ### Step 2.1: Retrieving list of article titles from Endnote ###
endnote_articles <- read_file("Diabetes_Articles.txt")
endnote_df <- as.data.frame(strsplit(endnote_articles, "Reference Type: "))
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
list_actually_english <- c(6, 11, 12, 18, 20, 21, 28, 30, 32, 33, 34, 36, 37)
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
## Export the three different tables to a separate folder (which only contains these three tables)
## e.g. write.table(df_one_pmid, file = ..../df_one_pmid.csv)
## For df_no_pmid and df_multi_pmid: find the right pmid manually and fill in (e.g. in Excel) -> save file again

# Import the three dataframes that you have manually completed 
read_in_one_pmid <- read.table("C:/Users/marti/OneDrive/Thesis/Datasets/pmid_dfs_to_combine/df_one_pmid.csv", header=TRUE)
read_in_no_pmid <- read.csv2("C:/Users/marti/OneDrive/Thesis/Datasets/pmid_dfs_to_combine/df_no_pmid.csv")
read_in_multi_pmid <- read.csv2("C:/Users/marti/OneDrive/Thesis/Datasets/pmid_dfs_to_combine/df_multi_pmid.csv")

df_all_pmid <- rbind(read_in_one_pmid, read_in_no_pmid, read_in_multi_pmid)
df_all_pmid <- df_all_pmid[df_all_pmid$pmid !='None',] # Remove all articles that had no pmid
df_all_pmid <- df_all_pmid %>% distinct() # A few duplicates remained -> Remove duplicates

write.table(df_all_pmid, "C:/Users/marti/OneDrive/Thesis/Datasets/Correct_Output/all_pmid.csv")

## Obtaining the PMID's takes +- 30 minutes
## Therefore, this step can also be skipped and 'df_all_pmid.csv' can
## be read as df_all_articles immediately at the beginning of step 3

##############################################################################

#### STEP 3: LABELLING RELEVANT AND IRRELEVANT ARTICLES IN COMPLETE DATASET

df_all_articles <- read.table("C:/Users/marti/OneDrive/Thesis/Datasets/Query_output_right_MeSH/all_queries_combined.csv")
df_all_pmid <- read.table ("C:/Users/marti/OneDrive/Thesis/Datasets/Intermediate_output/all_pmid.csv")
df_all_articles$relevance <- NA#add column that indicates which articles are relevant
for (row in 1:nrow(df_all_articles)){
  if (df_all_articles[row, 1]%in% df_all_pmid[,1])#moet complete_list_PMID worden
  {df_all_articles[row,6] <- 1} #article is relevant
  else {df_all_articles[row,6] <- 0} #article is irrelevant
}

print(length(which(df_all_articles$relevance==1))) #only 335
#write.table(df_all_articles, "C:/Users/marti/OneDrive/Thesis/Datasets/Intermediate_output/recreated_dataset_query_results_only.csv", row.names = FALSE)
