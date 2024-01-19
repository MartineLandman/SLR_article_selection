## Written in R version 4.2.0
library(easyPubMed)
library(readr)
library(stringr)
library(textcat)
library(dplyr)
library(tidyr)
my_api_key <-  "..." # see README file

################################################################################

#### STEP 1: RETRIEVING RESULTS FOR EACH QUERY ###

  ### Step 1.1: Retrieving title and abstract of each article ###
## Downloading articles from pubmed (API key required) and saving in 
## XML format as .txt file
query <- '...'# write a query exactly as you would in the PubMed search bar
batch_pubmed_download(query, dest_dir = NULL,
                      dest_file_prefix = "xml_example_", format = "xml",
                      api_key = my_api_key	,
                      batch_size = 400, res_cn = 1, encoding = "UTF8")
## function automatically parks output as file in your working directory-> need to read data in again
article_list <- articles_to_list("xml_example_01.txt", encoding = "UTF8")

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
batch_pubmed_download(query, dest_dir = NULL, 
                     dest_file_prefix = "medline_example_", format = "medline", 
                     api_key = my_api_key	, 
                     batch_size = 400, res_cn = 1, encoding = "UTF8")
## function automatically parks output as file in your working directory-> need to read data in again
medline_file <- read_file("medline_example_01.txt")

## Split medline txt file into separate articles and store as dataframe
separate_articles <- strsplit(medline_file, "\r\n\r\n")
separate_articles <- strsplit(separate_articles[[1]], "\r\n\r\n" )
sep_articles_df <- as.data.frame(do.call(rbind, separate_articles))
colnames(sep_articles_df) <- "medline_info"

## Create list with 169 existing publication types
all_pub_types <- c(read_lines("./all_pub_types.txt")) # provided in GitHub repository

##creating a dataframe with publication type and MeSH terms
##article_to_df() does not work for "medline" output from 
##batch_pubmed_download() (only "XML") -> have to create dataframe from scratch

df_medline <- sep_articles_df %>% 
  mutate(pmid = str_extract(medline_info, "PMID- [0-9]+"))%>%   ## Extract PMIDs
  mutate(pmid = str_extract(pmid, "[0-9]+")) %>% ## Only keep PMID code
  mutate(mesh = str_extract_all(medline_info, "MH  - .*")) %>% ## Extract MeSH
  ## Work-around to replace "," INSIDE MesH terms (e.g. Diet, Mediterranean) with ";" 
  ## in order to distinguish from comma's IN BETWEEN MeSH terms:
  mutate(mesh = str_replace_all(mesh, "\",", "###")) %>%
  mutate(mesh = str_replace_all(mesh, "," , ";" )) %>%
  mutate(mesh = str_replace_all(mesh, "###", "," )) %>%
  mutate(mesh = str_remove_all(mesh, "MH  - " ))%>% #Remove "MH - "
  mutate(mesh = str_remove_all(mesh, "[*\n\"]"))%>% #Remove *, \n and "
  mutate(mesh = str_sub(mesh,3,-2))%>% #Remove 'c(' at beginning and ')' at end
  mutate(pubtype = NA) #Create column to fill in pubtype in next step

## In the XML file, MeSH terms are formatted like Hypertension/etiology/mortality
## which needs to be Hypertension:etiology and Hypertension:mortality
for (row in 1:nrow(df_medline)){
  mesh_right_format = ""
  mesh_terms = str_split_1(df_medline[row,3], ",")
  for (mesh_term in mesh_terms){
    if (str_count((mesh_term), "/") > 1){
      sep_word <- str_split_1(mesh_term, "/")
      for (word in 2:length(sep_word)){
        new_combi = paste(sep_word[1], sep_word[word], sep='/')
        mesh_right_format <- paste(mesh_right_format, new_combi, sep =',')}}
    else {mesh_right_format <- paste(mesh_right_format, mesh_term, sep =',')}
  } 
  df_medline[row,3] <- str_sub(mesh_right_format,2,-1)}

## Retrieving publication types by looping over list of all existing publication types in Pubmed
for (row in 1:nrow(df_medline)){
  pt_one_article <- c()
  for (pt in all_pub_types){ # Retrieve publication types
    if (grepl(pt, df_medline[row,1])){ # Check if publication type is in medline file of article
      pt <- str_replace_all(pt, ',', ';')# To distinguish from comma's in between pubtypes
      pt_one_article <- toString(append(pt_one_article, pt))# Append to list with all pubtypes of article
      df_medline[row, 4 ] <- pt_one_article}}} # New row containing pubtype
df_medline<-select(df_medline,-1) # Remove first column

## Combining dataframes with title & abstract and pubtype & MeSH
df_all_articles <- merge(df_articles, df_medline, by= "pmid")
write.table(df_all_articles, ".example_query_results.csv")
# For each query for which you have created a dataframe, save it in the same folder
# so at step 2 you can easily loop over them


################################################################################

    #### STEP 2: COMBINING ALL QUERY FILES INTO ONE FILE ###
all_queries_df <- data.frame(matrix(nrow=0, ncol=5)) # create empty dataframe
colnames(all_queries_df) <- c('pmid', 'title', 'abstract', 'pubtype', 'mesh')
filenames <- Sys.glob("*.csv") # access names of all files containing the query result dataframes
for (filename in filenames){
  print(filename)
  next_df <- read.table(filename, header=TRUE)
  colnames(next_df) <- c('pmid', 'title', 'abstract', 'pubtype', 'mesh')
  all_queries_df <- rbind(all_queries_df, next_df)} #combine all dataframe

#only keep unique rows (some articles are the result of multiple queries)
all_queries_df_unique <- all_queries_df %>% distinct()
write.table(all_queries_df_unique, 'all_queries_combined.csv') #save 




