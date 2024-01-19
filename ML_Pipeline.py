# Written in Python version 3.9
#%% Importing packages
import pandas as pd
import gensim
from sklearn import svm, utils
from sklearn import preprocessing
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn import metrics

# Import self-written modules
import Feature_selection as fs
import Resampling as rs
import NLP_preprocessing as nlp
import BOW as bow
import TFIDF as tfidf
import Word2Vec as w2v


#%% Generating dataframes with the four different vector representations
# First, importing dataset with articles (result from R script)
# (PMID, title, abstract, MeSH terms, publication types, relevance)
df_articles = pd.read_csv(filepath_or_buffer="....csv")
# Adding prefixes
df_prefixes = nlp.add_prefixes(nlp.preprocess(df_articles))

#Generate BOW dataframe
df_bow = bow.generate_bow(df_prefixes)
df_bow.to_csv(path_or_buf="...BOW.csv", index=False) # takes long, so we recommend storing the end result

#Generate TFIDF dataframe
df_tfidf = tfidf.generate_tfidf(df_prefixes, df_bow)
df_tfidf.to_csv(path_or_buf="...TFIDF.csv", index=False)

#Generate self-trained Word2Vec dataframe
df_w2v_preproc =w2v.preprocess_for_w2v(df_articles, df_bow)
model_w2v= w2v.create_w2v_model(df_w2v_preproc, 300)
df_w2v = w2v.vectorize(df_w2v_preproc, model_w2v, 300)
df_w2v.to_csv(path_or_buf= "....W2V.csv", index=False)

#Generate pretrained Word2Vec dataframe
model_w2v_pretrained = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
df_w2v_pretrained = w2v.vectorize(df_w2v_preproc, model_w2v_pretrained, 300)
df_w2v_pretrained.to_csv(path_or_buf= "....W2V_pretrained.csv", index=False)

#%% Create DF with settings for all modelling instances for cNB
df_cnb_parameters = pd.DataFrame(columns = ['vec_repr', 'fs_method', 'n_features', 'smoothing_param'])
for vec_repr in ['df_bow', 'df_tfidf']:
    for fs_method in ['none', 'docfreq', 'chisq']:
        if fs_method == 'none':
            n = 'all'
            for smoothing_param in [.001, .4, .6, .8, 1.0]:
                df_cnb_parameters.loc[len(df_cnb_parameters)] = [vec_repr, fs_method, n, smoothing_param]
        else:
            for n in [5, 10, 25, 50, 100, 300, 500, 1000, 2000, 4000, 8000, 15000]:
                for smoothing_param in [.001, .4, .6, .8, 1.0]:
                    df_cnb_parameters.loc[len(df_cnb_parameters)] = [vec_repr, fs_method, n, smoothing_param]

#%% Generate cNB models and results
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5) # set 5*2 = 10 iterations for pipeline
cnb_results_df = pd.DataFrame(columns = ['model', 'vec_repr', 'fs_method', 'n_features', 'smoothing_param', 'precision', 'recall', 'f_one', 'wss_95'])
for row in range (len(df_cnb_parameters)): # iterate over df with hyperparameters
    vec_repr_df, train_fs_df, test_fs_df= pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # initialize to later be overwritten
    # Extract methods and parameters from df with hyperparameters:
    vec_repr = df_cnb_parameters.loc[df_cnb_parameters.index[row], 'vec_repr']
    fs_method = df_cnb_parameters.loc[df_cnb_parameters.index[row], 'fs_method']
    n_features = df_cnb_parameters.loc[df_cnb_parameters.index[row], 'n_features']
    smoothing_param = df_cnb_parameters.loc[df_cnb_parameters.index[row], 'smoothing_param']

    # Choose vector representation:
    if df_cnb_parameters.loc[df_cnb_parameters.index[row], 'vec_repr'] == 'df_bow':
        vec_repr_df = df_bow
    if df_cnb_parameters.loc[df_cnb_parameters.index[row], 'vec_repr'] == 'df_tfidf':
        vec_repr_df = df_tfidf

    precision_scores = [] # assign empty lists to later calculate average metrics over 10 iterations
    recall_scores = []
    f_one_scores = []
    wss95_scores = []

    X, y = vec_repr_df.iloc[:,2:], vec_repr_df.iloc[:,1] # assign input and output features
    for i, (train_index, test_index) in enumerate(rskf.split(X,y)): # create 5 x 2 stratified k-folds
        train_df = vec_repr_df.loc[train_index].reset_index(drop=True) # create training dataset
        test_df = vec_repr_df.loc[test_index].reset_index(drop=True) # create test dataset
        if df_cnb_parameters.loc[df_cnb_parameters.index[row], 'fs_method'] == 'docfreq': # apply docfreq feature selection to training data
            train_fs_df = fs.generate_docfreqFS(train_df, n_features) # select relevant features from training data
            test_fs_df = test_df[train_fs_df.columns] # select same features from test data
        if df_cnb_parameters.loc[df_cnb_parameters.index[row], 'fs_method'] == 'chisq': # apply chi square feature selection to training data
            train_fs_df = fs.generate_chisq_FS(train_df, n_features) # select relevant features from training data
            test_fs_df = test_df[train_fs_df.columns] # select same features from test data
        if df_cnb_parameters.loc[df_cnb_parameters.index[row], 'fs_method'] == 'none': # apply no feature selection to training data
            train_fs_df = train_df.copy()
            test_fs_df = test_df.copy()

        # Training of the model
        shuffled_train_df = utils.shuffle(train_fs_df) # shuffle order of training data so the model does not learn from the order of data
        X_train, y_train = shuffled_train_df.iloc[:,2:], shuffled_train_df.iloc[:,1] # assign input and output features of training data
        model =  ComplementNB(alpha= smoothing_param) # assign model (with smoothing parameter)
        model.fit(X_train, y_train) # train model on training data

        # Testing of the model
        X_test, y_test = test_fs_df.iloc[:,2:], test_fs_df.iloc[:,1] # assign input and output features of test data
        predictions = model.predict(X_test) # use model to predict output of test data
        # generate evaluation metrics for model performance and add to list:
        precision_scores.append(metrics.precision_score(y_test, predictions))
        recall_scores.append(metrics.recall_score(y_test, predictions))
        f_one_scores.append(metrics.f1_score(y_test, predictions))
        cm = confusion_matrix(y_test, predictions)
        wss95_scores.append((cm[0,0]+cm[1,0])/len(y_test)-0.05) # manually generate WSS@95 score

    # Calculate average of evaluation metric scores over 10 iterations:
    precision = sum(precision_scores)/len(precision_scores)
    recall = sum(recall_scores)/len(recall_scores)
    f_one = sum(f_one_scores)/len(f_one_scores)
    wss95 = sum(wss95_scores)/len(wss95_scores)
    # add results to dataframe with all results
    cnb_results_df.loc[len(cnb_results_df)] = ['cNB', vec_repr, fs_method, n_features, smoothing_param, precision, recall, f_one, wss95]
    # optional: print results, so you know how far along the function is
    print('cNB', vec_repr, fs_method, n_features, smoothing_param, precision, recall, f_one, wss95)

# Create DF with highest recall for each vector x feature selection method combination
cnb_results_copy = cnb_results_df.copy()
ids_cnb = cnb_results_copy.groupby(['vec_repr', 'fs_method'])['recall'].idxmax()
df_cnb_highest_recall = cnb_results_copy.loc[ids_cnb]

#%% Create DF with settings for all modelling instances for SVM
df_svm_parameters = pd.DataFrame(columns = ['vec_repr', 'fs_method', 'rs_method', 'n_features', 'rs_ratio'])
for vec_repr in ['df_bow', 'df_tfidf', 'df_w2v', 'df_w2v_pretrain']:
    for fs_method in ['chisq', 'docfreq']:
        for n in [10, 25, 50, 100, 300, 500, 1000, 2000, 4000, 8000, 15000]:
            for rs_method in ['none', 'rus', 'ros', 'smote', 'smote+rus']:
                if rs_method == 'none':
                    rs_ratio = 'na'
                    df_svm_parameters.loc[len(df_svm_parameters)] = [vec_repr, fs_method, rs_method, n, rs_ratio]
                if rs_method == 'rus':
                    for rs_ratio in [0.25, 0.43, 0.67, 1]:
                        df_svm_parameters.loc[len(df_svm_parameters)] = [vec_repr, fs_method, rs_method, n, rs_ratio]
                if rs_method == 'smote+rus':
                    for rs_ratio in [[0.21, 0.21], [0.21, 0.42,], [0.32, 0.21], [0.32, 0.42]]:
                        df_svm_parameters.loc[len(df_svm_parameters)] = [vec_repr, fs_method, rs_method, n, rs_ratio]
                if rs_method == 'ros'  or rs_method == 'smote':
                    for rs_ratio in [0.21, 0.32, 0.41, 0.53]:
                        df_svm_parameters.loc[len(df_svm_parameters)] = [vec_repr, fs_method, rs_method, n, rs_ratio]

# Drop all combinations that should not / cannot be generated
df_svm_parameters = df_svm_parameters.loc[~((df_svm_parameters['vec_repr'] == 'df_w2v') & (df_svm_parameters['n_features'] < 300))]
df_svm_parameters = df_svm_parameters.loc[~((df_svm_parameters['vec_repr'] == 'df_w2v_pretrain') & (df_svm_parameters['n_features'] < 300))]
df_svm_parameters = df_svm_parameters.reset_index(drop=True)

#%% Generate SVM models and results
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)
svm_results_df = pd.DataFrame(columns = ['vec_repr', 'fs_method', 'n_features', 'rs_method', 'rs_ratio', 'precision',
                                         'recall', 'f_one', 'wss_95'])
for row in range (len(df_svm_parameters)): # iterate over df with hyperparameters
    vec_repr_df, train_df, test_df, train_fs_df, test_fs_df, train_rs_df= (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                                                                           pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
                                                                           # initialize to later be overwritten
    # Extract methods and parameters from df with hyperparameters:
    vec_repr = df_svm_parameters.loc[df_svm_parameters.index[row], 'vec_repr']
    fs_method = df_svm_parameters.loc[df_svm_parameters.index[row], 'fs_method']
    n_features = df_svm_parameters.loc[df_svm_parameters.index[row], 'n_features']
    rs_method = df_svm_parameters.loc[df_svm_parameters.index[row], 'rs_method']
    rs_ratio = df_svm_parameters.loc[df_svm_parameters.index[row], 'rs_ratio']

    # Choose vector representation:
    if df_svm_parameters.loc[df_svm_parameters.index[row], 'vec_repr'] == 'df_bow':
        vec_repr_df = df_bow
    if df_svm_parameters.loc[df_svm_parameters.index[row], 'vec_repr'] == 'df_tfidf':
        vec_repr_df = df_tfidf
    if df_svm_parameters.loc[df_svm_parameters.index[row], 'vec_repr'] == 'df_w2v':
        vec_repr_df = df_w2v
    if df_svm_parameters.loc[df_svm_parameters.index[row], 'vec_repr'] == 'df_w2v_pretrain':
        vec_repr_df = df_w2v_pretrained

    precision_scores = [] # assign empty lists to later calculate average metrics over 10 iterations
    recall_scores = []
    f_one_scores = []
    wss95_scores = []

    X, y = vec_repr_df.iloc[:,2:], vec_repr_df.iloc[:,1] # assign input and output features
    for i, (train_index, test_index) in enumerate(rskf.split(X,y)): # create 5 x 2 stratified k-folds
        train_df = vec_repr_df.loc[train_index].reset_index(drop=True) # create training dataset
        test_df = vec_repr_df.loc[test_index].reset_index(drop=True) # create test dataset
        if df_svm_parameters.loc[df_svm_parameters.index[row], 'vec_repr'] == 'df_bow': # Exception: BOW needs to be scaled
             scaler = preprocessing.MinMaxScaler() # choose scaling method
             scaler.fit(train_df.iloc[:,2:]) # fit scaler on training data
             # apply scalers to both training and test data
             train_df = pd.concat([train_df.iloc[:,:2], pd.DataFrame(scaler.transform(train_df.iloc[:,2:]))], axis=1)
             test_df = pd.concat([test_df.iloc[:,:2], pd.DataFrame(scaler.transform(test_df.iloc[:,2:]))], axis=1)

        #Feature selection for BOW and TFIDF
        if df_svm_parameters.loc[df_svm_parameters.index[row], 'vec_repr'] == 'df_bow' or df_svm_parameters.loc[df_svm_parameters.index[row], 'vec_repr'] == 'df_tfidf':
            if df_svm_parameters.loc[df_svm_parameters.index[row], 'fs_method'] == 'docfreq': # apply docfreq feature selection to training data
                train_fs_df = fs.generate_docfreqFS(train_df, n_features) # select relevant features from training data
                test_fs_df = test_df[train_fs_df.columns] # select same features from test data
            if df_svm_parameters.loc[df_svm_parameters.index[row], 'fs_method'] == 'chisq': # apply chi square feature selection to training data
                train_fs_df = fs.generate_chisq_FS(train_df, n_features) # select relevant features from training data
                test_fs_df = test_df[train_fs_df.columns] # select same features from test data

        # Feature selection for W2V and W2V_pretrain: should always keep the 300 dimensions from the word2vec vectors as features
        if df_svm_parameters.loc[df_svm_parameters.index[row], 'vec_repr'] == 'df_w2v' or df_svm_parameters.loc[df_svm_parameters.index[row], 'vec_repr'] == 'df_w2v_pretrain':
            if df_svm_parameters.loc[df_svm_parameters.index[row], 'fs_method'] == 'docfreq': # apply docfreq feature selection to training data
                train_fs_df = docFS_df = pd.concat([fs.generate_docfreqFS(train_df.iloc[:,:-300], n_features-300), train_df.iloc[:,-300:]], axis=1)
                test_fs_df = test_df[train_fs_df.columns] # select same features from test data
            if df_svm_parameters.loc[df_svm_parameters.index[row], 'fs_method'] == 'chisq': # apply chi square feature selection to training data
                train_fs_df = pd.concat([fs.generate_chisq_FS(train_df.iloc[:,:-300], n_features-300), train_df.iloc[:,-300:]], axis=1)
                test_fs_df = test_df[train_fs_df.columns] # select same features from test data

        # Resampling of the training data
        if df_svm_parameters.loc[df_svm_parameters.index[row], 'rs_method'] == 'none':
            train_rs_df = train_fs_df.iloc[:,1:].copy()
        if df_svm_parameters.loc[df_svm_parameters.index[row], 'rs_method'] == 'rus':
            train_rs_df = rs.generate_rus(train_fs_df, rs_ratio)
        if df_svm_parameters.loc[df_svm_parameters.index[row], 'rs_method'] == 'ros':
            train_rs_df = rs.generate_ros(train_fs_df, rs_ratio)
        if df_svm_parameters.loc[df_svm_parameters.index[row], 'rs_method'] == 'smote':
            train_rs_df = rs.generate_smote(train_fs_df, rs_ratio)
        if df_svm_parameters.loc[df_svm_parameters.index[row], 'rs_method'] == 'smote+rus':
            part_smote_df = rs.generate_smote(train_fs_df, rs_ratio[0])
            part_rus_df = rs.generate_rus(train_fs_df, rs_ratio[1])
            train_rs_df = pd.concat([part_smote_df[part_smote_df['relevance']==1], part_rus_df[part_rus_df['relevance']==0]], ignore_index = True)

        # Training of the model
        shuffled_train_df = utils.shuffle(train_rs_df) # shuffle so model does not learn from the order of training data
        dual_param = None # initiate hyperparameter for SVM
        if len(shuffled_train_df) > n_features: # if n observations > n features:
            dual_param = False # set hyperparameter
        if len(shuffled_train_df) < n_features: # if n observations < n features:
            dual_param = True # set hyperparameter
        svm_for_gridsearch = svm.LinearSVC(dual = dual_param)
        grid = None # initiate hyperparameter for SVM
        if n_features < 300:
            grid = model_selection.GridSearchCV(estimator = svm_for_gridsearch, param_grid= {'C': [0.00001,0.0001,0.001,
                                                                                                   0.01,0.1,1,10,100,1000,10000]}, scoring= 'recall')
        else: # large C causes causes strange model behaviour when n_features >= 300, so limit possibilities for C
            grid = model_selection.GridSearchCV(estimator = svm_for_gridsearch, param_grid= {'C': [0.00001,0.0001,0.001,
                                                                                                   0.01,0.1,1]}, scoring= 'recall')
        X_model_training= shuffled_train_df.iloc[:,1:] # assign title, abstract, MeSH terms and pubtype variables as input
        y_model_training= shuffled_train_df.iloc[:,0] # assign relevance as output
        grid.fit(X_model_training, y_model_training)
        C_param = grid.best_params_['C'] # apply optimized value for C
        model = svm.LinearSVC(dual = dual_param, C = C_param) # set parameters
        model.fit(X_model_training, y_model_training) # train model

        # Testing of the model
        X_test, y_test = test_fs_df.iloc[:,2:], test_fs_df.iloc[:,1] # assign input and output features of test data
        predictions = model.predict(X_test) # use model to predict output of test data
        # generate evaluation metrics for model performance and add to list
        precision_scores.append(metrics.precision_score(y_test, predictions))
        recall_scores.append(metrics.recall_score(y_test, predictions))
        f_one_scores.append(metrics.f1_score(y_test, predictions))
        cm = confusion_matrix(y_test, predictions)
        wss95_scores.append((cm[0,0]+cm[1,0])/len(y_test)-0.05) # manually generate WSS@95 score

    # calculate average of evaluation metric scores over 10 iterations:
    precision = sum(precision_scores)/len(precision_scores)
    recall = sum(recall_scores)/len(recall_scores)
    f_one = sum(f_one_scores)/len(f_one_scores)
    wss95 = sum(wss95_scores)/len(wss95_scores)
    # add results to dataframe with all results
    svm_results_df.loc[len(svm_results_df)] = [vec_repr, fs_method, n_features, rs_method, rs_ratio, precision, recall, f_one, wss95]
    # optional: print results, so you know how far along the function is
    print('SVM', vec_repr, fs_method, n_features, rs_method, rs_ratio, precision, recall, f_one, wss95)

# Create DF with highest recall for each vector x feature selection method x resampling method combination
svm_results_copy = svm_results_df.copy()
ids_svm = svm_results_copy.groupby(['vec_repr', 'fs_method', 'rs_method'])['recall'].idxmax()
df_svm_highest_recall = svm_results_copy.loc[ids_svm]
