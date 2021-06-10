#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import pca
from sklearn.metrics import (f1_score, roc_auc_score, roc_curve, auc, confusion_matrix, precision_recall_curve, 
                             average_precision_score, classification_report, accuracy_score)
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier

from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, TomekLinks
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

from sklearn.preprocessing import MinMaxScaler
import itertools
import operator

from sklearn.model_selection import StratifiedKFold


# In[2]:


def generateStandardTimeSeriesStructure(all_releases_df, ws):
 
    print("Generating a new dataframe without containing the last release...")
    df = all_releases_df[all_releases_df['release'] != all_releases_df['release'].max()]
    print("... DONE!")

    df.drop(columns=['class_frequency', 'number_of_changes', 'release', 'Name', 'Kind', 'File'])
    
    print("checking class larger than window size...")

    window_size=ws;

    class_names_list = df['Path'].unique().tolist()
    classes_to_drop_list = list()
    for class_name in class_names_list:
        if len(df[df.Path == class_name].iloc[::-1]) <= window_size:
            for drop_class in df.index[df.Path==class_name].tolist():
                classes_to_drop_list.append(drop_class)


    df = df.drop(classes_to_drop_list, axis=0)
    df = df.iloc[::-1]

    print("DONE")
    
    print("Setting the features...")
    class_names_list = df['Path'].unique().tolist()
    features_list = ['CountClassCoupled', 'CountClassDerived', 'CountDeclMethod', 'CountDeclMethodAll', 'CountLineCode', 'MaxInheritanceTree', 'PercentLackOfCohesion', 'SumCyclomatic']
    print("DONE")
    
    timeseries_list = list()
    timeseries_labels = list()
    for class_name in class_names_list:
        class_sequence = df[df.Path == class_name].reset_index()
        for row in range(len(class_sequence)-1):
            window = list()
            # print('row: ', row)
            if row + window_size < len(class_sequence) + 1:
                for i in range(window_size):
                    #print(row+i)
                    window.extend(class_sequence.loc[row + i, features_list].values.astype(np.float64))
                timeseries_labels.append(class_sequence.loc[row + i, 'will_change'])
                timeseries_list.append(window)
                
    timeseries_X = np.array(timeseries_list)
    timeseries_labels = np.array(timeseries_labels).astype(np.bool)
    
    print("X:", timeseries_X.shape, "y:", timeseries_labels.shape)
    
    return timeseries_X, timeseries_labels


# # Performance Metrics

# In[3]:


def get_scores(y_test, y_pred):
    scores = []
    
    scores.append(f1_score(y_test, y_pred, average='micro'))
    print("F1-Score(micro): " + str(scores[-1]))
    
    scores.append(f1_score(y_test, y_pred, average='macro'))
    print("F1-Score(macro): " + str(scores[-1]))
    
    scores.append(f1_score(y_test, y_pred, average='weighted'))
    print("F1-Score(weighted): " + str(scores[-1]))
    
    scores.append(f1_score(y_test, y_pred, average=None))
    print("F1-Score(None): " + str(scores[-1]))
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    #ACC
    scores.append(accuracy_score(y_test, y_pred, normalize=True))
    print("Accuracy: " + str(scores[-1]))
    
    #Sensitivity
    sensitivity = tp / (tp+fn)
    scores.append(tp / (tp+fn))
    print("Sensitivity: " + str(scores[-1]))
    
    #Specificity
    specificity = tn / (tn+fp)
    scores.append (tn / (tn+fp))
    print("Specificity: " + str(scores[-1]))
    
    #VPP
    scores.append(tp / (tp+fp))
    #print("VPP: " + str(scores[-1]))
    
    #VPN
    scores.append(tn / (tn+fn))
    #print("VPN: " + str(scores[-1]))
    
    #RVP
    scores.append(sensitivity / (1-specificity))
    #print("RVP: " + str(scores[-1]))
    
    #RVN
    scores.append((1 - sensitivity) / specificity)
    #print("RVN: " + str(scores[-1]))
    
    #Confusion Matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    print("Confusion Matrix: [" + str(cnf_matrix[0][0]) + ", " + str(round(cnf_matrix[1][1],2)) + "]")
    
    #ROC_AUC
    scores.append(roc_auc_score(y_test, y_pred))
    print("ROC AUC score: " + str(scores[-1]))
        
    scores.append([tn, fp, fn, tp])
    
    return scores


# In[4]:


def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #    print("Normalized confusion matrix")
    #else:
    #    print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    #tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, classes, rotation=45)
    #plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def plot_confusion_matrixes(y_test, y_pred):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    #plt.figure()
    #plt.subplots(1,2,figsize=(20,4))
    #plt.subplot(1,2,1)
    #plot_confusion_matrix(cnf_matrix, title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    #plt.subplot(1,2,2)
    plot_confusion_matrix(cnf_matrix, normalize=True, title='Normalized confusion matrix')

    plt.tight_layout()
    plt.show()


# # Algorithms

# In[5]:


def LogisticRegr_(Xtrain, Ytrain, Xtest, Ytest):
    print("\nLOGISTIC REGRESSION")
    cv_score = []
    i = 1
    print("TRAIN AND VALIDATION SETS:")
    for train_index, test_index in kf.split(Xtrain, Ytrain):
        print('{} of KFold {}'.format(i,kf.n_splits))
        xtr_LR, xvl_LR = X_train.iloc[train_index], X_train.iloc[test_index]
        ytr_LR, yvl_LR = y_train.iloc[train_index], y_train.iloc[test_index]

        #model
        lr = LogisticRegression(solver='lbfgs', random_state=42, class_weight='balanced')
        lr.fit(xtr_LR, ytr_LR.values.ravel())
        score = roc_auc_score(yvl_LR, lr.predict(xvl_LR))
        print('ROC AUC score:',score)
        cv_score.append(score)    
        i+=1

    print('\nCROSS VALIDANTION SUMMARY:')
    print('Mean: ' + str(np.mean(cv_score)))
    print('Std deviation: ' + str(np.std(cv_score)))

    print("\nTEST SET:")
    get_scores(Ytest, lr.predict(Xtest))


# In[6]:


def DecisionTree_(Xtrain, Ytrain, Xtest, Ytest):
    get_ipython().run_line_magic('time', '')
    print("\nDECISION TREE")
    cv_score = []
    i = 1
    print("TRAIN AND VALIDATION SETS:")
    for train_index, test_index in kf.split(Xtrain, Ytrain):
        print('{} of KFold {}'.format(i,kf.n_splits))
        xtr_DT, xvl_DT = X_train.iloc[train_index], X_train.iloc[test_index]
        ytr_DT, yvl_DT = y_train.iloc[train_index], y_train.iloc[test_index]

        #model
        dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
        dt.fit(xtr_DT, ytr_DT.values.ravel())
        score = roc_auc_score(yvl_DT, dt.predict(xvl_DT))
        print('ROC AUC score:',score)
        cv_score.append(score)    
        i+=1


    print('\nCROSS VALIDANTION SUMMARY:')
    print('Mean: ' + str(np.mean(cv_score)))
    print('Std deviation: ' + str(np.std(cv_score)))    

    print("\nTEST SET:")
    get_scores(Ytest, dt.predict(Xtest))


# In[7]:


def RandomForest_(Xtrain, Ytrain, Xtest, Ytest):
    get_ipython().run_line_magic('time', '')
    print("RANDOM FOREST")
    cv_score = []
    i = 1
    print("TRAIN AND VALIDATION SETS:")
    for train_index, test_index in kf.split(Xtrain, Ytrain):
        print('{} of KFold {}'.format(i,kf.n_splits))
        xtr_RF, xvl_RF = X_train.iloc[train_index], X_train.iloc[test_index]
        ytr_RF, yvl_RF = y_train.iloc[train_index], y_train.iloc[test_index]

        #model
        rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
        rf.fit(xtr_RF, ytr_RF.values.ravel())
        score = roc_auc_score(yvl_RF, rf.predict(xvl_RF))
        print('ROC AUC score:',score)
        cv_score.append(score)    
        i+=1

    print('\nCROSS VALIDANTION SUMMARY:')
    print('Mean: ' + str(np.mean(cv_score)))
    print('Std deviation: ' + str(np.std(cv_score)))    

    print("\nTEST SET:")
    get_scores(Ytest, rf.predict(Xtest))


# In[8]:


def NN_(Xtrain, Ytrain, Xtest, Ytest):
    get_ipython().run_line_magic('time', '')

    print("NEURAL NETWORK")
    cv_score = []
    i = 1
    print("TRAIN AND VALIDATION SETS:")
    for train_index, test_index in kf.split(Xtrain, Ytrain):
        print('{} of KFold {}'.format(i,kf.n_splits))
        xtr_NN, xvl_NN = X_train.iloc[train_index], X_train.iloc[test_index]
        ytr_NN, yvl_NN = y_train.iloc[train_index], y_train.iloc[test_index]

        #model
        nn = MLPClassifier(random_state=42)
        nn.fit(xtr_NN, ytr_NN.values.ravel())
        score = roc_auc_score(yvl_NN, nn.predict(xvl_NN))
        print('ROC AUC score:',score)
        cv_score.append(score)    
        i+=1

    print('\nCROSS VALIDANTION SUMMARY:')
    print('Mean: ' + str(np.mean(cv_score)))
    print('Std deviation: ' + str(np.std(cv_score)))   

    print("\nTEST SET:")
    get_scores(Ytest, nn.predict(Xtest))


# # Algorithms (no iloc)

# In[9]:


def LogisticRegr_NoIloc(Xtrain, Ytrain, Xtest, Ytest):
    get_ipython().run_line_magic('time', '')
    print("\nLOGISTIC REGRESSION")
    cv_score = []
    i = 1
    print("TRAIN AND VALIDATION SETS:")
    for train_index, test_index in kf.split(Xtrain, Ytrain):
        print('{} of KFold {}'.format(i,kf.n_splits))
        xtr_LR, xvl_LR = Xtrain[train_index], Xtrain[test_index]
        ytr_LR, yvl_LR = Ytrain[train_index], Ytrain[test_index]

        #model
        lr = LogisticRegression(solver='lbfgs', random_state=42, class_weight='balanced')
        lr.fit(xtr_LR, ytr_LR)
        score = roc_auc_score(yvl_LR, lr.predict(xvl_LR))
        print('ROC AUC score:',score)
        cv_score.append(score)    
        i+=1

    print('\nCROSS VALIDANTION SUMMARY:')
    print('Mean: ' + str(np.mean(cv_score)))
    print('Std deviation: ' + str(np.std(cv_score)))

    print("\nTEST SET:")
    get_scores(Ytest, lr.predict(Xtest))


# In[10]:


def DecisionTree_NoIloc(Xtrain, Ytrain, Xtest, Ytest):
    get_ipython().run_line_magic('time', '')
    print("\nDECISION TREE")
    cv_score = []
    i = 1
    print("TRAIN AND VALIDATION SETS:")
    for train_index, test_index in kf.split(Xtrain, Ytrain):
        print('{} of KFold {}'.format(i,kf.n_splits))
        xtr_DT, xvl_DT = Xtrain[train_index], Xtrain[test_index]
        ytr_DT, yvl_DT = Ytrain[train_index], Ytrain[test_index]
        #model
        dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
        dt.fit(xtr_DT, ytr_DT)
        score = roc_auc_score(yvl_DT, dt.predict(xvl_DT))
        print('ROC AUC score:',score)
        cv_score.append(score)    
        i+=1

    print('\nCROSS VALIDANTION SUMMARY:')
    print('Mean: ' + str(np.mean(cv_score)))
    print('Std deviation: ' + str(np.std(cv_score)))    

    print("\nTEST SET:")
    get_scores(Ytest, dt.predict(Xtest))


# In[11]:


def RandomForest_NoIloc(Xtrain, Ytrain, Xtest, Ytest):
    get_ipython().run_line_magic('time', '')
    print("RANDOM FOREST")
    cv_score = []
    i = 1
    print("TRAIN AND VALIDATION SETS:")
    for train_index, test_index in kf.split(Xtrain, Ytrain):
        print('{} of KFold {}'.format(i,kf.n_splits))
        xtr_RF, xvl_RF = Xtrain[train_index], Xtrain[test_index]
        ytr_RF, yvl_RF = Ytrain[train_index], Ytrain[test_index]

        #model
        rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
        rf.fit(xtr_RF, ytr_RF)
        score = roc_auc_score(yvl_RF, rf.predict(xvl_RF))
        print('ROC AUC score:',score)
        cv_score.append(score)    
        i+=1

    print('\nCROSS VALIDANTION SUMMARY:')
    print('Mean: ' + str(np.mean(cv_score)))
    print('Std deviation: ' + str(np.std(cv_score)))    

    print("\nTEST SET:")
    get_scores(Ytest, rf.predict(Xtest))


# In[12]:


def NN_NoIloc(Xtrain, Ytrain, Xtest, Ytest):
    get_ipython().run_line_magic('time', '')

    print("NEURAL NETWORK")
    cv_score = []
    i = 1
    print("TRAIN AND VALIDATION SETS:")
    for train_index, test_index in kf.split(Xtrain, Ytrain):
        print('{} of KFold {}'.format(i,kf.n_splits))
        xtr_NN, xvl_NN = Xtrain[train_index], Xtrain[test_index]
        ytr_NN, yvl_NN = Ytrain[train_index], Ytrain[test_index]

        #model
        nn = MLPClassifier(random_state=42)
        nn.fit(xtr_NN, ytr_NN)
        score = roc_auc_score(yvl_NN, nn.predict(xvl_NN))
        print('ROC AUC score:',score)
        cv_score.append(score)    
        i+=1

    print('\nCROSS VALIDANTION SUMMARY:')
    print('Mean: ' + str(np.mean(cv_score)))
    print('Std deviation: ' + str(np.std(cv_score)))   

    print("\nTEST SET:")
    get_scores(Ytest, nn.predict(Xtest))

