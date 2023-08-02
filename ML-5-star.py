import sys, os, re, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.utils import resample 

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline



nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

dir_data = "yelp_academic_dataset_review.json"


# Vectorizer for extracting features 
def extract_features(df_train, df_test):
    vectorizer = TfidfVectorizer(analyzer = 'word', stop_words = 'english', ngram_range=(1,1), lowercase=True)
    X_train = vectorizer.fit_transform(df_train["text"])
    X_test = vectorizer.transform(df_test["text"])
    y_train = df_train["stars"].tolist()
    y_test = df_test["stars"].tolist()
    
    return X_train, X_test, y_train, y_test

# Undersample function to get the same number of samples from each star rating
def undersample(df, group_size = 400000): 
    dfs = []

    for label in df["stars"].value_counts().keys():
        df_group = df[df["stars"] == label]
        df_group_undersampled = resample(df_group, replace = False, n_samples=group_size, random_state=0)
        dfs.append(df_group_undersampled)
  
    return pd.concat(dfs).sample(frac=1, random_state=0)


# Plotting distribution of labels for stars for checking their distribution
def plot_labels(df, title=None):
    ds_labels = df["stars"].value_counts(normalize=True)
    ds_labels.sort_index(inplace=True)
    plt.figure(figsize=(4,3))
    ax = ds_labels.plot(kind="bar")
    ax.set_xlabel("Stars")
    ax.set_ylabel("Ratio")
    if title is not None:
      plt.savefig(title + ".eps")
    plt.show()

# Reading in the data by chunks and putting it into DataFrame named df_review
counter = 0 
recorder = []

for chunk in pd.read_json("yelp_academic_dataset_review.json", lines=True, chunksize=1000): 
    recorder.append(chunk)
    
    ''' #For smaller sized chunks to test code
    
    if counter == 10: 
        break
    '''
    counter += 1
    if counter % 1000 == 0: 
        print(counter)

df_review = pd.concat(recorder)
print("df_review made")

# Balanced dataset 
df_train_raw, df_test = train_test_split(df_review, test_size=500000, random_state=42, shuffle=True)

df_train = undersample(df_train_raw) # Resampled training set with 500'000 samples per star
print("df_train undersampling complete")

''' # use graph to check distribution of raw training data and resampled data
plot_labels(df_train_raw)
print(df_review["stars"].value_counts(normalize=False))

plot_labels(df_train)
print(df_train["stars"].value_counts(normalize=False))

df_train["stars"].value_counts(normalize=False)
'''

X_train, X_test, y_train, y_test = extract_features(df_train, df_test)
print("Extracted features complete")


# Define a function to evaluate a model 

def evaluate_model(model, X, y, y_pred=None, label="Training", model_name="model"): 
    if y_pred is None: 
        y_pred = model.predict(X) 
    
    # Accuracy + classification report
    print(label + ' Set')
    print("Accuracy: ", accuracy_score(y, y_pred), "\n")
    print(classification_report(y, y_pred, digits=4))
    
    # Confusion matrix - count is just displayed
    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=model.classes_)
    disp.plot()
    plt.show() 
    plt.savefig(model_name+"_"+label.lower() + ".eps")
    
    # Confusion matrix - normalized
    cm_normalize = confusion_matrix(y, y_pred, labels=model.classes_, normalize="all")
    disp = ConfusionMatrixDisplay(confusion_matrix = cm_normalize, display_labels=model.classes_)
    disp.plot()
    plt.show() 
    plt.savefig(model_name+"_"+label.lower() +"normalized" + ".eps")



# Evaluate NAIVE BAYES model on training, validation and testing
classifier_nb = MultinomialNB(alpha=0.5, fit_prior=True)
classifier_nb.fit(X_train, y_train) 
evaluate_model(classifier_nb, X_train, y_train, label="Training", model_name="naive bayes training")
evaluate_model(classifier_nb, X_test, y_test, label="Testing", model_name="naive bayes testing")


# LOGISTIC REGRESSION 
classifier_lr = LogisticRegression(penalty="l2", tol=0.0004, C=1.0, fit_intercept=True,
                            class_weight="balanced", random_state=0, solver="lbfgs",
                            max_iter=100, multi_class="auto", verbose=1, n_jobs=-1)
classifier_lr.fit(X_train, y_train)
evaluate_model(classifier_lr, X_train, y_train, label="Training", model_name="logistic regression training")
evaluate_model(classifier_lr, X_test, y_test, label="Testing", model_name="logistic regression testing")


# RANDOM FOREST 
classifier_rf = RandomForestClassifier(n_estimators=200, criterion="gini", max_depth=None,
                                       min_samples_split=2, min_samples_leaf=10,
                                       n_jobs=-1, verbose=1, random_state=0,
                                       class_weight="balanced")
classifier_rf.fit(X_train, y_train)
evaluate_model(classifier_rf, X_train, y_train, label="Training", model_name="random forest training")
evaluate_model(classifier_rf, X_test, y_test, label="Testing", model_name="random forest testing")


# SVM 
classifier_svm = make_pipeline(StandardScaler(with_mean=False),
                               SGDClassifier(loss='hinge', penalty='l2', alpha=30,
                                      max_iter=1000, tol=1e-3,shuffle=True,verbose=1,
                                      n_jobs=-1,random_state=0, learning_rate='optimal',
                                      early_stopping=True, class_weight='balanced'))
classifier_svm.fit(X_train, y_train)
evaluate_model(classifier_svm, X_train, y_train, label="Training", model_name="random forest training")
evaluate_model(classifier_svm, X_test, y_test, label="Testing", model_name="random forest testing")


# BERT 
























