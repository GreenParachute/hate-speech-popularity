#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import numpy as np
import pandas as pd
from sklearn import svm, preprocessing, linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', default='features.csv', type=str, help="Name or path of input data file (be it features or corpus")
parser.add_argument('--features', default='nlp', type=str, help="Features you wish to train on. Options: nlp or ngram. Default: nlp")
parser.add_argument('--model', default='reg', type=str, help="Machine learning model you wish to use. Options: reg or svm. Default: reg")
args = parser.parse_args()

def build_model(arg):

	if arg == 'reg':
		clf = linear_model.LogisticRegression(C=1.0)
		print("Logistic regression model built...")
	elif arg == 'svm':
		clf = svm.SVC(kernel="linear", C=1.0)
		print("SVM model built...")
	else:
		print("Model argument invalid (Options: reg or svm")
	return clf

if args.features == 'nlp':

	#import csv with data
	nlp_feature_data = pd.read_csv(args.input, header=0) # to import all extracted features (binarized)
	print("Data imported...")

	#these are our features that we use for prediction; not included in the features.csv file, but obtainable from API: ['is_favorited','is_retweeted','num_fav_corpus', 'num_RT_corpus', 'num_replies', 'is_reply', 'is_reply_to_hate_tweet', 'is_quote_status', 'num_followers', 'num_followees', 'num_times_user_was_listed', 'num_posted_tweets', 'num_favorited_tweets', ]
	features=['has_mentions', 'num_mentions', 'has_hashtags', 'num_hashtags', 'has_urls', 'num_urls', 'total_positive_tokens', 'positive_token_ratio', 'total_negative_tokens', 'negative_token_ratio', 'total_subjective_tokens', 'subjective_token_ratio', 'blacklist_total', 'blacklist_ratio', 'char_count', 'token_count', 'has_uppercase_token', 'uppercase_token_ratio', 'lowercase_token_ratio', 'mixedcase_token_ratio', 'has_digits', 'has_questionmark', 'has_fullstop', 'has_exclamationpoint', 'len_handle', 'len_name', 'account_age', 'tweet_age', 'tweet_hour']

	#main pandas data frame from which the machine learning will happen 
	df_main = pd.DataFrame({'is_hate_tweet':nlp_feature_data['is_hate_tweet']})	

	#extracting and scaling the nlp features
	X_nlp = np.array(nlp_feature_data[features].values)
	X_nlp = preprocessing.scale(X_nlp)
	print('NLP features scaled...')


	#defining independent (X) and dependent (y) variables
	y = df_main["is_hate_tweet"].values.tolist() #options: is_hate_tweet, is_favorited, is_retweeted, #favorited_class #retweeted_class

	X = X_nlp #this is for JUST nlp features

	print('Variables defined...')

	#building the chosen model
	clf = build_model(args.model)

elif args.features == 'ngram':
	ngram_train_data = pd.read_csv(args.input, sep='\t', header=0) # to import actual tweet text obtained through API
	print("Data imported...")

	#main pandas data frame from which the machine learning will happen 
	df_main = pd.DataFrame({'is_hate_tweet':ngram_train_data['is_hate_tweet']})	

	#extracting and vectorizing tweets, breaking them down to character n-grams and calculating tf-idf scores
	tfidf = TfidfVectorizer(analyzer = "char", ngram_range=(2,4))
	X_ngrams = tfidf.fit_transform(ngram_train_data['tweet_text'])
	#adding the n-gram feature vector into the main pandas data frame
	df_main['tweetsVect']=list(X_ngrams.toarray())
	print('N-grams extracted...')

	#defining independent (X) and dependent (y) variables
	y = df_main["is_hate_tweet"].values.tolist() #options: is_hate_tweet, is_favorited, is_retweeted, #favorited_class #retweeted_class

	X = np.array(df_main['tweetsVect'].values).tolist() #this is JUST for ngram features

	print('Variables defined...')

	#building the chosen model
	clf = build_model(args.model)

else:
	print("Feature argument invalid (Options: nlp or ngram)")


#evaluating the chosen model

print("Running classification report...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)	
	
clf.fit(X_train,y_train)
	
y_pred = []
y_true = []
target_names = ["non_hate_tweet","hate_tweet"]
labels = [False,True]

for x in range(1, len(X_test)+1):
	prediction=clf.predict(X[-x])
	y_pred.append(prediction[0])
	y_true.append(y[-x])

print(classification_report(y_true, y_pred, labels=labels, target_names=target_names))

#other
a_scores = cross_val_score(clf, X, y, cv=10)
print("Cross validation accuracy scores:", a_scores)
print("Accuracy: %0.2f np (+/- %0.2f)" % (a_scores.mean(), a_scores.std() * 2))
print("...")
f1_scores = cross_val_score(clf, X, y, cv=10, scoring='f1')
print("Cross validation F1 scores:", f1_scores)
print("F1-score: %0.2f (+/- %0.2f)" % (f1_scores.mean(), f1_scores.std() * 2))
