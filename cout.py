# written by Emmanuel FERRET
# thanks to 


import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import pandas as pd

from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")


eco = ['modique','économique','ténu','japonais', 'chinois','kebab', 'self','dix','10','pourri','quinze','15','avantageux']
troquet = ['troquet','bistrot','cantine','self']
negation = ['pas','ni','peu']
cher = ['cher','bon','cent', 'cinquante','luxe','luxueux','fiche','moyens', 'gastronomique', 'classe','riche','michelin','etoile','étoiles',
'standing','haut','100', 'cinquante', 'vingt','20','chicos','onéreux','onereux']
monnaie = ['gamme', 'euro', 'prix', 'euros']


def ingest():
	dataset = pd.read_csv("RuleCout2.csv", delimiter=";", encoding='utf-8')
	print ( 'dataset loaded with shape',dataset.shape)
	X = dataset['text']
	Y = dataset['class']
	return dataset, X, Y

dataset, X, Y = ingest()
print(X)
#print(Y)
#dataset.head(5)

def score(word):
	if word in eco: 
		return 2 
	elif word in troquet: 
		return 3
	elif word in negation:
		return 4
	elif word in cher:
		return 5
	elif word in monnaie:
		return 6
	else: 
		return 1
	
print ( 'score modique', score('modique'))
print ( 'score troquet', score('troquet'))
print ( 'score inconnu', score('inconnu'))


def tokenize(sentence):
	tokens = tokenizer.tokenize(sentence)
	return tokens
	
print ("test de tokenize : ",tokenize("vraiement pas cher"))


def postprocess(data):
	data['tokens']= dataset['text'].progress_map(tokenize)
	return data

data = postprocess(dataset)
print(data)
print ( 'data shape : ',data.shape)

def token2code(tokens):
	return list(map(score,tokens))

def encode(data):
	data['codes']= data['tokens'].map(token2code)
	return data
		
X = encode(data)
print(X)
print ( 'data shape : ',X.shape)
X = pad_sequences(X['codes'])
print(X)


# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=5, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
	
seed = 7
numpy.random.seed(seed)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

