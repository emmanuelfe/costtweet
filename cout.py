# written by Emmanuel FERRET
# thanks to 

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

pd.set_option('display.width', 1000)

from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")


eco = ['modique','économique','ténu','japonais', 'chinois','kebab', 'self','dix','10','pourri','quinze','15','avantageux','marché','bas']
troquet = ['troquet','bistrot','cantine','self']
negation = ['pas','ni','peu']
cher = ['cher','bon','cent', 'haut','cinquante','luxe','luxueux','fiche','fous','moyens', 'gastronomique', 'classe','riche','michelin','etoile','étoiles',
'standing','haut','100','cent', 'cinquante', 'vingt','20','chicos','onéreux','onereux','super','ternte','30','meilleur','meilleurs','grattin','gratin']
monnaie = ['gamme', 'euro', 'prix', 'euros','problème','pb','qualité']
ambigu = ['milieu','moyen']


def ingest():
	dataset = pd.read_csv("RuleCout2.csv", delimiter=";", encoding='utf-8')
	print ( 'dataset loaded with shape',dataset.shape)
	X = dataset['text']
	Y = dataset['class']
	dataset.set_index('text') #pour avoir les classes dans un ordre quelconque
	return dataset, X, Y

dataset, X, Y = ingest()
print("dataset résultat du read.csv",dataset)
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
	elif word in ambigu:
		return 7
	else: 
		return 0
	
#print ( 'score modique', score('modique'))
#print ( 'score troquet', score('troquet'))
#print ( 'score inconnu', score('inconnu'))


def tokenize(sentence):
	tokens = tokenizer.tokenize(sentence)
	return tokens
	
#print ("test de tokenize : ",tokenize("vraiement pas cher"))


def postprocess(data):
	data['tokens']= dataset['text'].progress_map(tokenize)
	return data

dataset = postprocess(dataset)
print("dataset readcsv + tokens",dataset )
#print ( 'data shape : ',data.shape)

def token2code(tokens):
	return list(map(score,tokens))

def encode(data):
	data['Xcodes']= data['tokens'].map(token2code)
	return data
		
dataset = encode(dataset)

print("dataset text+class+tokens+Xcodes",dataset)
print ( 'data shape : ',dataset.shape)
XEncoded = pad_sequences(dataset['Xcodes'])
#print(XEncoded)


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
ycodes = pd.Series(list(dummy_y))
dataset['Ycodes'] = ycodes

print("dataset text class tokens Xcodes Ycodes",dataset)

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(11, input_dim=5, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model = baseline_model()
	
estimator = KerasClassifier(build_fn=baseline_model, epochs=300, batch_size=5, verbose=0)
	
seed = 7
np.random.seed(seed)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

result = model.predict(XEncoded)
dataset['predicted'] = pd.Series(list(result)) 

print(dataset)
results = cross_val_score(estimator, XEncoded, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
