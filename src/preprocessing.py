import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from Linear_Re_for_Preprocessing import LinearRegression
import json

path='/home/jack/Machine-Learning/ml_Problems/KNN/film_recommendation/movie.csv'
data=pd.read_csv(path)

#--------------------------------------------

def makeSet(column_str_list):
	words=column_str_list.values
	all_words=[re.split('\s|,',x) for x in words] # all words in 2D list
	all_words_1D=[y for x in all_words for y in x]  # all words in 1D lsit(repeated counted)
	my_words = {i:all_words_1D.count(i) for i in all_words_1D}.keys()
	return my_words, all_words


# to get set of genres
def getGen(column_str=data['genres']):
	column_str = column_str.str.strip('[]').str.replace(' ','').str.replace("'",'')
	return makeSet(column_str)


# to get set of names
def getName(column_str=data['name']):
	#column_str=column_str['name']
	# all_names=[x.split(' ') for x in column_str]
	# names=[y for y in x for x in all_names]
	# return names,all_names
	return makeSet(column_str)


# def writeJson(path,my_dict):
# 	with open(path, "w") as f:
#     	json.dump(my_dict, f, indent=4)


# encode 0_1
def encode_str(my_words, all_words):
	encoding_genres=[]
	gen_num= len(list(my_words))
	for x in all_words:
	  y=[0]*gen_num
	  for i in range (gen_num):
	    if list(my_words)[i] in x:
	      y[i]=1  
	  encoding_genres.append(y)

	return np.asarray(encoding_genres)


def decoding(gen_list,my_words):  # my_words should be getGen(data['genres'])[0]:)) 
  gen_l=[]
  for i in range(len(my_words)):
    if gen_list[i]==1:
      gen_l.append(my_words[i])

  return gen_l


def encoded_dataframe(data_=data):
	
	genre,all_gens=getGen()
	encoding_genres=encode_str(genre,all_gens)
	#data_=data_.drop(columns=['revenue', 'budget','Unnamed: 0','genres','popularity','vote_counts'])
	#print(encoding_genres.T)
	for i,x in enumerate( encoding_genres.T):
	  #print(x.shape)
	  #for i in range(x.shape[0]):
		data_[ list(genre)[i] ] = x

	return data_



def handle_nan_AND_minmax_scaler(data_,columns):
	for column in columns:
		nan_handled_data=data_[column].fillna(0).to_numpy() # replace nan by 0
		data_=data_.drop([column],axis=1)
		min=np.min(nan_handled_data)
		max=np.max(nan_handled_data)
		data_[column]=np.array([(x-min)/(max-min) for x in nan_handled_data])
	return data_
		

# def minmax_scaler(_l,data_):
# 	min=np.min(_l)
# 	max=np.max(_l)
# 	return np.array([(x-min)/(max-min) for x in _l])


def handle_missing_revenue_value(data):

	data_= data.drop(['name','vote_counts','runtime','budget','genres','Unnamed: 0'],axis=1)

	def split_train_test(data_):
		train=data_[ data['revenue'] != 0]
		X_train=train.iloc[:,1:].to_numpy()
		y_train=train.iloc[:,0].to_numpy()
		test=data_[ data['revenue'] == 0]
		X_test=test.iloc[:,1:].to_numpy()
		#y_test=test.iloc[:,0].to_numpy() # 0:))
		return X_train,y_train,X_test

	def main_f(X_train,y_train,X_test,data_):
		model = LinearRegression(0.001, 10,"SGD")
		model.fit(X_train, y_train)
		y_test=model.predict(X_test)
		data_.replace({'revenue': 0}, y_test)
		return data_

	return main_f(split_train_test(data_),data_)


