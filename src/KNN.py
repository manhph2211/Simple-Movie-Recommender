import json
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np
from preprocessing import *



movie_names=data['name'].to_numpy()
runtime=data['runtime'].to_numpy()


def readJson(path):
	with open(path, "r") as f:
	  	genres_dic=json.load(f)
	  	return genres_dic
		

def recommend_using_sklearn(data_fr,test_id):
	data_=data_fr.iloc[:,1:]
	data_matrix = csr_matrix(data_.values)
	model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
	model_knn.fit(data_matrix)
	distances, indices = model_knn.kneighbors(data_.iloc[test_id,:].values.reshape(1, -1), n_neighbors = 6)
	genres=list(readJson('../genres.json').keys()) 
	print()
	for i in range(0, len(distances.flatten())):
	    if i == 0: # itself
	        print('Recommendations(using sklearn) for {0} - Genres = {1} - Runtime = {2}: \n'.format(movie_names[test_id], decoding(data_fr.iloc[test_id,2:-1].to_numpy(),genres), runtime[indices.flatten()[i]]))
	    else:
	        print('{0}: {1} - Genres =  {2} - Runtime = {3}'.format(i, movie_names[indices.flatten()[i]],decoding(data_fr.iloc[indices.flatten()[i],2:-1].to_numpy(),genres),runtime[indices.flatten()[i]]))

  

def recommend_mycode(data_fr,test_id):
	X=data_fr.iloc[:,1:].to_numpy()


	def cosine_simi(X1,X2):
		weights=[1]*X1.shape[0]
		X1=X1*weights
		X2=X2*weights
		return np.sum(X1*X2)/np.sqrt(np.sum(X1**2))/np.sqrt(np.sum(X2**2))

	distances={}
	for i in range(len(data_fr.index)):
		if i != test_id:
			distances[i]=cosine_simi(X[i],X[test_id])
	distances={k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
	index_l=list(distances.keys())[994:]
	genres=list(readJson('../genres.json').keys()) 
	print('Recommendations(using mycode) for {0} - Genres = {1} - Runtime = {2}: \n'.format(movie_names[test_id], decoding(data_fr.iloc[test_id,2:-1].to_numpy(),genres), runtime[test_id]))
	for i in range(len(index_l)):
		print('{0}: {1} - Genres =  {2} - Runtime = {3}'.format(i+1, movie_names[index_l[i]],decoding(data_fr.iloc[index_l[i],2:-1].to_numpy(),genres),runtime[index_l[i] ] ))


