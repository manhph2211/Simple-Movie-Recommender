from preprocessing import *
from KNN import *



def main():
	data_fr=encoded_dataframe(data)
	data_fr=data_fr.drop(['genres','Unnamed: 0','vote_counts','revenue','budget'],axis=1)
	data_fr=handle_nan_AND_minmax_scaler(data_fr,['runtime','popularity'])
	test_id = np.random.choice(1000)
	 
	recommend_mycode(data_fr,test_id)	

	recommend_using_sklearn(data_fr,test_id)

if __name__=='__main__':
	main()
