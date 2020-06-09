from skExSTraCS import ExSTraCS
import pandas as pd
import numpy as np
import pickle

data = pd.read_csv('/Users/robert/Desktop/vizData/Multiplexer20Modified.csv')
dataFeatures = data.drop('Class',axis=1).values #DEFINE classLabel variable as the Str at the top of your dataset's action column
dataPhenotypes = data['Class'].values

#Shuffle Data Before CV
formatted = np.insert(dataFeatures,dataFeatures.shape[1],dataPhenotypes,1)
np.random.shuffle(formatted)
dataFeatures = np.delete(formatted,-1,axis=1)
dataPhenotypes = formatted[:,-1]

#Initialize ExSTraCS Model
model = ExSTraCS(learning_iterations = 10000,nu=10,N=2000,rule_compaction=None)
model.fit(dataFeatures,dataPhenotypes)
outfile = open('/Users/robert/Desktop/vizData/ModelMP20', 'wb')
pickle.dump(model, outfile)
outfile.close()
