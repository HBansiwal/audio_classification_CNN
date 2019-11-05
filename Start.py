#import dependancy
import librosa
import librosa.display
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import os
import pandas as pd
import glob 
import scipy.io.wavfile
from sklearn.utils import shuffle
from math import sqrt
import sys
import keras
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--test", required=True, help="path to test folder")
args = vars(ap.parse_args())

# loading json file and creating model
from keras.models import model_from_json
opt = keras.optimizers.Adam( beta_1=0.9, beta_2=0.999, amsgrad=False)# define optimizer
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")#loading weights
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print("MODEL LOADED")
print(args["test"])
# for i in os.listdir(args["test"]):
# 	print(type(i))
# 	print(i)
path = args["test"]
f= open("Prediction.txt","w+")
#[Extracting the feature from >wav file and convert into vectors form]
for i in os.listdir(args["test"]):
	I=[]
	M = pd.DataFrame(columns=['feature'])
	X, sample_rate = librosa.load(path + '/'+i, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
	sample_rate = np.array(sample_rate)
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
	featurelive = mfccs
	M.loc[0] = [featurelive]
	M = pd.DataFrame(M['feature'].values.tolist())
	M=M.fillna(0)
	u=featurelive.shape[0]
	if (u<216):
	    for i in range(0,1):
	        I.append(0)
	I = pd.DataFrame(I)
	for i in range(0,(216-u)):
	    M = pd.concat([M,I], axis=1)
	livedf2 = M
	livedf2= pd.DataFrame(data=livedf2)
	livedf2 = livedf2.stack().to_frame().T
	twodim= np.expand_dims(livedf2, axis=2)
	livepreds = loaded_model.predict(twodim, batch_size=32, verbose=1)
	livepreds1=livepreds.argmax(axis=1)
	liveabc = livepreds1.astype(int).flatten()
	livepredictions = (lb.inverse_transform((liveabc)))
	print(livepredictions.tolist())
	f=open("Prediction.txt", "a+")
	for i in (livepredictions.tolist()):
		f.write(i)
	f.write( '\r\n')
