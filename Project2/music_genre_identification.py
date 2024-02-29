import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import librosa




fileNames_genre = pd.read_csv('allFilesListing.csv')

y = pd.get_dummies(fileNames_genre['genre'] ).astype(int)

trainX , testX, trainY, ground_truth = train_test_split( fileNames_genre['filepath'] , y, test_size=0.2, random_state=43 )

trainX.reset_index(drop=True, inplace=True)
trainY.reset_index(drop=True, inplace=True)
testX.reset_index(drop=True, inplace=True)
ground_truth.reset_index(drop=True, inplace=True)

#----------------------------------------------- Read and process all filepaths in  (trainX, testX) ---------------------------------------

def processFiles(filesPaths):

	perFileAmp = []
	perFileFFT = []
	
	for filePath in filesPaths:

		audioAsRowVec = librosa.load(filePath, mono=True, sr=3000, duration=3.5)[0]
		
		fftAsColumnVec = np.abs(np.fft.fft(audioAsRowVec)).reshape(-1, 1)
		fftAsColumnVec = (fftAsColumnVec - np.mean(fftAsColumnVec) ) / np.std(fftAsColumnVec)
		perFileFFT.append(fftAsColumnVec)
		
		audioAsColumnVec = audioAsRowVec.reshape(-1, 1)
		audioAsColumnVec = (audioAsColumnVec - np.mean(audioAsColumnVec) ) / np.std(audioAsColumnVec)
		
		perFileAmp.append(audioAsColumnVec)


	perFileAmp = np.array(perFileAmp)
	perFileAmp = np.where( np.isnan(perFileAmp) , 0, perFileAmp ) # Replace nan with 0

	perFileFFT = np.array(perFileFFT)
	perFileFFT = np.where( np.isnan(perFileFFT) , 0, perFileFFT ) # Replace nan with 0


	return (perFileAmp, perFileFFT)


(trainAudio, trainFFT) = processFiles(trainX)
(testAudio,  testFFT)  = processFiles(testX)



from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Input, GlobalAveragePooling1D, AveragePooling1D, BatchNormalization, Dropout, UnitNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras import metrics

#------------------------------ Fit model on direct amplitude data -----------------------------------------------------------

model = Sequential()


model.add( Conv1D(10, kernel_size =40, strides=2, input_shape=[trainAudio.shape[1], 1] , activation='relu', padding='causal') )
model.add(BatchNormalization())
model.add( MaxPooling1D(pool_size=2) )


model.add( Conv1D(35, kernel_size =30, activation='relu', padding='causal') )

model.add(GlobalAveragePooling1D())

model.add( UnitNormalization() )

model.add( Dense(10, activation='softmax') )


optimizer = Adam(learning_rate=0.05)


model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[metrics.TrueNegatives()])


reduce_lr  =  ReduceLROnPlateau(monitor = 'val_loss',  factor = 0.5,  patience = 5,  min_lr = 2.0480e-04)

model.fit(trainAudio, trainY, epochs=50, batch_size=25, validation_data=[testAudio, ground_truth], callbacks=[reduce_lr])

probabiltyPredictionOfDirectModel = model.predict(testAudio)


#--------------------------------- Fit model on transformed data ------------------------------------------------------------

model = Sequential()

model.add( Conv1D(10, kernel_size =50, input_shape=[trainFFT.shape[1], 1] , activation='relu', padding='causal') )
model.add(BatchNormalization())
model.add( AveragePooling1D(pool_size=2) )

model.add(GlobalAveragePooling1D())

model.add( UnitNormalization() )

model.add( Dense(10, activation='softmax') )



optimizer = Adam(learning_rate=0.04)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[metrics.TrueNegatives()])


reduce_lr  =  ReduceLROnPlateau(monitor = 'val_loss',  factor = 0.5,  patience = 5,  min_lr = 2.0480e-04)

model.fit(trainFFT, trainY, epochs=55, batch_size=45, validation_data=[testFFT, ground_truth], callbacks=[reduce_lr])

probabiltyPredictionOfFFTModel = model.predict(testFFT)

#----------------------------- Take weighted mean of the two models predictions ------------------------------------------------

genreNamesInGroundTruth = pd.from_dummies(ground_truth)
genreIndexInGroundTruth = [list(ground_truth.columns).index(genreName) for genreName in genreNamesInGroundTruth.values ]


for weightOfFirst in np.linspace(0, 1, 20):

	probabilties = ( (probabiltyPredictionOfDirectModel*weightOfFirst) + (probabiltyPredictionOfFFTModel*(1-weightOfFirst)) ) / 2
	print('if weight of first is: ', weightOfFirst, 'then score is ', ( (np.argmax(probabilties, axis=1) == genreIndexInGroundTruth).sum() ) / 200)



