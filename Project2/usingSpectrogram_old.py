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
spectogramShape = (0, 0)


def processFiles(filesPaths):

	perFileSpectogram = []
	
	global spectogramShape
	
	for filePath in filesPaths:

		audioAsRowVec = librosa.load(filePath, mono=True, sr=1700, duration=4)[0]
		
		spectogram = librosa.feature.melspectrogram(y=audioAsRowVec, sr=3000)
		

		spectogram = (spectogram - np.mean(spectogram) ) / np.std(spectogram)
		spectogramShape = spectogram.shape
		
		perFileSpectogram.append(spectogram.reshape(-1, 1))
	
	perFileSpectogram = np.array(perFileSpectogram)
	perFileSpectogram = perFileSpectogram.reshape( (len(filesPaths), spectogramShape[0], spectogramShape[1], 1))

	return perFileSpectogram


trainAudio = processFiles(trainX)
testAudio  = processFiles(testX)



from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, GlobalAveragePooling2D, AveragePooling2D, BatchNormalization, Dropout, UnitNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau

#--------------------------------- Fit model on transformed data ------------------------------------------------------------

model = Sequential()

model.add( Conv2D(20, kernel_size =(13, 13), input_shape=[spectogramShape[0], spectogramShape[1], 1] , activation='relu', padding='same') )
model.add(BatchNormalization())
model.add( MaxPooling2D(pool_size=4) )


model.add( Conv2D(35, kernel_size =(4, 4), activation='relu', padding='same') )
model.add(BatchNormalization())
model.add(Flatten())


model.add( Dense(30, activation='relu') )

model.add( UnitNormalization() )
model.add( Dense(10, activation='softmax') )


from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.09)

from keras import metrics
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()



model.fit(trainAudio, trainY, epochs=20, batch_size=35, validation_data=[testAudio, ground_truth])

probabilties = model.predict(testAudio)



genreNamesInGroundTruth = pd.from_dummies(ground_truth)
genreIndexInGroundTruth = [list(ground_truth.columns).index(genreName) for genreName in genreNamesInGroundTruth.values ]


print( ( (np.argmax(probabilties, axis=1) == genreIndexInGroundTruth).sum() ) / len(ground_truth))



