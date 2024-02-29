import numpy as np
import pandas as pd

WIDTH = 80
HEIGHT = 60


fileNames_class = pd.read_csv('smallFilesListing.csv')

y = pd.get_dummies(fileNames_class['className'] ).astype(int)

from sklearn.model_selection import train_test_split
trainX , testX, trainY, ground_truth = train_test_split( fileNames_class['filePath'] , y, test_size=0.2, random_state=554 )

trainX.reset_index(drop=True, inplace=True)
trainY.reset_index(drop=True, inplace=True)
testX.reset_index(drop=True, inplace=True)
ground_truth.reset_index(drop=True, inplace=True)


#------------------------------------------------------ Read Files ---------------------------------------------------------------------


import cv2

def processFiles(filePaths):
	
	images = []
	
	for filePath in filePaths:
		img = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
		img  = (img - np.mean(img)) / np.std(img)
		images.append(img.reshape(-1, 1))
	
	images = np.array(images)
	images = images.reshape( (len(filePaths), HEIGHT, WIDTH, 1))
	
	return images


trainImages = processFiles(trainX)
testImages = processFiles(testX)

#-------------------------------------------------- Create and compile model and predict ---------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, GlobalAveragePooling2D, AveragePooling1D, BatchNormalization, Dropout, UnitNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau


model = Sequential()

model.add( Conv2D(10, kernel_size =(13, 13), input_shape=[HEIGHT, WIDTH, 1] , activation='relu', padding='same') )
model.add(BatchNormalization())
model.add( MaxPooling2D(pool_size=2) )


model.add( Conv2D(35, kernel_size =(4, 4), activation='relu', padding='same') )
model.add(BatchNormalization())
model.add(Flatten())


model.add( Dense(20, activation='relu') )

model.add( UnitNormalization() )
model.add( Dense(10, activation='softmax') )


from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.05)

from keras import metrics
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


model.fit(trainImages, trainY, epochs=20, batch_size=25, validation_data=[testImages, ground_truth])

probabilties = model.predict(testImages)

#----------------------------------------------------------------- Calculate Score -------------------------------------------------

classNamesInGroundTruth = pd.from_dummies(ground_truth)
classIndexInGroundTruth = [list(ground_truth.columns).index(className) for className in classNamesInGroundTruth.values ]

print( ( (np.argmax(probabilties, axis=1) == classIndexInGroundTruth).sum() ) / 200)


