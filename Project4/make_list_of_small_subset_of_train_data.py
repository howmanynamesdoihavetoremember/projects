import numpy as np
import pandas as pd

import os

dataDir = 'state-farm-distracted-driver-detection/imgs'

imagePaths = []
imageClass = []

classNames = os.listdir(dataDir + '/train')
for className in classNames:
	counter = 0
	for imgName in os.listdir(dataDir + '/train/' + className):
		if counter < 100:
			fullPathToImage = dataDir + '/train/' + className+'/' + imgName
			imagePaths.append(fullPathToImage)
			imageClass.append(className)
		else:
			break
		counter += 1

imagePaths = np.array(imagePaths).reshape(-1, 1)
imageClass = np.array(imageClass).reshape(-1, 1)

smallFilesListing =  pd.DataFrame(np.concatenate((imagePaths, imageClass), axis=1), columns=['filePath', 'className'])

import sklearn

afterFirstShuffle = sklearn.utils.shuffle(smallFilesListing)
sklearn.utils.shuffle(afterFirstShuffle).to_csv('smallFilesListing.csv', index=False)


