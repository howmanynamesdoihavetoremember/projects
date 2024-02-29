import cv2

import os

dataDir = 'state-farm-distracted-driver-detection/imgs'

fileNames = os.listdir(dataDir + '/test')
for imgName in fileNames:
	fullPathToImage = dataDir + '/test/' + imgName
	img = cv2.imread(fullPathToImage, cv2.IMREAD_REDUCED_GRAYSCALE_8)
	cv2.imwrite(fullPathToImage, img)



classNames = os.listdir(dataDir + '/train')
for className in classNames:
	for imgName in os.listdir(dataDir + '/train/' + className):
		fullPathToImage = dataDir + '/train/' + className+'/' + imgName
		img = cv2.imread(fullPathToImage, cv2.IMREAD_REDUCED_GRAYSCALE_8)
		cv2.imwrite(fullPathToImage, img)
