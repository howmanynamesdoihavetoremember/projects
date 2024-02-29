import os
import pandas as pd
import sklearn

projectDataPath = "genres/"

filePath_genre = []

for genreFolder in os.scandir(projectDataPath):
    generePath = projectDataPath + genreFolder.name  + '/'
    
    for files in os.scandir(generePath):
        filePath = projectDataPath + genreFolder.name + '/' + files.name
        filePath_genre.append([filePath, genreFolder.name])

df = pd.DataFrame(filePath_genre, columns=["filepath", "genre"])

afterFirstShuffle = sklearn.utils.shuffle(df)
sklearn.utils.shuffle(afterFirstShuffle).to_csv('allFilesListing.csv', index=False)
