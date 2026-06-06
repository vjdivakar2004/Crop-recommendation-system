import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import joblib
import os


my_dir = os.path.dirname(__file__)

file_path = os.path.join(my_dir, 'Crop_recommendation.csv')


dataset=pd.read_csv(file_path)

X=dataset.iloc[:,[0,1,2,3,4,5,6]].values
Y=dataset.iloc[:,[7]].values

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=20)

model=KNeighborsClassifier()  # model is object of class KNeghborsClassifier



ytrain=np.ravel(ytrain)
model.fit(xtrain,ytrain)



joblib.dump(model,"knn_crop_model.pkl")
