import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def load_data():
    fruits = pd.read_table("Data/fruit_data_with_colors.txt")
    return fruits

def split_data(df):
    trainingFeatures = ['height', 'width', 'mass', 'color_score']
    trainingTargets = 'fruit_label'
    X = df[trainingFeatures]
    y = df[trainingTargets]
    Xtrain , Xtest, yTrain, yTest = train_test_split(X,y,random_state=0)
    return Xtrain , Xtest, yTrain, yTest

def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = X - mean
    X = X / std
    return X

def train_classifier(Xtrain,yTrain):
    knn = KNeighborsClassifier()
    knn.fit(Xtrain,yTrain)
    return knn

if __name__=="__main__":
    fruits=load_data()
    Xtrain , Xtest, yTrain, yTest = split_data(fruits)
    Xtrain = normalize_features(Xtrain)
    Xtest = normalize_features(Xtest)
    knn = train_classifier(Xtrain,yTrain)
    print(knn.score(Xtest,yTest))