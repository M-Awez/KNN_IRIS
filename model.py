import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report
import pickle
import warnings
warnings.filterwarnings("ignore")

class KNN:
  def __init__(self,path):
    self.df=pd.read_csv(path)
    self.df=self.df.drop(['Id'],axis=1)
    d={}
    for i in range(len(self.df['Species'].unique())):
      d[self.df['Species'].unique()[i]]=i
    self.df['Species']=self.df['Species'].map(d)
    self.X=self.df.iloc[:,0:4]
    self.y=self.df.iloc[:,-1]

  def split(self):
    X_train,X_test,y_train,y_test=train_test_split(self.X,self.y,train_size=0.8,random_state=42)
    return X_train,X_test,y_train,y_test

  def train(self,X_train,y_train):
    self.cla=KNeighborsClassifier()
    self.cla.fit(X_train,y_train)

  def prediction(self,a):
    prd=self.cla.predict(a)
    return prd

  def accuracy_confusion_report(self,a,b):
    print("-------------------")
    print(f"|Confusion Matrix:|")
    print("-------------------")
    print(confusion_matrix(a,b))
    print("------------------------")
    print(f"|Classification Report:|")
    print("------------------------")
    print(classification_report(a,b))

  def test_sample(self,a):
    print(self.cla.predict(a))

  def saving_file(self):
    with open("model.pkl",'wb') as f:
      pickle.dump(self.cla,f)

if __name__=="__main__":
  obj=KNN('Z:\ML Viharatech Projects\KNN(Classification)\Iris.csv')
  X_train,X_test,y_train,y_test=obj.split()
  obj.train(X_train,y_train)
  train_prd=obj.prediction(X_train)
  obj.accuracy_confusion_report(y_train,train_prd)
  print("------------------------------------------------------------")
  test_prd=obj.prediction(X_test)
  obj.accuracy_confusion_report(y_test,test_prd)
  obj.test_sample([[7.7,3,6.1,2.3]])
  obj.saving_file()
