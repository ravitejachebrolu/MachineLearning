# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 13:13:44 2016

@author: raviteja
"""
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import numpy as np
import csv
from math import log
from sklearn.metrics import confusion_matrix

def main():
     
     #print("Classification based on one feature")
     #loading the data
     X = []
     Y = []
     f=open("iris-dataset.csv")
     
     for row in csv.reader(f):
         
         X.append(row[0:4])
         Y.append(row[4])
     Y = np.asanyarray(Y)
     X = np.asanyarray(X).astype(np.float)
     Y = Y[:150]
     X = X[:150]
     X_cl1 = X[:30] 
     X_cl2 = X[70:]  
     X_cl3 = X[120:]
     X_test =np.concatenate( (X[30:50], X[50:70],X[100:120]),axis = 0)
     Y_test = np.concatenate( (Y[30:50], Y[50:70],Y[100:120]),axis = 0)
    
     mean_cl1,mean_cl2,mean_cl3= mean(X_cl1,X_cl2,X_cl3)
     print "mean of Iris-setosa"
     print mean_cl1
     
     print "mean of Iris-versicolor"
     print mean_cl2
     
     print "mean of Iris-virginica"
     print mean_cl3
     variance_cl1,variance_cl2,variance_cl3 = variance(X_cl1,X_cl2,X_cl3)
     covar_cl1,covar_cl2,covar_cl3 = covariance_matrix(X_cl1,X_cl2,X_cl3)
     print "covariance matrix of Iris-setosa"
     print covar_cl1
     
     print "covariance matrix of Iris-versicolor"
     print covar_cl2
     
     print "covariance matrix of Iris-virginica"
     print covar_cl3
     prediction= membership_func(mean_cl1,mean_cl2,mean_cl3,covar_cl1,covar_cl2,covar_cl3,X_test)
     #print prediction
     cfm= confusion_matrix(Y_test,prediction, labels=["Iris-setosa", "Iris-versicolor","Iris-virginica"])
     print "confusion matrix:" 
     print cfm
    
     accuracy = accuracy_score(Y_test,prediction)
     precision = precision_score(Y_test,prediction)
     recall = recall_score(Y_test,prediction)
     f1= f1_score(Y_test,prediction)
   
     print "precision: %s" %precision
     print "F-measure: %s" %f1
     print "accuracy: %s"  %accuracy 
     print "recall: %s" %recall

 
def mean(X_cl1,X_cl2,X_cl3):

      mean_cl1 = np.mean(X_cl1 , axis =0)
      mean_cl2 = np.mean(X_cl2,axis =0)
      mean_cl3 = np.mean(X_cl3 , axis =0)
      return  mean_cl1,mean_cl2,mean_cl3 
      
def variance(X_cl1,X_cl2,X_cl3):
     
     variance_cl1 = np.std(X_cl1,axis = 0)
     variance_cl2 = np.std(X_cl2,axis = 0)
     variance_cl3 = np.std(X_cl3,axis = 0)
     #print variance_cl1,variance_cl2
     return variance_cl1,variance_cl2,variance_cl3 
     

def covariance_matrix(X_cl1,X_cl2,X_cl3):
    
     covar_cl1 = np.cov(X_cl1.T)
     covar_cl2 = np.cov(X_cl2.T)
     covar_cl3 = np.cov(X_cl3.T)
     #print covar_cl1,covar_cl2     
     return covar_cl1,covar_cl2,covar_cl3     
    
def discrimination_func(g_cl1,g_cl2,g_cl3):
        Pred_class = []
        for i in range(0,len(g_cl1)):
                if g_cl1[i]>g_cl2[i] and g_cl1[i]>g_cl3[i]:
                        Pred_class.append("Iris-setosa")
                elif g_cl2[i]>g_cl1[i] and g_cl2[i]>g_cl3[i]: 
                        Pred_class.append("Iris-versicolor")
                elif g_cl3[i]>g_cl1[i] and g_cl3[i]>g_cl1[i]:
                        Pred_class.append("Iris-virginica")
    
        return Pred_class   
            
def membership_func(mean_cl1,mean_cl2,mean_cl3,covar_cl1,covar_cl2,covar_cl3,X_test):
       
     g_cl1 =[]
     g_cl2 = []
     g_cl3 =[]
     
     for i in range(0,len(X_test)):
           temp1 = np.dot(np.dot((X_test[i]-mean_cl1).transpose(),np.linalg.inv(covar_cl1)),X_test[i]-mean_cl1)
           temp2 = np.dot(np.dot((X_test[i]-mean_cl2).transpose(),np.linalg.inv(covar_cl2)),X_test[i]-mean_cl2)
           temp3 = np.dot(np.dot((X_test[i]-mean_cl3).transpose(),np.linalg.inv(covar_cl3)),X_test[i]-mean_cl3)
           g_cl1.append(-log(np.linalg.det(covar_cl1))-0.5*temp1+log(0.5))
           g_cl2.append(-log(np.linalg.det(covar_cl2))-0.5*temp2+log(0.5))
           g_cl3.append(-log(np.linalg.det(covar_cl3))-0.5*temp3+log(0.5))
     
     return discrimination_func(g_cl1,g_cl2,g_cl3)   
    

        
    
if __name__ == "__main__":
    main() 