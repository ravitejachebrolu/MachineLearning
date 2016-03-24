# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 11:10:50 2016

@author: raviteja
"""
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
            X.append(row[3])
            Y.append(row[4])

     X = [float(i) for i in X]
     X = X[:100]
     Y=Y[:100]
     mean_cl1,mean_cl2 =  mean(X)
     var_cl1,var_cl2 = variance(X)
     X_test = X[21:50]+X[51:80]
#     print mean_cl1,mean_cl2,var_cl1,var_cl2
#     print X_test
     Y_test = Y[21:50]+Y[51:80]
    
     Pred_class = membership_func(X_test,mean_cl1,mean_cl2,var_cl1,var_cl2)
     cfm= confusion_matrix(Y_test,Pred_class, labels=["Iris-setosa", "Iris-versicolor"])
     print "confusion matrix:" 
     print cfm
     accuracy_func(cfm)   
     Fmeasure_func(precision_func(cfm),recall_func(cfm))

def precision_func(cfm):
            
     precision = cfm[0][0]/(cfm[0][0]+cfm[0][1])
     print "precision: %s" %precision
     return precision
     
def recall_func(cfm):
        
        recall = cfm[0][0]/(cfm[0][0]+cfm[1][0])
        print "recall: %s" %recall
        return recall
        
def Fmeasure_func(precision,recall):
     
        Fmeasure = 2*precision*recall/(precision+recall)
        print "F-measure: %s" %Fmeasure
    
def accuracy_func(cfm):
    
     accuracy = (cfm[0][0]+cfm[1][1])/(cfm[0][0]+cfm[1][0]+cfm[0][1]+cfm[1][1])      
     print "accuracy: %s"  %accuracy
     
#discirmination function , a condition for memebership function     
def discrimination_func(g_cl1,g_cl2):
       Pred_class =[]
       for i in g_cl1:
             for j in g_cl2:
                     if i>j:
                           Pred_class.append( "Iris-setosa")
                           break
                     else:
                           Pred_class.append("Iris-versicolor")
                           break
       return Pred_class
#membership_func to say which test data belongs to which class
def membership_func(X_test,mean_cl1,mean_cl2,var_cl1,var_cl2):
    
        
        
        
         g_cl1 = -log(var_cl1)-((0.5)*((X_test-mean_cl1)**2/var_cl1**2))+log(0.5)  
         g_cl2 = -log(var_cl2)-((0.5)*((X_test-mean_cl2)**2/var_cl2**2))+log(0.5)
         #print g_cl1,g_cl2
         
         #print Pred_class         
         return discrimination_func(g_cl1,g_cl2)       
def variance(X):
    
      
    var_cl1 = np.asanyarray(X[0:20]) 
    var_cl2 = np.asanyarray(X[81:100])
    print "variance"
    print np.std(var_cl1),np.std(var_cl2)
    return np.std(var_cl1),np.std(var_cl2)
    
def mean(X):
       
    mean_cl1 = np.asanyarray(X[0:20])
    mean_cl2=  np.asanyarray(X[81:100])
    print "mean"
    print np.mean(mean_cl1),np.mean(mean_cl2)
         
    return np.mean(mean_cl1),np.mean(mean_cl2)
         
    
     

if __name__ == "__main__":
    main() 
