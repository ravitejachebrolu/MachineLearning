# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 13:13:44 2016

@author: raviteja
"""
from sklearn.metrics import roc_curve, auc
import numpy as np
import csv
from math import log
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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
     Y = Y[:100]
     X = X[:100]
   
     X_cl1 = X[:30] 
     X_cl2 = X[70:]  
   
     X_test =np.concatenate( (X[30:50], X[50:70]),axis = 0)
     Y_test = np.concatenate( (Y[30:50], Y[50:70]),axis = 0)
    
     mean_cl1,mean_cl2 = mean(X_cl1,X_cl2)
     print "mean of Iris-setosa"
     print mean_cl1
     print "mean of versi-color"     
     print mean_cl2
     print "covariance matrix for Iris-setosa"
     
     variance_cl1,variance_cl2 = variance(X_cl1,X_cl2)
     covar_cl1,covar_cl2 = covariance_matrix(X_cl1,X_cl2)
     print covar_cl1
     print "covariance matrix for Iris-versicolor"     
     print covar_cl2
     prediction,bpc= membership_func(mean_cl1,mean_cl2,covar_cl1,covar_cl2,X_test)
     #print prediction
     cfm= confusion_matrix(Y_test,prediction, labels=["Iris-setosa", "Iris-versicolor"])
     print "confusion matrix:" 
     print cfm
     accuracy_func(cfm)   
     Fmeasure_func(precision_func(cfm),recall_func(cfm))
     fpr, tpr, thresholds = roc_curve(bpc,bpc)
     roc_auc = auc(fpr, tpr)
     print "area under curve:%s" %roc_auc
     plt.figure()
     plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
     plt.plot([0, 1], [0, 1], 'k--')
     plt.xlim([0.0, 1.0])
     plt.ylim([0.0, 1.05])
     plt.xlabel('False Positive Rate')
     plt.ylabel('True Positive Rate')
    
     plt.legend(loc="lower right")
     plt.show()
 
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
 
def mean(X_cl1,X_cl2):

      mean_cl1 = np.mean(X_cl1 , axis =0)
      mean_cl2 = np.mean(X_cl2,axis =0)
      
      return  mean_cl1,mean_cl2
      
def variance(X_cl1,X_cl2):
     
     variance_cl1 = np.std(X_cl1,axis = 0)
     variance_cl2 = np.std(X_cl2,axis = 0)
     #print variance_cl1,variance_cl2
     return variance_cl1,variance_cl2
     

def covariance_matrix(X_cl1,X_cl2):
    
     covar_cl1 = np.cov(X_cl1.T)
     covar_cl2 = np.cov(X_cl2.T)
     #print covar_cl1,covar_cl2     
     return covar_cl1,covar_cl2      
    
def discrimination_func(g_cl1,g_cl2):
       Pred_class = []
       Binary_pred_Class =[]
       for i in g_cl1:
             for j in g_cl2:
                     if i>j:
                           Pred_class.append( "Iris-setosa")
                           Binary_pred_Class.append(0)
                           break
                     else:
                           Pred_class.append("Iris-versicolor")
                           Binary_pred_Class.append(1)
                           break 
       return Pred_class ,Binary_pred_Class       
       
def membership_func(mean_cl1,mean_cl2,covar_cl1,covar_cl2,X_test):
       
     g_cl1 =[]
     g_cl2 = []
    
  
    
     for i in range(0,len(X_test)):
           temp1 = np.dot(np.dot((X_test[i]-mean_cl1).transpose(),np.linalg.inv(covar_cl1)),X_test[i]-mean_cl1)
           temp2 = np.dot(np.dot((X_test[i]-mean_cl2).transpose(),np.linalg.inv(covar_cl2)),X_test[i]-mean_cl2)
           g_cl1.append(-log(np.linalg.det(covar_cl1))-0.5*temp1+log(0.5))
           g_cl2.append(-log(np.linalg.det(covar_cl2))-0.5*temp2+log(0.5))
     
         #print Pred_class         
     return  discrimination_func(g_cl1,g_cl2)
        
    

        
    
if __name__ == "__main__":
    main() 