# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 08:16:05 2016

@author: raviteja
"""
import matplotlib.pyplot as plt
import numpy as np
def main():
    
    files  =["mvar-set1.dat.txt","mvar-set2.dat.txt","mvar-set3.dat.txt","mvar-set4.dat.txt"]
    file_to_use = input("enter the file to be used:")
    data = np.loadtxt(files[file_to_use-1])
    degree = input("enter the degree of the polynomial:")
  
    columns = data.shape[1]
    
    for i in range(columns-1):
       
        plt.scatter(data[:,i],data[:,columns-1])
        plt.show()
#    new_feature =  data[:,0]+data[:,1]
#    plt.scatter(new_feature ,data[:,columns-1])
#    plt.show()
    X = data[:,:columns-1]
    X_train =  X[0:X.shape[0]/2]
    X_test = X[X.shape[0]/2+1:]
    Y = data[:,columns-1]
    Y_train = Y[0:Y.shape[0]/2]
    Y_test = Y[Y.shape[0]/2+1:]
    
    Z_train= polynomial(X_train,degree) 
    
    Z_test = polynomial(X_test,degree)
    theta =  theta_computation(Z_train,Y_train)
    print theta
    Yt = prediction(theta,Z_test)
    difference = (Yt.T-Y_test)**2
    error = sum(difference)/len(Y_test)
    print error
    
    
def prediction(theta,Z):

      Yt =np.dot(theta.T,Z.T)
      return Yt  
   
        
def theta_computation(Z,Y):
    
     theta = np.dot(np.dot(np.linalg.inv(np.dot(Z.T,Z)),Z.T),Y)          
     return theta   
    
def polynomial(X,degree):
    
     z = np.ones((X.shape[0],1), dtype=np.int) 
     Z = np.append(z,X,axis=1)
     for i in range(degree-1):
        X =X*X
        Z =np.append(Z,X,axis=1) 
    
     return Z
    
        
if __name__ == "__main__":
    main()          
    