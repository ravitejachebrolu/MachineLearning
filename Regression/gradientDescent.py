# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 23:27:15 2016

@author: raviteja
"""
import numpy as np
def gradientDescent(x, y, theta, alpha, m, iter):
    xT = x.T
    for i in range(0, iter):
        theta = theta - alpha * (np.dot(xT, np.dot(x, theta) - y) / m )
    return theta
 
 
def main():
    
    files  =["mvar-set1.dat.txt","mvar-set2.dat.txt","mvar-set3.dat.txt","mvar-set4.dat.txt"]
    file_to_use = input("enter the file to be used:")
    data = np.loadtxt(files[file_to_use-1])
    #degree = input("enter the degree of the polynomial:")
    columns = data.shape[1]
    X = data[:,:columns-1]
    Y = data[:,columns-1]
    z = np.ones((X.shape[0],1), dtype=np.int) 
    Z = np.append(z,X,axis=1)
    theta = np.ones(Z.shape[1])
    alpha  = 0.03
    iters = 10000
    m = X.shape[0]
    theta =gradientDescent( Z, Y, theta, alpha, m, iters )
    
    
    Yt = prediction(theta,Z)
    difference = (Yt.T-Y)**2
    error = sum(difference)/len(Y)
    print error
def prediction(theta,Z):

      Yt =np.dot(theta.T,Z.T)
      return Yt      
        
if __name__ == "__main__":
    main()       