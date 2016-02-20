# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 10:42:43 2016

@author: raviteja
"""
import matplotlib.pyplot as plt
import numpy as np

# this function is used to find the mean square error
def mse(Z,theta,Y):
      
    Yt =prediction(theta,Z) # this line takes to prediction function 
    difference = (Yt.T-Y)**2
    error = sum(difference)/len(Y)
    return error
#this function is used to generate polynomial of higher degree and add it to the feature vector    
def polynomial(X,degree):
    
     z_train = np.ones((X.size,1), dtype=np.int)
     Z_train = np.append(z_train,X,axis =1)
     for i in range(degree-1):
        X =X*X
        Z_train =np.append(Z_train,X,axis=1) 
    
     return Z_train
    
# theta values are computated in this function    
def theta_computation(Z,Y):
    
     theta = np.dot(np.dot(np.linalg.inv(np.dot(Z.T,Z)),Z.T),Y)          
     return theta
  
#this function is for polynomial and cross_validation    
def polynomial_withCV(X,Y,degree,X_test):
   
    z_test = np.ones((X_test.size,1), dtype=np.int)
    Z_test = np.append(z_test,X_test,axis =1)
    z_train = np.ones((X.size,1), dtype=np.int)
    Z_train = np.append(z_train,X,axis =1)
    for i in range(degree-1):
        X =X*X
        X_test = X_test*X_test
        Z_train =np.append(Z_train,X,axis=1) 
        Z_test =np.append(Z_test,X_test,axis=1) 
    theta = theta_computation(Z_train,Y)
    return Z_train,theta,Z_test
#this function to plot sample data
def plot_sample_data(X,Y):
    
     plt.scatter(X,Y,color="red")
     plt.show()
#this function to plot the model data     
def plot_model_data(X,Y,Yt):
     
    plt.scatter(X,Y,color="red") 
    plt.scatter(X,Yt.T)
    plt.show()
# this function is used to split the data     and do cross validation
def split_data(X,Y,degree):
       
      Testing_error =[] #all the testing errors of 10 fold cross validations
      Training_error = [] #all the training errors  of 10 fold cross validations
      X_sets =  np.split(X,10)
      Y_sets = np.split(Y,10)
      
      for i in range(len(X_sets)):
          X_test =np.vstack( X_sets[i])
          Y_test = np.vstack(Y_sets[i])
          if i<len(X_sets)-1: 
             X_train = np.vstack(X_sets[i+1:])      
             Y_train =np.vstack(Y_sets[i+1:])
          elif i==len(X_sets)-1 : 
             X_train = np.vstack(X_sets[:i])
             Y_train = np.vstack(Y_sets[:i])
          while i>0:
              tempX = np.vstack(X_sets[i-1])
              X_train = np.append(tempX,X_train)
              tempY = np.vstack(Y_sets[i-1])
              Y_train = np.append(tempY,Y_train)
              i = i-1
          X_train = np.vstack(X_train)
          Y_train = np.vstack(Y_train)
          Z_train,theta,Z_test = polynomial_withCV(X_train,Y_train,degree,X_test)
          Testing_error.append( mse(Z_test,theta,Y_test))
          Training_error.append(mse(Z_train,theta,Y_train))
      return sum(Testing_error),sum(Training_error)
#this function to predict the Y       
def prediction(theta,Z):

      Yt =np.dot(theta.T,Z.T)
      return Yt
          
       
    
def main():
    
    #the program starts here
    #the line below has all the file name saved
    files  =["svar-set1.dat.txt","svar-set2.dat.txt","svar-set3.dat.txt","svar-set4.dat.txt"]
    file_to_use = input("enter the file to be used:")
    # the line below is used to load the
    #particular file and distributes the data to X and Y
    X,Y = np.loadtxt(files[file_to_use-1],unpack =True,usecols=[0,1])
    degree = input("enter the degree of the polynomial:")
    #reshaping the feature vector and label vector
    X = X.reshape(X.size,1)
    Y = Y.reshape(Y.size,1) 
    #plotting the sample data to visualize the data beforing using it develop  model
    plot_sample_data(X,Y)
    #the below line takes the program to split_data function which 
    #splits the dat and sends data for cross_validation and return the  MSE
    Testing_error, Training_error = split_data(X,Y,degree)
    # the average MSE of 10 fold cross validation is calculated
    Testing_error = Testing_error/10
    Training_error = Training_error/10
    
    print "Testing error for %s using the degree of polynomial of  %d" %(files[file_to_use-1],degree)
    print Testing_error
    print "Training error for %s using the degree of polynomial of  %d" %(files[file_to_use-1],degree)
    print Training_error
    
    # the part below this where the whole data is trained and predicted
    Z= polynomial(X,degree)
    theta = theta_computation(Z,Y)
   
    Yt =  prediction(theta,Z)
    
    plot_model_data(X,Y,Yt)

    
if __name__ == "__main__":
    main()          
    