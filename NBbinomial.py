# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 15:18:44 2016

@author: raviteja
"""
from collections import defaultdict
from math import log
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_curve, auc,f1_score

def main():
      
      #readin the file
      filename = "SMSSpamCollection.txt"
      
      file_data = open(filename)
      lines = file_data.readlines()
      doc =[]
      classes =[]
      #splitting the data into document and classes
      for line in lines:
           if line.split()[0]=='spam':
               classes.append('spam')
               doc += [" ".join([word.lower() for word in line.split()[1:]])]
               
           else:
               classes.append('ham')
               doc += [" ".join([word.lower() for word in line.split()[1:]])]
     
      alpha,prior_spam,prior_ham = naiveBayes(doc[0:400],classes[0:400])
      pred_class = predict(alpha,doc[400:],prior_spam,prior_ham)
      cfm= confusion_matrix(classes[400:],pred_class, labels=["ham", "spam"])
      print cfm
      test = classes[400:]
      y = []
      yhat =[]
      for i in range(0,len(test)):
        y+=[1 if test[i] == 'spam' else 0]
        yhat+=[1 if pred_class[i] == 'spam' else 0]
    
      #calculating the metrics like acccuracy precision .....
      fpr, tpr, thresholds = roc_curve(y,yhat)
      roc_auc = auc(fpr, tpr)
      accuracy = accuracy_score(y,yhat)
      precision = precision_score(y,yhat)
      recall = recall_score(y,yhat)
      f1= f1_score(y,yhat)
      print "accuracy : %s" %accuracy
      print "precision : %s" %precision
      print "recall : %s" %recall
      print "f1score : %s" %f1
      print "Area under the ROC curve: %s" %roc_auc 

#naive bayes algorithm     
def naiveBayes(doc,classes):

      spam_words =0 
      not_spam_words =0
      word_dict = defaultdict(lambda: defaultdict(lambda: 0))
      for i in range(0,len(doc)):
              for word in doc[i].split():
                  word_dict[word][i] =word_dict[word][i]+ 1
                  if classes[i] == 'spam':
                       spam_words =spam_words+ 1
                 
                  else : 
                        not_spam_words = not_spam_words+ 1
      #priors of both classes   
      prior_spam = len([val for val in classes if val=='spam'])/float(len(classes))
      prior_ham = len([val for val in classes if val=='ham'])/float(len(classes))
      alpha ={}           
      for word in word_dict:
          alpha[word] = calculateAlpha(word_dict[word],classes,spam_words,not_spam_words,smoothing=1.)  
      return alpha,prior_spam,prior_ham  
      
#calculating the parameters to predict      
def calculateAlpha(docDict,y,total_spam_words,total_ham_words,smoothing=1.):
   
    spam_sum = sum([docDict[doc_id] for doc_id in docDict if y[doc_id] == 'spam'])
    ham_sum = sum([docDict[doc_id] for doc_id in docDict if y[doc_id] == 'ham'])
    return ((spam_sum+smoothing)/(total_spam_words+2.*smoothing),(ham_sum+smoothing)/(total_ham_words+2.*smoothing))

#predicting the test data to classify        
def predict(alpha,docs,prior_spam,prior_ham):
  pred_class = []  
  for document in docs:
     ham =0.
     spam =0.
     for word in list(set(document.split())):
            if word in alpha:
               spam = spam+log(alpha[word][0])
               ham = ham+log(alpha[word][1])
         
     pred_class.append('ham' if spam <= ham else 'spam')
  return pred_class  
if __name__ == "__main__":
    
       main() 