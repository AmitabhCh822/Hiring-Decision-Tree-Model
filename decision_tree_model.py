import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from io import StringIO
from IPython.display import Image
import pydotplus
from sklearn.model_selection import train_test_split
from sklearn import tree
from collections import Counter

def entropy(prob: list):
    sum_of_prob = np.sum(prob)
    
    if (sum_of_prob>1.00 or sum_of_prob<1.00):
        print("The probability list is inaccurate.")
    
    e = 0
    for i in prob:
        e = e - (i* math.log(i,2))  #log function code retrieved from Stack Overflow
    return e
        

assert(entropy([0.5,0.5])==1)
assert(entropy([1.0])==0)
assert(entropy([0.75,0.25])<= .83 and entropy([0.75,0.25])>= .81)


def data_entropy(data: list):
    d = Counter(data)
    prob = []
    for i in d:
        p = d[i]/len(data)
        prob.append(p)
    return (entropy(prob))
    
assert(data_entropy(['a','a']) == 0)
assert(data_entropy(['a']) == 0)
assert(data_entropy([True, False]) == 1)
assert(data_entropy([3, 4, 4, 4]) <= 0.82 and data_entropy([3, 4, 4, 4]) >= 0.81) 


"""Returns the entropy from a partition of data into subsets"""
def split_entropy(subsets: list):
    total_count = sum(len(subset) for subset in subsets)
    return (sum(data_entropy(subset) * len(subset) / total_count for subset in subsets))

            
assert(split_entropy([['T','T','F','T','T','F','T','T','F','T','T','F'],
                     ['F','F','F','F','F','F','F','F']]) >= 0.550 and 
       split_entropy([['T','T','F','T','T','F','T','T','F','T','T','F'],
                     ['F','F','F','F','F','F','F','F']]) <= 0.551)


#Reading the data
df = pd.read_excel('candidate interviews.xlsx')


#Using the data_entropy function and split_entropy function to trace the decision tree algorithm by hand

print(data_entropy(df["did_well"]))
#The entropy of the class attribute "did_well" is 0.94

print(split_entropy([['F','F','F','T','T'],['T','T','T','T',],['T','T','T','F','F']]))
#The entropy of did_well when split on level is 0.694. The information gain is (0.94-0.694)=0.246 when split on level.
print(split_entropy([['F','F','T'],['T','T','T','T','T', 'F', 'F'],['T','T','T','F']]))
#The entropy of did_well when split on lang is 0.86. The information gain is (0.94-0.86)=0.08 when split on lang.
print(split_entropy([['F','F','T','T','F','T','F'],['T','F','T','T','T','T','T']]))
#The entropy of did_well when split on tweets is 0.7885. The information gain is (0.94-0.7885)=0.1515 when split on tweet.
print(split_entropy([['F','T','T','T','F','T','T','T'],['T','T','T','F','F','F']]))
#The entropy of did_well when split on grad_degree is 0.89. The information gain is (0.94-0.89)=0.05 when split on grad_degree.
      
assert(data_entropy(df["did_well"])<=.941 and
      data_entropy(df["did_well"])>=.94)
assert(split_entropy([['F','F','F','T','T'],['T','T','T','T',],['T','T','T','F','F']])<=0.695 and
      split_entropy([['F','F','F','T','T'],['T','T','T','T',],['T','T','T','F','F']])>= 0.693)
assert(split_entropy([['F','F','T'],['T','T','T','T','T', 'F', 'F'],['T','T','T','F']])<=0.861 and
      split_entropy([['F','F','T'],['T','T','T','T','T', 'F', 'F'],['T','T','T','F']])>=0.86)
assert(split_entropy([['F','F','T','T','F','T','F'],['T','F','T','T','T','T','T']])<=0.7886 and
      split_entropy([['F','F','T','T','F','T','F'],['T','F','T','T','T','T','T']])>=0.7884)
assert(split_entropy([['F','T','T','T','F','T','T','T'],['T','T','T','F','F','F']])<=0.893 and
      split_entropy([['F','T','T','T','F','T','T','T'],['T','T','T','F','F','F']])>=0.892)

#Since the highest information gain is when split on level, we are picking level as the first decision in the tree.
#When level=Mid, all the class attribute values are T. So, level=Mid is a leaf node of the tree.

print(data_entropy(['T','T','F','F','F']))
#The entropy in case of "Senior" is 0.971

print(split_entropy([['F','F'],['T','F'],['T']]))
#The entropy of Senior when split on lang is 0.4. The information gain is (0.971-0.4)=0.371 when split on lang.
print(split_entropy([['F','F','F'],['T','T']]))
#The entropy of Senior when split on tweets is 0. The information gain is (0.971-0.0)=0.971 when split on tweet.
print(split_entropy([['F','F','T'],['F','T']]))
#The entropy of Senior when split on grad_degree is 0.951. The information gain is (0.971-0.951)=0.02 when split on grad_degree.

assert(data_entropy(['T','T','F','F','F'])<=.972 and 
       data_entropy(['T','T','F','F','F'])>=.97)
assert(split_entropy([['F','F'],['T','F'],['T']])==0.4)
assert(split_entropy([['F','F','F'],['T','T']])==0)
assert(split_entropy([['F','F','T'],['F','T']])<=0.951 and 
       split_entropy([['F','F','T'],['F','T']])>=0.950)

#Since the highest information gain is when split on tweets, we should peek tweets as the second decision from level=Senior in the tree.
#When tweets=F, all the class attribute values are F and when tweets=T, all the class attribute values are T. So, tweets=F and tweets=T are leaf nodes of the tree.

print(data_entropy(['T','T','T','F','F']))
#The entropy in case of "Junior" is 0.971
print(split_entropy([['T','F'],['T','T','F']]))
#The entropy of Junior when split on lang is 0.951. The information gain is (0.971-0.951)=0.02 when split on lang.
print(split_entropy([['T','T','F'],['T','F']]))
#The entropy of Junior when split on tweets is 951. The information gain is (0.971-0.951)=0.02 when split on tweet.
print(split_entropy([['T','T','T'],['F','F']]))
#The entropy of Junior when split on grad_degree is 0. The information gain is (0.971-0)=0.971 when split on grad_degree.

assert(data_entropy(['T','T','T','F','F'])<=.971 and 
       data_entropy(['T','T','T','F','F'])>=.97)
assert(split_entropy([['T','F'],['T','T','F']])<=.952 and 
       split_entropy([['T','F'],['T','T','F']])>=.95)
assert(split_entropy([['T','T','F'],['T','F']])<=.952 and
      split_entropy([['T','T','F'],['T','F']])>=.95)
assert(split_entropy([['T','T','T'],['F','F']])==0)

#Since the highest information gain is when split on grad_degree, we should peek grad_degree as the second decision from level=Junior in the tree.
#When grad_degree=F, all the class attribute values are T and when grad_degree=T, all the class attribute values are F. So, grad_degree=F and grad_degree=T are leaf nodes of the tree.

#So, in the decision tree, the level attribute is the root node
#tweets and grad_degree are the second decision splitting level=Senior and level=Junior respectively
#level=Mid, tweets=F, tweets=T, grad_degree=T and grad_degree=F are the leaf nodes of the tree


#Building the decsion tree using Python libraries
#One hot encoding on nominal values
one_hot_data = pd.get_dummies(df.drop('did_well', axis=1))
print(one_hot_data.head())

#Creating X and Y
X = one_hot_data
y = df[["did_well"]]

"""
X_train = X
X_test = X
y_train = y
y_test = y
"""
#Making the training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

#Making an object of the Classifier
clf = DecisionTreeClassifier(criterion='entropy',random_state=42).fit(X_train, y_train)
#Predicting on the test set
predictions = clf.predict(X_test)

fn= one_hot_data.columns
cn= ['did_well=F','did_well=T']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=1000)
tree.plot_tree(clf,
               feature_names = fn, 
               class_names=cn,
               filled = True);

#Printing the accuracy
print("Accuracy:", accuracy_score(y_test, predictions))
#Printing the confusion matrix
print(confusion_matrix(y_test,predictions))
#Printing the classification report
print(classification_report(y_test,predictions))