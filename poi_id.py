#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

### Task 1: Select Features
features_list = ['poi','salary', 'deferral_payments', 
                 'total_payments', 'loan_advances', 
                 'bonus', 'restricted_stock_deferred', 
                 'deferred_income', 'total_stock_value', 
                 'expenses', 
                 'exercised_stock_options', 
                 'long_term_incentive', 
                 'restricted_stock',
                 'to_messages', 
                 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi'
                 ]
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
##Data Exploration
#total number of data points
total_people = len(data_dict.keys())

print 'Number of people:', total_people
print 'Number of features:',(len(data_dict.values()[0]))
##Finding POIs in the Enron Data
poi_count=0
for x, y in data_dict.items():
    if y['poi']==1:
        poi_count+=1
print 'Person of Interest count:', poi_count

#Percent POI
poi_perc = round((1.*poi_count/total_people)*100,2)
print 'POI percentage:',poi_perc,"%"

# # of features used
feat_len = len(features_list)
print 'Number of features used:', feat_len

### Task 2: Remove outliers
##check for outliers
features = ['salary','bonus']
data = featureFormat(data_dict, features)

plt.scatter(data[:,0], data[:,1])
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

##Remove
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)

person = [] 
for p in data_dict: 
     if data_dict[p]['salary'] != "NaN": 
         person.append((p, data_dict[p]['salary'])) 
print "Outliers left:" 
print sorted(person, key = lambda x: x[1], reverse=True)[0:4]

plt.scatter(data[:,0], data[:,1])
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

### Task 3: Create new feature(s)
my_dataset = data_dict
for p in my_dataset:
    from_poi = my_dataset[p]['from_poi_to_this_person']
    total_to_msg = my_dataset[p]['to_messages']
    if from_poi != 'NaN' and total_to_msg !='NaN':
        my_dataset[p]['from_poi_perc'] = (1.*from_poi/total_to_msg)*100
    else:
        my_dataset[p]['from_poi_perc'] = 0
        
for p in my_dataset:
    to_poi = my_dataset[p]['from_this_person_to_poi']
    total_from_msg = my_dataset[p]['from_messages']
    if to_poi != 'NaN' and total_from_msg !='NaN':
        my_dataset[p]['to_poi_perc'] = (1.*to_poi/total_from_msg)*100
    else:
        my_dataset[p]['to_poi_perc'] = 0

features_list = features_list + ['to_poi_perc']+['from_poi_perc']
features_list


### Extract features and labels from dataset for local testing
my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


features_list = ['poi',
 'salary',
 'deferral_payments',
 'total_payments',
 'loan_advances',
 'bonus',
 'restricted_stock_deferred',
 'deferred_income',
 'total_stock_value',
 'expenses',
 'exercised_stock_options',
 'long_term_incentive',
 'restricted_stock',
 'to_messages',
 'from_poi_to_this_person',
 'from_messages',
 'from_this_person_to_poi',
 'to_poi_perc',
 'from_poi_perc']
from sklearn.feature_selection import f_classif, SelectKBest 
k = 7
selector = SelectKBest(f_classif, k) 
selector.fit_transform(features, labels) 
print "Best features:"
scores = zip(features_list[1:],selector.scores_)
scores_sorted = sorted(scores, key = lambda x: x[1], reverse=True  )
#scores_sorted
best_features = scores_sorted[:k]
best_features

##Final set of features
features_list = ['poi','salary', #'deferral_payments', 
                 #'total_payments', 
                 #'loan_advances', 
                 'bonus', #'restricted_stock_deferred', 
                 'deferred_income', 
                 'total_stock_value', 
                 #'expenses', 
                 'exercised_stock_options',
                 'long_term_incentive',
                 #'restricted_stock',
                 #'to_messages', 
                 #'from_poi_to_this_person', 'from_messages',
                 #'from_this_person_to_poi'
                 'to_poi_perc'
                 #'from_poi_perc'
                 ]
                 
    
### Task 4: Classifiers
## use KFold for split and validate algorithm 
from sklearn.cross_validation import KFold 
kf=KFold(len(labels),3) 
for train_indices, test_indices in kf: 
    #make training and testing sets 
    features_train= [features[ii] for ii in train_indices] 
    features_test= [features[ii] for ii in test_indices] 
    labels_train=[labels[ii] for ii in train_indices] 
    labels_test=[labels[ii] for ii in test_indices] 


##Naive Bayes
from time import time
t0 = time() 
 
clf = GaussianNB() 
clf.fit(features_train, labels_train) 
pred = clf.predict(features_test) 
accuracy = accuracy_score(pred,labels_test) 
print 'Naive Bayes accuracy:',round(accuracy,4) 
 
print 'Naive Bayes time:', round((time()-t0), 3), "s" 

##Decision Tree
t0 = time() 

clf = tree.DecisionTreeClassifier() 
clf.fit(features_train,labels_train) 
pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)  
print 'Decision Tree accuracy:', round(accuracy,4)
 
print 'Decision Tree time:', round((time()-t0), 3), "s"

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. 

##Tuning
##Min_Sample_split = 5
t0 = time() 
clf = tree.DecisionTreeClassifier(min_samples_split=5) 
clf = clf.fit(features_train,labels_train) 
pred= clf.predict(features_test) 
accuracy = accuracy_score(labels_test, pred)
print 'Decision Tree Tuning 1:'
print 'Accuracy:', round(accuracy,4)
print 'Decision Tree time:', round((time()-t0), 3), "s"
test_classifier(clf, my_dataset, features_list)


##Min_Sample_split = 10
t0 = time() 
clf = tree.DecisionTreeClassifier(min_samples_split=15) 
clf = clf.fit(features_train,labels_train) 
pred= clf.predict(features_test) 
accuracy = accuracy_score(labels_test, pred)
print '\nDecision Tree Tuning 2:'
print 'Accuracy:', round(accuracy,4)
print 'Decision Tree time:', round((time()-t0), 3), "s"
test_classifier(clf, my_dataset, features_list)

##Min_Sample_split = 15
t0 = time() 
clf = tree.DecisionTreeClassifier(min_samples_split=30) 
clf = clf.fit(features_train,labels_train) 
pred= clf.predict(features_test) 
accuracy = accuracy_score(labels_test, pred)
print '\nDecision Tree Tuning 1:'
print 'Accuracy:', round(accuracy,4)
print 'Decision Tree time:', round((time()-t0), 3), "s"
test_classifier(clf, my_dataset, features_list)


##Final Algorithm
t0 = time()
clf = GaussianNB() 
clf.fit(features_train, labels_train) 
pred = clf.predict(features_test) 
accuracy = accuracy_score(pred,labels_test)

print '\nFinal Algorithm NB:'
print 'Naive Bayes accuracy:',accuracy

print "time:", round((time() - t0),3), "s."


##Precision
print 'Precision : ', precision_score(labels_test,pred)

##Recall
print 'Recall:', recall_score(labels_test, pred)

### Task 6: Dump your classifier, dataset, and features_list.
test_classifier(clf, my_dataset, features_list)
dump_classifier_and_data(clf, my_dataset, features