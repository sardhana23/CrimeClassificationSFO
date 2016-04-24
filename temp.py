# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import cross_validation
from sklearn import tree
from sklearn import svm
from sklearn import linear_model
import csv
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
import gzip
import zipfile
from sklearn.ensemble import GradientBoostingClassifier

def loadData(df,scaler=None):
    data = pd.DataFrame(index=range(len(df)))

    data = df.get(['X','Y']) #2

    #data = data.join(pd.get_dummies(df['DayOfWeek'])) #7
    DayOfWeeks = df.DayOfWeek.unique()
    DayOfWeekMap = {}
    i = 0
    for s in DayOfWeeks:
        DayOfWeekMap[s] = i
        i += 1
    data = data.join(df['DayOfWeek'].map(DayOfWeekMap))

    #data = data.join(pd.get_dummies(df['PdDistrict'])) #10
    PdDistricts = df.PdDistrict.unique()
    PdDistrictMap = {}
    i = 0
    for s in PdDistricts:
        PdDistrictMap[s] = i
        i += 1
    data = data.join(df['PdDistrict'].map(PdDistrictMap))

    date_time = pd.to_datetime(df.Dates)
    year = date_time.dt.year
    data['Year'] = year
    month = date_time.dt.month
    data['Month'] = month
    day = date_time.dt.day
    data['Day'] = day
    hour = date_time.dt.hour
    data['hour'] = hour
    minute = date_time.dt.minute
    time = hour*60+minute
    data['Time'] = time
    #data = data.join(pd.get_dummies(year)) #13
    #data = data.join(pd.get_dummies(month)) #12

    data['StreetCorner'] = df['Address'].str.contains('/').map(int)
    data['Block'] = df['Address'].str.contains('Block').map(int)

    X = data.values
    Y = None
    if 'Category' in df.columns:
        Y = df.Category.values

    #scaler = preprocessing.StandardScaler().fit(data.values)
    #X = scaler.transform(data.values)
    return X,Y,scaler

def RF(X,Y):
    print("RF")
    clf = RandomForestClassifier()
    #cross validation
    #n_estimators = [20]
    #clf.set_params(n_estimators=500) #10
    #max_features = ['auto']
    clf.set_params(max_depth=20) #['auto','sqrt',None,'log2']
    #bootstrap = [True,False]
    #clf.set_params(bootstrap=False) #Ture, False
    #criterion = ['gini','entropy']
    #clf.set_params(criterion='entropy') #gini, entropy
    #min_samples_split = [9,11,12,13,14,15]
    #clf.set_params(min_samples_split=12) #2
    #min_samples_leaf = [1,2,3,4,5] #1
    scoreSum = 0
    kfold = cross_validation.KFold(len(X),n_folds=9)
    for train, test in kfold:
        #clf.set_params(max_features='log2')
        #clf.set_params(bootstrap=False)
        #clf.set_params(criterion='entropy')
        clf.set_params(min_samples_split=1000)
        clf.fit(X,Y)
        Yhat = clf.predict(X[test])
        
        score = clf.score(X[test], Y[test])
        print(score)
        scoreSum += score
    #print(scoreSum/4.0)
    '''for i in range(len(max_features)):#
        clf.set_params(max_features=max_features[i])##
        score = cross_validation.cross_val_score(clf,X,Y,cv=4,scoring='log_loss')
        print(score)
        avgScore = np.mean(score)
        #print(max_features[i],avgScore)#
        if minScore>avgScore:
            minScore = avgScore
            best_param=max_features[i]#
    print(best_param,minScore)'''
    return clf

def AdaBoost(X,Y):
    print("Ada")
    clf = AdaBoostClassifier(n_estimators=50)
    
    
    #cross validation
    #n_estimators = [20]
    #clf.set_params(n_estimators=500) #10
    #max_features = ['auto']
    #clf.set_params(max_features=None) #['auto','sqrt',None,'log2']
    #bootstrap = [True,False]
    #clf.set_params(bootstrap=False) #Ture, False
    #criterion = ['gini','entropy']
    #clf.set_params(criterion='entropy') #gini, entropy
    #min_samples_split = [9,11,12,13,14,15]
    #clf.set_params(min_samples_split=12) #2
    #min_samples_leaf = [1,2,3,4,5] #1
    scoreSum = 0
    kfold = cross_validation.KFold(len(X),n_folds=9)
    for train, test in kfold:
        #clf.set_params(max_features='log2')
        #clf.set_params(bootstrap=False)
        #clf.set_params(criterion='entropy')
        #clf.set_params(min_samples_split=1000)
        clf.fit(X[train],Y[train])
        Yhat = clf.predict(X[test])
        
        score = clf.score(X[test], Y[test])
        print(score)
        scoreSum += score
    print(scoreSum/4.0)
    '''for i in range(len(max_features)):#
        clf.set_params(max_features=max_features[i])##
        score = cross_validation.cross_val_score(clf,X,Y,cv=4,scoring='log_loss')
        print(score)
        avgScore = np.mean(score)
        #print(max_features[i],avgScore)#
        if minScore>avgScore:
            minScore = avgScore
            best_param=max_features[i]#
    print(best_param,minScore)'''
    return clf


def RFpredict(X,Y,Xhat):
    clf = RandomForestClassifier()
    #clf.set_params(n_estimators=20)
    #clf.set_params(max_features=None)
    #clf.set_params(criterion='entropy')
    clf.set_params(min_samples_split=1000)
    clf.fit(X,Y)
    Yhat = clf.predict_proba(Xhat)
    return Yhat,clf
    
def NB(X,Y):
    print("NB")
    clf = GaussianNB()
    #clf = MultinomialNB()
    #clf = BernoulliNB()
    scoreSum = 0
    kfold = cross_validation.KFold(len(X),shuffle=True,n_folds=10)
    for train, test in kfold:
        train_arr.append(train)
        test_arr.append(test)
        #clf.set_params(max_features=max_features[0])
        clf.fit(X,Y)
        Yhat = clf.predict(X[test])
        print(clf.score(X[test],Y[test]))
        #yhat_arr.append(Yhat)
        #print Yhat
        #print len(Yhat[0])
        #score = metrics.log_loss(Y[test],Yhat)  
        #print(score)
        #scoreSum += score
    #print (scoreSum/4.0)
    return clf

def NBpredict(X,Y,Xhat):
    clf = GaussianNB()
    clf.fit(X,Y)
    Yhat = clf.predict_proba(Xhat)
    return Yhat,clf

train_arr = []
test_arr = []
yhat_arr = []

train = pd.read_csv("Dataset/train.csv")
df = pd.DataFrame(train)
df['category_count']  = df.groupby('Category')['Category'].transform('count')
df2 = df[df['category_count'].between(4280,174901)]
data_filtered = df2.drop('category_count',1)

test = pd.read_csv("Dataset/test.csv")

data1 = pd.DataFrame(data_filtered.get(['X','Y']))
data1len = len(data1)
data2 = pd.DataFrame(test.get(['X','Y']))
data2len = len(data2)

data = data1.append(data2, ignore_index=True)


est = KMeans(n_clusters=1000, max_iter=100)
est.fit(data)



data_filtered=data_filtered.drop('X',1)
data_filtered=data_filtered.drop('Y',1)
test=test.drop('X',1)
test=test.drop('Y',1)

data_filtered['cluster_ids']=est.labels_[0:data1len]
test['cluster_ids']=est.labels_[data1len:data1len+data2len]

X,Y,scaler = loadData(data_filtered)
AdaBoost(X,Y)

#clf = RF(X,Y)
#clf = AdaBoost(X,Y)
#clf = NB(X,Y)


#Xhat,_,__ = loadData(test,scaler)
#Yhat,clf = RFpredict(X,Y,Xhat)
#Yhat,clf = NBpredict(X,Y,Xhat)

#submission = pd.DataFrame(Yhat,columns=clf.classes_)
#submission['Id'] = test.Id.tolist()