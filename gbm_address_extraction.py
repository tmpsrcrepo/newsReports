# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 13:45:30 2015

@author: Xiaoqian
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import hmm
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
import re,string
from bs4 import BeautifulSoup
from urllib import urlopen
import nltk
from nltk.tokenize import RegexpTokenizer
import multiprocessing
from functools import partial
import sys
import time
from sklearn.cross_validation import train_test_split

tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
#==============================================================================
# data collected from Yelp data (restaurants, business)
#==============================================================================
def extract_Info(index,loc):
    #location: tuple(city,state)
    location = loc.split(',')
    link = 'http://www.yelp.com/search?find_loc='+location[0]+'%2C+'+\
    location[1]+'&start='+str(index)
    html = urlopen(link).read()
    soup = BeautifulSoup(html)
    addressList = []
    tokenList = []
    if html and soup:
        addressList = [str(addr) for addr in soup.findAll('address') if addr]
        #sub:
        addressList = (re.sub('</*[a-z]*/*>',' ',address) for address in addressList)
#        for link in soup.findAll('a',attrs={'class':'page-option available-number'}):
#            print link['href']
        raw_txt = nltk.clean_html(html)
        tokenList = tokenizer.tokenize(raw_txt)
    time.sleep(2)
    print loc
    return {'address':''.join(addressList),'raw_txt':raw_txt}
        
def Yelp_scraper():
    locationList = ['Charlottesville,VA','New York,NY','San Francisco,CA',\
    'Los Angeles,CA']
    result = (extract_Info(i,t) for t in locationList for i in xrange(0,210,10))
    
    df = pd.DataFrame(list(result))
    df.to_csv('_yelp_data_set.csv',sep=',')
    return df
    
def Yelp_scraper_MP(location,indices):
    p = multiprocessing.Pool()
    results = p.imap(partial(extract_Info,loc=location), indices)
    num_tasks = len(indices)

    while (True):
        completed = results._index
        print "\r--- Completed {:,} out of {:,}".format(completed, num_tasks)
        sys.stdout.flush()
        time.sleep(3)
        if (completed == num_tasks): break
    p.close()
    p.join()
    df = pd.DataFrame(list(results))
    print 'all done'

    a = re.sub(',','_',location)
    df.to_csv(a+'_yelp_'+str(num_tasks)+'_addresses.csv',',')
    return df

#locationList = ['Charlottesville,VA','New York,NY','San Francisco,CA','Los Angeles,CA']
#Yelp_scraper()

s_time = time.time()
df = pd.DataFrame.from_csv('_yelp_data_set.csv')
row1 =  df.loc[0]

def cleanAddress(addr):
    addr1 = addr.split(',')[0]
    addr1 = addr1.split()[:-1]
    if len(addr1)>0 and (addr1[-1] == 'Los' or addr1[-1] =='San'):
        addr1 = addr1[:-1]
    return addr1

#addr1 = [cleanAddress(a) for a in row1['address'].split('\n')]
#addr1 = [a for a in addr1 if len(a)>0]

#tokens= tokenizer.tokenize(row1['raw_txt'])
#print addr1
#print tokens
#print len(addr1),len(tokens)

#print string.punctuation


def featureRegex():
    #all capitalized
    allLow = re.compile('[a-z]\w+')
    allDigit = re.compile('[0-9]\w+')
    DigitLetter = re.compile('[0-9]\w+[A-Z]*[a-z]*')
    email = re.compile('@')
    url1 = re.compile('http')
    url2 = re.compile('www')
    phone1 = re.compile('\([0-9]{3}\)')
    phone2 = re.compile('\-[0-9]{4}')
    return allLow,allDigit,DigitLetter,email,url1,url2,phone1,phone2

allLow,allDigit,DigitLetter,email,url1,url2,phone1,phone2 = featureRegex()
punctuations = string.punctuation

dict_Unigrams = {}
dict_Bigrams = {}
dict_Bigrams1 = {}
Count_Bigrams = {}


class Unigram(object):
    def __init__(self,x):
        self.val = x
        self.label = -2


class Bigram(object):
    def __init__(self,x):
        self.val = x
        self.features = []
        self.label = -1


def featureExtract(token):
    features = [int(token.isupper()),int(token[0].isupper()),int(token.islower()),\
    int(punctuations in token),int(token.isdigit()),len(DigitLetter.findall(token)),\
    len(email.findall(token))]
    
    return features


labels = defaultdict()

def addressBigrams(addr):
    addr_label = {}
    
    for ad in addr:
        len_ = len(ad)-1
        for i,a in enumerate(ad):
            if i ==0:
                start = ' '.join(ad[:2])
                addr_label[start]=0
            elif a == ad[-1]:
                end = ad[-2]+' '+a
                addr_label[end] = 2
            else:
                mid = ' '.join(ad[i:i+2])
                addr_label[mid] = 1
    return addr_label


def bigramCreation(row):
    addr = [cleanAddress(a) for a in row['address'].split('\n')]
    addr = [a for a in addr if len(a)>0]
    #addr_ = [a_ for a in addr for a_ in a]
    
    addr_label = addressBigrams(addr)
    
    #head = map(lambda x:x[0],addr)
    #mid = map(lambda x:x[1:-1],addr)
    #end = map(lambda x:x[-1],addr)
    
    tokens= tokenizer.tokenize(row['raw_txt'])
    # add first letter to the token list
    tokens = ['~']+tokens
    featureList = []
    
    #Labeling + feature extraction
    for i,token in enumerate(tokens[:-2]):
        bigram = tokens[i:i+3]
        if i == 0:
            lookup = ' '.join(tokens[:2])
        else:
            lookup = ' '.join(bigram[:-1])
        
        bigram_ = ' '.join(bigram)
        if lookup in addr_label:
            labels[bigram_] = addr_label[lookup]
        else:
            labels[bigram_] = -1
        
        #if bigram_ not in dict_Bigrams:
        dict_Bigrams[bigram_]=np.array(featureExtract(bigram[0])+featureExtract(bigram[1])+featureExtract(bigram[2]))
        l1 = np.array(featureExtract(bigram[0]))
        l2 = np.array(featureExtract(bigram[1]))
        l3 = np.array(featureExtract(bigram[2]))
        result= np.add(l1,l2,l3)

        if bigram_ not in dict_Bigrams1:
            dict_Bigrams1[bigram_] = result
        else:
            tmp = dict_Bigrams1[bigram_]
            dict_Bigrams1[bigram_] = np.add(result,tmp)
        #else:


msk = np.random.rand(len(df)) < 0.75
train_df = df[msk]
test_df = df[~msk]



def createDicts(df):
    dict_Bigrams1 = {}
    df.apply(bigramCreation,axis=1)
    X = np.array(dict_Bigrams1.values())
    Y = np.array(labels.values())
    return dict_Bigrams1,X,Y

#dict_Bigrams1,trainX,trainY = createDicts(train_df)
#print trainX.shape,trainY.shape
#dict_Bigrams2,testX,testY = createDicts(test_df)


dict_Bigrams1 = {}
train_df.apply(bigramCreation,axis=1)
trainX = np.array(dict_Bigrams1.values())
trainY = np.array(labels.values())
trainKeys = dict_Bigrams1.keys()


dict_Bigrams1 = {}
labels = {}
test_df.apply(bigramCreation,axis=1)
testX = np.array(dict_Bigrams1.values())
testY = np.array(labels.values())
testKeys = dict_Bigrams1.keys()

#print testX.shape,testY.shape

def HMM_Model(X):
    model = hmm.GaussianHMM(X.shape[1], "full")
    model.fit([X])
    Z = model.predict(X)
    return Z

    #calculate RMSE, precision, recall

def GBM_Model(trainX,trainY,testX,testY):
    est = GradientBoostingClassifier(n_estimators=1000, max_depth=5)
    est.fit(trainX,trainY)
    pred = est.predict(testX)
    return pred

pred =GBM_Model(trainX,trainY,testX,testY)



def concatenateStr(pred,keys_):
    results = []
    tmp = ''
    start = -1
    mid = -1
    
    for i,p in enumerate(pred):
        if p == 0:
            start = i
            tmp= keys_[i].split()[0]+' '+keys_[i].split()[1]
        if p == 1 and start > 0:
            mid = i
            tmp+=' '+keys_[i].split()[1]
        if p == 2 and start >= 0 and mid > 0:
            results.append(tmp+' '+keys_[i].split()[0]+' '+keys_[i].split()[1])
            tmp = ''
            start,mid = -1,-1
    return results



print '------------------Predicted Addresses------------------'


print pred.shape,testY.shape

pred_results= concatenateStr(pred,testKeys)
for p in pred_results:
    print p


print
print '------------------Addresses in the test set------------------'
print
test_result = concatenateStr(testY,testKeys)
for test in test_result:
    print test


print time.time() - s_time




#==============================================================================
# Using Hidden Markov Model to extract addresses from news articles (web)
# (can try CRF as well)
# reference from http://pe.usps.gov/cpim/ftp/pubs/Pub28/pub28.pdf
# supervised learning could also work (need to be trained on labeled data)
#==============================================================================






#==============================================================================
# Standardize 
#==============================================================================
