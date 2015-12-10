# -*- coding: utf-8 -*-
"""
    @author: Xiaoqian
    """

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
import re,string
from bs4 import BeautifulSoup
from urllib import urlopen
import nltk
from nltk.tokenize import RegexpTokenizer
import multiprocessing
from functools import partial
import sys
import time
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import glob
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
        raw_txt = nltk.clean_html(html)
        tokenList = tokenizer.tokenize(raw_txt)
    time.sleep(1)
    return {'address':''.join(addressList),'raw_txt':raw_txt}

def Yelp_scraper(filename,locationList,limit):
                    result = (extract_Info(i,t) for t in locationList for i in xrange(0,limit,10))
                    df = pd.DataFrame(list(result))
                    df.to_csv(filename,sep=',')
                    return df

locationList = ['Charlottesville,VA']
#Yelp_scraper('_yelp_data_set_Charlottesville.csv',locationList,810)

#done with scraping

df = pd.DataFrame.from_csv('_yelp_data_set_Charlottesville.csv')
#df = pd.DataFrame.from_csv('_yelp_data_set.csv')


#street abbreviations
street_abr = pd.read_csv('street_abbr.csv')
street_abr.fillna(method = 'pad',inplace=True)
st_abbr_val= {v:1 for v in street_abr['VALUES'].values}

#Directions
directions = ['N','E','S','W','NE','SE','NW','SW']

def cleanAddress(addr):
    addr1 = addr.split(',')[0]
    addr1 = addr1.split()[:-1]
    if len(addr1)>0 and (addr1[-1] == 'Los' or addr1[-1] =='San' or addr1[-1] == 'New'):
        addr1 = addr1[:-1]
    return addr1




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
    num_abbr = re.compile('th|rd|st|nd',re.IGNORECASE)
    ampm = re.compile('am|pm',re.IGNORECASE)
    times_ = re.compile('[0-9]*:[0-9]*')
    timezone = re.compile('PT|CST|PST|EST|ET')
    
    return allLow,allDigit,DigitLetter,email,url1,url2,phone1,phone2,num_abbr,ampm,times_,timezone

allLow,allDigit,DigitLetter,email,url1,url2,phone1,phone2,num_abbr,ampm,times_,timezone = featureRegex()
punctuations = string.punctuation

def featureExtract(token):
    features = [int(token.isupper()),int(token.upper() in st_abbr_val),int(token[0].isupper()),int(token.islower()),int(num_abbr.findall(token)!=None),len(ampm.findall(token)),len(times_.findall(token)),int(timezone.findall(token)!=None), int(token.upper() in directions),\
                int(punctuations in token),int(token.isdigit()),len(DigitLetter.findall(token)),int(email.findall(token)!=None),len(token),token[0].isalpha()]
    return features


labels = defaultdict()


def addressBigrams(addr):
    addr_label = {}
    
    for ad in addr:
        
        for i,a in enumerate(ad[:-1]):
            bigram = ' '.join(ad[i:i+2])
            addr_label[bigram] = 1

        addr_label[' '.join(ad[:2])] = 0
        addr_label[' '.join(ad[-2:])] = 2

    return addr_label



X = []
Y = []
dict_Bigrams1 = {}

def bigramCreation(row):
    
    addr = [cleanAddress(a) for a in row['address'].split('\n')]
    addr = [a for a in addr if len(a)>0]
    
    addr_label = addressBigrams(addr)
    
    
    tokens= tokenizer.tokenize(row['raw_txt'])
    
    # add first letter to the token list
    featureList = []
    
    #Labeling + feature extraction
    for i,token in enumerate(tokens[:-1]):
        bigram = tokens[i:i+2]
        bigram_ = ' '.join(bigram)
        
        if bigram_ in addr_label:
            #labels[bigram_] = addr_label[bigram_]
            Y.append(addr_label[bigram_])
            #trainX = np.append(trainX,dict_Bigrams1[bigram_],axis=0)
            #X.append(dict_Bigrams1[bigram_],axis=1)
        else:
            #labels[bigram_] = 3
            Y.append(3)
        
        if bigram_ not in dict_Bigrams1:
            dict_Bigrams1[bigram_]=np.array(featureExtract(bigram[0])+featureExtract(bigram[1]))
                                            #+featureExtract(bigram[2]))
            #trainX = np.append(trainX,dict_Bigrams1[bigram_],axis=0)
        X.append(dict_Bigrams1[bigram_])





def RF_model(trainX,trainY):
    est = RandomForestClassifier(n_estimators=1000, max_depth=10,n_jobs=2)
    #75% train set and 25% trainset
    x_train, x_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.25, random_state=20)
    print trainX.shape,trainY.shape
    est.fit(x_train,y_train)
    print est.feature_importances_
    target_names = ['Start','Mid','End','Others']
    #target_names = ['start','mid','others']
    y_pred = est.predict(x_test)
    print classification_report(y_test, y_pred, target_names=target_names)
    #fit the full model
    est.fit(trainX,trainY)
    #scores = cross_validation.cross_val_score(est, trainX, trainY, cv=10)
    #print scores.mean(),scores.std()*2
    return est
#print np.array(X).shape

trainX = np.empty((0,14*2),int)
df.apply(bigramCreation,axis=1)
est = RF_model(np.array(X),np.array(Y))


######Test on the test set######
soup = BeautifulSoup(urlopen('/Users/Xiaoqian/documents/Research 2015-2016/News report/factiva/Factiva.html').read())

lis = [re.sub('\.',' ',a.text) for a in soup.findAll('div',attrs={'class':'article'})]

def readF(path):
    files = glob.glob(path)
    out = []
    for f in files:
        out.append(' '.join([re.sub('\.',' ',l) for l in open(f)]))
    return out

lis = (readF('WP/*.txt'))
addr_path = 'WP.csv'
#addr_path = 'Factiva.csv'

#lis = (readF('University Wire_keywords/*.txt'))
#####Preprocessing data#######
#remove words in every single

def importLabels(path):
    values = pd.read_csv(path,names=['Addr'])
    values = [v.split() for v in values.Addr]
    return addressBigrams(values)


def FeatureToken_Label(tokens,x,Y,addr_label):
    bigrams = []
    dict_Bigrams1 = {}
    for i,token in enumerate(tokens[:-1]):
        bigram = tokens[i:i+2]
        
        
        bigram_ = ' '.join(bigram)
        if bigram_ in addr_label:
            Y.append(addr_label[bigram_])
        else:
            Y.append(3)
                     
        
        if bigram_ not in dict_Bigrams1:
            dict_Bigrams1[bigram_]=np.array(featureExtract(bigram[0])+featureExtract(bigram[1]))
        #+featureExtract(bigram[2]))
        x.append(dict_Bigrams1[bigram_])
        bigrams.append(bigram_)
    return bigrams


def createBigramList(lis):
    bigrams = []
    tmpset = set()
    outX = []
    #labels
    addr_label = importLabels(addr_path)
    Y = []
    for l in lis:
        tokens = tokenizer.tokenize(l)
        #bigrams.append(FeatureToken_Label(tokens,x)
        bigrams+=(FeatureToken_Label(tokens,outX,Y,addr_label))

    return bigrams,outX,Y


bigramList,outX,Y = createBigramList(lis)
outX = np.array(outX)
out = est.predict(outX)
target_names = ['Start','Mid','End','Others']
print classification_report(out, Y, target_names=target_names)



def TokenPredsequence(pred,bigramList):
    last = ''
    tmp = ''
    count = 0
    indices = xrange(0,len(bigramList))
    result = []
    #a = [(bigram,label) for bigram,label in zip(bigramList,pred)]
    
    for i,bigram,label in zip(indices,bigramList,pred):
        if label == 3 and pred[i-1] == 3:
            if tmp!='' and count>1:
                st_flag = 0
                t = tmp.split()[-1]
                if t.upper() in st_abbr_val or t.upper() in directions:
                    result.append(tmp.strip())
                else:
                    diff = len(t)
                    t_ = tmp.split()
                    if t_[-3].upper() in st_abbr_val or t_[-2].upper() in directions:
                        result.append(tmp[:-diff].strip())
            
            tmp = ''
        else:
            bigrams = bigram.split()
            if bigrams[0] == last and (bigrams[1][0].isalpha() or bigrams[1][0].isdigit()):
                tmp += ' '+bigrams[1]
                count+=1
            else:
                tmp = bigram
                count = 1
            last = bigrams[1]

    #0,1,2 or 0,1 or 0,2 or 0,3 or 1


    return result

#then check if street abbr is contained in the retrieved addresses



print '------------------New Addresses------------------'
out_result = set(TokenPredsequence(out,bigramList))
print (out_result)
vals = list(pd.read_csv('WP.csv',names=['Addr']).Addr)
print vals
vals = set([' '.join(val.split()[:-1]) for val in vals])
totalnum = len(out_result)
print float(len([1 for o in out_result if o in vals]))/totalnum
print float(len([1 for v in vals if v not in out_result]))/totalnum
print [o for o in out_result if o.strip() not in vals]




