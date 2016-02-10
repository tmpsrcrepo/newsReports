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
from statsmodels.tools import categorical
import collections

tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')

#street abbreviations
street_abr = pd.read_csv('street_abbr.csv')
street_abr.fillna(method = 'pad',inplace=True)
st_abbr_val= {v:1 for v in street_abr['VALUES'].values}

#Directions
directions = ['N','E','S','W','NE','SE','NW','SW']

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def readF(path):
    files = sorted(glob.glob(path), key=numericalSort)
    for f in files:
        #yield (' '.join([re.sub('\.',' ',l) for l in open(f)]))
        yield (''.join([l for l in open(f)]))

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


def addressBigrams(addr):
    addr_label = {}
    
    for ad in addr:
        
        for i,a in enumerate(ad[:-1]):
            bigram = ' '.join(ad[i:i+2])
            addr_label[bigram] = 1
        
        addr_label[' '.join(ad[:2])] = 0
        addr_label[' '.join(ad[-2:])] = 2
    
    return addr_label


def addressBigrams1(addr,addr_label):
        for i,a in enumerate(addr[:-1]):
            bigram = ' '.join(addr[i:i+2])
            addr_label[bigram] = 1
        
        addr_label[' '.join(addr[:2])] = 0
        addr_label[' '.join(addr[-2:])] = 2


def read_taggedFiles(path):
    tagged = path
    lis = (readF(tagged))
    #entities = np.array([token.split('/')[-1] for text in list(lis) for token in text.split()])
    regex = re.compile('\/[A-Z]*')
    textlist = [regex.sub('',t) for t in list(lis)]
    #print textlist
    
    #entities = categorical(entities,drop=True).argmax(1)
    entities = [[token.split('/')[-1]*(1+len(tokenizer.tokenize(''.join(token.split('/')[:-1])))) for token in text.split()] for text in list(readF(tagged))]
    
    
    entities = [categorical(np.array(t),drop=True).argmax(1) for t in entities]
    return textlist,entities

textlist,entities = read_taggedFiles('WP_output2/*.txt')






def FeatureToken(index,tokens,x,y,addressDict):
    bigrams = []
    dict_Bigrams1 = {}
    count_bigram = collections.defaultdict(int)
    
    for i,token in enumerate(tokens[:-1]):
        bigram = tokens[i:i+2]
        #if i == 0:
        #lookup = ' '.join(tokens[:2])
        #else:
        #lookup = ' '.join(bigram[:-1])
        
        bigram_ = ' '.join(bigram)
        count_bigram[bigram_]+=1
        if bigram_ in addressDict:
            y.append(addressDict[bigram_])
        else:
            y.append(3)
        
        if bigram_ not in dict_Bigrams1:
            #dict_Bigrams1[bigram_]=np.array(featureExtract(bigram[0])+[entities[index][i]]+featureExtract(bigram[1])+[entities[index][i+1]])
            dict_Bigrams1[bigram_]=np.array(featureExtract(bigram[0])+featureExtract(bigram[1]))
        x.append(dict_Bigrams1[bigram_])
        bigrams.append(bigram_)
    return bigrams,count_bigram

def addressLabelDict(path):
    label = pd.read_csv(path)
    tmp_dict = dict(zip(label['Article Index'],label['Addresses']))
    addr_label = {}
    
    for k,v in tmp_dict.items():
        l = v.rstrip().split(';')
        tmp = {}
        for d in l:
            addressBigrams1(tokenizer.tokenize(d),tmp)
            #addressBigrams1(d.split(),tmp)
        addr_label[k] = tmp
        #addr_label[k] = [addressBigrams([d for a in l for d in tokenizer.tokenize(a)])
    return addr_label

def createBigramList(lis):
    bigrams = []
    count_bigrams = []
    
    tmpset = set()
    outX = []
    Y = []
    addr_label = addressLabelDict('WP_address.csv')
    address_dict = {}
    
    for i,l in enumerate(lis):
        tokens = l.split()
        #tokens = tokenizer.tokenize(l)
        #bigrams.append(FeatureToken_noLabel(tokens,x)
        if i in addr_label:
            address_dict = addr_label[i]
        bigrams_,count_bigram=(FeatureToken(i,tokens,outX,Y,address_dict))
        bigrams+=bigrams_

        #print count_unigram

        count_bigrams.append(count_bigram)
    
    return bigrams,count_bigrams,outX,Y


bigramList,count_bigrams,outX,Y = createBigramList(textlist)


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
est = RF_model(np.array(outX),np.array(Y))



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
                    result.append(tmp)
                else:
                    diff = len(t)
                    t_ = tmp.split()
                    if t_[-3].upper() in st_abbr_val or t_[-2].upper() in directions:
                        result.append(tmp[:-diff])
            
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
out = est.predict(outX)
out_result = TokenPredsequence(out,bigramList)

print set(out_result)

