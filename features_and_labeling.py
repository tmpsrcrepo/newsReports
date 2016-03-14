#from sklearn.semi_supervised import LabelPropagation
import numpy as np
import pandas as pd
from collections import defaultdict
import re,string
#from bs4 import BeautifulSoup
#from urllib import urlopen
import nltk
from nltk.tokenize import RegexpTokenizer
import multiprocessing
from functools import partial
import sys
import time
#from sklearn import cross_validation
#from sklearn.metrics import classification_report
#from sklearn.metrics import accuracy_score
import glob
#from sklearn.cross_validation import train_test_split
from statsmodels.tools import categorical
import collections



k=1

class ColumnSelector(object):
    """
        A feature selector for scikit-learn's Pipeline class that returns
        specified columns from a numpy array.
        
        """
    
    def __init__(self, cols):
        self.cols = cols
    
    def transform(self, X, y=None):
        return X[:, self.cols]
    
    def fit(self, X, y=None):
        return self



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
                int(punctuations in token),int(token.isdigit()),len(DigitLetter.findall(token)),int(email.findall(token)!=None),len(token),int(token[0].isalpha())]
    return features


def addressBigrams1(k,addr,addr_label):
        for i in xrange(k,len(addr)-k):
            bigram = ' '.join(addr[i-k:i+k+1])
            addr_label[bigram] = 1
        
        addr_label[' '.join(addr[:k+2])] = 0
        addr_label[' '.join(addr[-k-2:])] = 2

        #print addr_label
mapping = {'O':0,'ORGANIZATION':1,'LOCATION':2,'PERSON':3,'TIME':4,'DATE':5,'MONEY':6,'PERCENT':7}

def read_taggedFiles(path,k):
    tagged = path
    lis = (readF(tagged))
    #entities = np.array([token.split('/')[-1] for text in list(lis) for token in text.split()])
    regex = re.compile('\/[A-Z]*')
    textlist = [regex.sub('',t) for t in list(lis)]
    #print textlist
    
    #entities = categorical(entities,drop=True).argmax(1)
    entities = [[0]*k+[mapping[token.split('/')[-1]]*(1+len(tokenizer.tokenize(''.join(token.split('/')[:-1])))) for token in text.split()]+[0]*k for text in list(readF(tagged))]
    
    return textlist,entities

textlist,entities = read_taggedFiles('WP_output2/*.txt',k)
#textlist_unlabeled,entities_unlabeled = read_taggedFiles('WP_1516_output/*.txt')


def getWordDistance(tokens,keywords):
    indices = [i for i,w in enumerate(tokens) if w in keywords]
    if indices:
        for i,token in enumerate(tokens[:-1]):
            out = [abs(i-j) for j in indices]
            yield min(out)


def getDistance(tokens,entities,file_index,entity_index):
    indices = [i for i,w in enumerate(tokens) if entities[file_index][i]==entity_index]
    if indices:
        for i,token in enumerate(tokens[:-1]):
            out = [abs(i-j) for j in indices]
            yield min(out)

def FeatureToken(k,index,tokens,entities,x,y,addressDict):
    bigrams = []
    dict_Bigrams1 = {}
    count_bigram = collections.defaultdict(int)
    
    for i in xrange(k,len(tokens)-k):
        bigram = tokens[i-k:i+k+1]
        bigram_ = ' '.join(bigram)
        count_bigram[bigram_]+=1
        if bigram_ in addressDict:
            y.append(addressDict[bigram_])
        else:
            y.append(3)
        
        
        if bigram_ not in dict_Bigrams1:
            features = []
            for j in xrange(len(bigram)):
                features+=featureExtract(bigram[j])
            dict_Bigrams1[bigram_]=np.array(features+[entities[index][i]]+[entities[index][i+1]])
            
            #print len(dict_Bigrams1[bigram_])
        x.append(dict_Bigrams1[bigram_])
        
        bigrams.append(bigram_)
    return bigrams,count_bigram


def addressLabelDict(k,path):
    label = pd.read_csv(path)
    tmp_dict = dict(zip(label['Article Index'],label['Addresses']))
    addr_label = {}
    
    for key,v in tmp_dict.items():
        l = v.rstrip().split(';')
        tmp = {}
        for d in l:
            addressBigrams1(k,tokenizer.tokenize(d),tmp)
            #addressBigrams1(d.split(),tmp)
        addr_label[key] = tmp
        #addr_label[k] = [addressBigrams([d for a in l for d in tokenizer.tokenize(a)])
    return addr_label

def createBigramList(k,lis,entities,address_csv):
    bigrams = []
    count_bigrams = []
    
    tmpset = set()
    outX = []
    word_dist_list = []
    time_dist_list = []
    Y = []
    addr_label = addressLabelDict(k,address_csv)
    address_dict = {}
    
    for i,l in enumerate(lis):
        tokens = ['~']+l.split()
        word_dist = (list(getWordDistance(tokens,['assault','sexual','sex','rape'])))
        time_dist = (list(getDistance(tokens,entities,i,4)))
        
        if not word_dist:
            word_dist_list+=(len(tokens)-1)*[len(tokens)]
        else:
            word_dist_list+=word_dist
        if not time_dist:
            time_dist_list+=(len(tokens)-1)*[len(tokens)]
        else:
            time_dist_list+=time_dist
        tokens = ['~']*(k-1)+tokens+['~']*k
        #tokens = tokenizer.tokenize(l)
        #bigrams.append(FeatureToken_noLabel(tokens,x)
        if i in addr_label:
            address_dict = addr_label[i]
        bigrams_,count_bigram=(FeatureToken(k,i,tokens,entities,outX,Y,address_dict))
        bigrams+=bigrams_

        #print count_unigram

        count_bigrams.append(count_bigram)



    return bigrams,count_bigrams,outX,word_dist_list,time_dist_list,Y




bigramList,count_bigrams,outX,word_dist_list,time_dist_list,Y = createBigramList(k,textlist,entities,'WP_address_window_1.csv')
outX = np.array(outX)
print outX.shape
print np.array(word_dist_list).shape
print set(Y)
outX = np.hstack((outX,np.array(word_dist_list).reshape(outX.shape[0],1)))
outX = np.hstack((outX,np.array(time_dist_list).reshape(outX.shape[0],1)))
import cPickle
cPickle.dump(outX,open('trainX_window_'+str(k)+'.p','wb'))
cPickle.dump(Y,open('trainY_window_'+str(k)+'.p','wb'))
cPickle.dump(bigramList,open('bigramList_window_'+str(k)+'.p','wb'))
print 'dumped'
