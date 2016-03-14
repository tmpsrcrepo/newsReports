import pandas as pd
import re
import string
import collections
import nltk
import glob
from preprocessing_text import *

#from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TreebankWordTokenizer

#tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
tokenizer = TreebankWordTokenizer()
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

def read_taggedFiles(path):
    tagged = path
    lis = (readF(tagged))
    #entities = np.array([token.split('/')[-1] for text in list(lis) for token in text.split()])
    regex = re.compile('\/[A-Z]*')
    textlist = [preprocessingFunc.cleanText(regex.sub('',t),split_=False) for t in list(lis)]
    
    print textlist[0]
    
    #entities = categorical(entities,drop=True).argmax(1)
    #entities = [[mapping[token.split('/')[-1]]*(1+len(tokenizer.tokenize(''.join(token.split('/')[:-1])))) for token in text.split()] for text in list(readF(tagged))]
    return textlist


def extend(start,end,text,k):
    substr = tokenizer.tokenize(text[start:end+1])
    before = tokenizer.tokenize(text[:start])
    after = tokenizer.tokenize(text[end+1:])
    
    #substr = text[start:end+1].split()
    #before = text[:start].split()
    #after = text[end+1:].split()
    
    out = before[-k:]+substr+after[:(k-1)]
    #print
    #print ' '.join(out)
    #print
    if len(before)<k:
        return ' '.join(['_']*(k-len(before))+out)
    
    return ' '.join(out)


def expandWindow(text,addr,k):
    pattern = re.compile(addr[:-1])
    pairs = [(match.start(),match.end()) for match in re.finditer(pattern, text)]
    return [extend(start,end,text,k) for (start,end) in pairs]


def findWindows(file_path,address_path,k):
    textlist = read_taggedFiles(file_path)
    labels = pd.read_csv(address_path)
    out = collections.defaultdict(list)
    for i,v in dict(zip(labels['Article Index'],labels['Addresses'])).items():
        l = v.rstrip().split(';')
        
        l = [preprocessingFunc.cleanText(a,split_=False) for a in l]
        #print tmp
        out[i] +=[expandWindow(textlist[i],addr,k) for addr in l if addr]
    return out


#out = findWindows('WP_output2/*.txt','WP_address.csv',1)
#out1 = [preprocessingFunc.cleanText(o,tokenizer) for o in out for w in o]

#print out[249]
#print out[249]



