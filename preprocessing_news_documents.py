# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 13:00:40 2015

@author: Xiaoqian
"""

#import re  
import os   
import numpy as np
from HTMLParser import HTMLParser
from collections import defaultdict

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()
    
    
#news report object
class NewsObj:
    def __init__(self):
        self.date = None
        self.title = None
        self.year = None
        self.time = None
        self.content = ''

#look up table
lookup=defaultdict(int)

#May 25 - June 1
#news = open('1-1000_jun2_may1.TXT','r')

def readFile(report,endLine,textTitle):
    news = open(textTitle,'r')
    allnews = []  
    for line in iter(news):
        if report+'\r\n' in line:
        #if 'Washingtonpost.com\r\n' in line:
            newsobj = NewsObj()
            
            line = next(news)
            while ',' not in line:
                
                line = next(news)
            #Date
            a=line[:-2].split(', ')
            newsobj.date = a[0].strip()
            a = a[1].split(' ')
            newsobj.year = a[0]
            #newsobj.time = a[2]+' '+a[3]
            
            line = next(news)
            while line[:8]!='HEADLINE':
                line = next(news)
                #get title
            
            newsobj.title = line[10:].rstrip('\r\n')
            print newsobj.title
                #i+=1
                    #if i==3:
            while line[:4]!='BODY':
                line = next(news)
                
            #target = open(report+'file_'+str(len(allnews))+'.txt','w')
            #target.write(line[5:])
            line = (next(news))
            
            #while line[-11:]!='DOCUMENTS\r\n':
            
            
            while line[-11:]!=endLine+'\r\n':
                line = line.strip().rstrip('\r\n')
                if line!='':
                    newsobj.content=newsobj.content+line
                    #target.write(' '+line)
                line = next(news)
            #target.close()
            allnews.append(newsobj)
    return allnews
        
                    
allnews=readFile('University Wire','DOCUMENTS','keywords_assaults_cavdaily.TXT')                
dcnews =readFile('The Washington Post','DOCUMENTS','washingtonpost_keywords.TXT')            
        
#write titles
def writeTitles(filename,dict_):
    target = open(filename,'w')
    for news in dict_:
        title = news.title
        if title!=None:
            if not title[0].isalpha():
                title = title[1:]
            
            target.write(title+'\n')
    
    target.close()


def writeContents(directory,dict_):
    index = 0
    for news in dict_:
        target = open(directory+str(index)+'.txt','w')
        target.write(news.content)
        target.close()  
        index+=1

target = writeTitles('University Wire_'+'titles_.txt',allnews)
targetdc = writeTitles('Washington Post_'+'titles_.txt',dcnews)
        
writeContents('WP/Washington Post_',dcnews)

path = r'Output/'
data =  defaultdict(int)

ls_title_entities = defaultdict(int)
title_output = open('title1.txt','r')
for title in title_output:
    if title!=None:
        if title[0].isdigit():
            num = int(title.rstrip('\n'))
            ls_title_entities[num]=(next(title_output).rstrip('\n'))


def identifyKeywd(line,keywords):
    #keywords
    #cumulative count
    count=0
    for i in keywords:
        if i in line:
            count+=1
    return count
            
    

def identifyTags(file_):
    ls_ner = defaultdict(list)
    ls_tf = defaultdict(int)
    
    index = 0
    for line in file_:
        i = 0
        #line = line.rstrip('\n')
        while i < len(line):
            if line[i]=='<':
                tuple1 = ()
                entity = ''
                word = ''
                while line[i]!='>':
                    
                    entity+=line[i+1]
                    i+=1
                    
                end = i
                while line[i]!='<':
                    
                    word += line[i+1]
                    i+=1
                
                
                word = word.rstrip('<')
                entity = entity.rstrip('>')
                #addDict(entity+':'+word,ls_tf)
                ls_tf[entity+':'+word]+=1
                # tuple = (entity, start, freq, end, sentence index,sentence content)
                tuple1=(entity, end+1,i-1,index,line.rstrip('\n'))
                #add
                ls_ner[word].append(tuple1)
                #addDictList(word,tuple1,ls_ner)
                
            i+=1
            index+=1
    #since the last line is always (C) Year Cavalier Daily
    #remove it
    return ls_ner,ls_tf
    
def readListFiles(path):
    index = 0
    dict_Tuples =defaultdict(dict)
    dict_TF = defaultdict(dict)
    
    for dir_entry in os.listdir(path):
        dir_entry_path = os.path.join(path,dir_entry)
        if os.path.isfile(dir_entry_path):
            with open(dir_entry_path,'r') as file_:
                #title_entity = ''
                #if ls_title_entities.__contains__(index):
                #    title_entity = ls_title_entities[index]
                dict_Tuples[index],dict_TF[index]=identifyTags(file_)
                #dict_Tuples[index]=file_.read()
                file_.close()
                index+=1    
    #return dict_Tuples
    return dict_Tuples,dict_TF
    
            
dict_Tuples_dc,dict_TF_dc = readListFiles(r'WP_output/')
dict_Tuples, dict_TF = readListFiles(r'Output/')


def searchKSentence(file_,kwd,k):
    index=0
    dict_sentences=defaultdict(str)
    sent_kwd = []
    #find sentences containing keywords
    for line in file_:
        line = line.lower()
        dict_sentences[index]=line
        for k in kwd:
            if k in line:
                sent_kwd+=(range(index-k,index+k))
        index+=1        
    
    sent_kwd = np.array(sent_kwd)
    sent_kwd[(sent_kwd>=0) & (sent_kwd<len(dict_sentences))]
    sent_kwd = np.unique(sent_kwd)
    
    #delete elements not in sent_kwd
    for k in dict_sentences:
        if k not in sent_kwd:
            del dict_sentences[k]
    return dict_sentences
    
