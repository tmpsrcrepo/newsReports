# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 00:02:52 2015

@author: Xiaoqian
"""
from collections import defaultdict
import pandas as pd
import re
import numpy as np
import requests
#import time
from newspaper import Article
from geopy.geocoders import Nominatim


def getColnames(file_):
    colnames =[]
    #only one line 
    for line in file_:
        colnames = line.split('\t')
    colnames[-1]=colnames[-1].rstrip('\r\n')
    return colnames

def filterData(old,country):
        new = old[(old['ActionGeo_CountryCode'].values=='US')]
        new = new[['ActionGeo_FullName','ActionGeo_Lat','ActionGeo_Long','DATEADDED','SOURCEURL']]
        new = new.drop_duplicates(['ActionGeo_FullName','ActionGeo_Lat','ActionGeo_Long','DATEADDED'])
        return new
        
#==============================================================================
# Selected method
def readArticle(url_,urlList): #input = url2015[11]
    article = Article(url_)
    article.download()
    article.parse()
    #return text and index
    return (article.text,int(np.where(urlList==url_)[0]))

def narrowNews(type_,keywordPattern,urlList):
    p2 = re.compile(type_,re.IGNORECASE)   #sex = type of crime
    p = re.compile(keywordPattern,re.IGNORECASE)  #keyword = keyword pattern
    #new_url1 = a list of urls & '-'
    new_url1= np.array([u if (re.findall(p,u) and re.findall(p2,u)) else '-' for u in urlList])
    
    indexList = np.array(np.arange(0,len(new_url1)))
    #index list of urls
    indexList = indexList[new_url1!='-']
    #list of final urls (with actual contents)
    final_url  =new_url1[(new_url1!='-')]
    return new_url1,indexList,final_url
        
def identifyLocation1(file_,pattern):
    list_string = []
    p = re.compile(pattern,re.IGNORECASE)
    find_ = p.finditer(file_)
    
    if find_:
        locations = [m.span() for m in find_ if m]
        #if road, street, avenue, building, hall, school is found, backtrack previous words
        start = 0
        end = 0
        for sp in locations:
            start = sp[0]
            street=''
            word = ''
            name_end = file_[start:sp[1]+1]
            f = file_[end:sp[0]]
            
            start = len(f)-2
            while start> 0:          
                if f[start]==' ' and word!='':
                    if word[0].islower():
                        break
                    if word[0].isdigit():
                        street =  word +' '+street
                        break
                    if word[0].isupper():
                        street =  word +' '+street
                        start-=1
                        word = ''
                else:
                    word = f[start]+word
                    start-=1
            #add to the dictionary
            street += name_end
            if len(street)-1 >len(name_end):
                list_string.append(street[:-1])
            end = sp[1]
            
        return list_string

    
def Findlatlong(street,others,geolocator):
    #call geolocator API
    print street+','+others
    location = geolocator.geocode(street+', '+others)
    if location:
        return (location.latitude, location.longitude)
    else:
        return (0,0)


def returnAddress(columnPath,dfPath,keyword_type,pattern):
    colnames_2013 = getColnames(open(columnPath,'r'))
    #==============================================================================
    # Import the data  
    #==============================================================================
    #gdelt data from 1979 to 2013
    df15712 = pd.read_csv(dfPath,sep='\t',dtype='unicode')
    df15712.columns = colnames_2013    

    #==============================================================================
    # Extract US data / rows with URL's
    #==============================================================================
    #drop duplicates:
    US_df = filterData(df15712,'US')
    #open a random url and see the content
    url2015 = US_df['SOURCEURL'].values    
    
    #==============================================================================
    # #narrow down to sexual-assault, create an index list to store indices
    #==============================================================================    
    url_sex1,indexList,url_sex=narrowNews(keyword_type,pattern,url2015)
    
    #==============================================================================
    # #METHOD for reading articles########
    #==============================================================================
    
    articlelist = np.array([readArticle(url_,url_sex1)  for url_ in url_sex if requests.get(url_).ok ])
    #update indexlist
    articleList,indexList = zip(*articlelist)
    
    #==============================================================================
    # Identify street addresses from those articles relevant w/ sexual assaults
    #==============================================================================
    results = defaultdict()
    idx = 0
    for text_ in articlelist:
        results[indexList[idx]]=set(identifyLocation1(''.join(text_),'Plaza|Road|Street|Avenue|Building|Hall|School|Hotel|Restaurant|Mall|Center|Bar'))
        idx +=1
    
        
    # #find the row from the original data set
    StreetIndex = ([key for key,val in results.items() if len(val)>0])
    FinalRows = US_df.iloc[StreetIndex]
    
    street_ = np.array(map(lambda x: ','.join(list(x)) if len(x)>0 else None, (results.values())))
    street_ = (street_[street_ != np.array(None)])
    
    FinalRows.insert(len(FinalRows.columns),'Street_',(street_))
    StreetC= FinalRows['Street_'].values
    
    #==============================================================================
    # Concatenate the street addresses with the city of the event 
    #==============================================================================
        
    Regions = FinalRows['ActionGeo_FullName'].values
    
    #geocoding    
    geolocator = Nominatim()
    #output 
    l =np.array([Findlatlong(j,Regions[i],geolocator) for i in range(0,len(StreetC)) for j in StreetC[i].split(',') ])
    StreetLat,StreetLong = zip(*l)
    FinalRows.insert(len(FinalRows.columns),'StreetLat',(StreetLat))
    FinalRows.insert(len(FinalRows.columns),'StreetLon',(StreetLong))

    return FinalRows
    
    
    
if __name__ == '__main__':
    output,FinalRows=returnAddress('CSV.header.after2013.txt','20150712.export.CSV','sexual','assault|rape')
    