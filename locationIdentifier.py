# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 20:30:16 2015

@author: Xiaoqian
"""
import os   
import numpy as np
#from HTMLParser import HTMLParser
from collections import defaultdict
from bs4 import BeautifulSoup
import pandas as pd
from googlemaps import GoogleMaps

#v1 (then timeit)


def identifyLocation(file_,list_Name):
    list_Address = defaultdict(list)
    #split each sentence by space
    line = file_.split()
    #if road, street, avenue, building, hall, school is found, backtrack previous words
    i=0
    while i<len(line):
        for name in list_Name:
            streetlist=[]
            #if find, 
            if line[i] == name:
                streetlist.append(name)
                tmp=i-1
                while not line[tmp][0].isupper():
                    
                    word = line[tmp]
                    if word[0].isupper():
                        streetlist.append(word)
                        tmp=tmp-1
                    if word[0].isdigit():
                        streetlist.append(word)
                        break
                if line[tmp][0].isupper():
                    streetlist.append(line[tmp])
                    
                list_Address[len(list_Address)].append(streetlist)
                     
        i+=1
            
    return list_Address
    
a='The attack occurred in the Venable area between 15th Street NW and Rugby Road , off of Grady Avenue , shortly after 4 a.m .'
a='University Police Sgt. Thomas Durrer said two males sexually assaulted a female student Friday night between 8:30 p.m. and 10:30 p.m. near the Albert H. Small Building in the Engineering School .'
identifyLocation(a,['Road','Street','Avenue','Building','Hall','School'])

########Need a function here###############
#remove prepositions (from the end of list, and revert the list)


def identifyTags(file_tagged):
    list_locations = defaultdict(str)
    list_date = defaultdict(str)
    list_time = defaultdict(str)
    i = 0
    for line in file_tagged:
        #use BeautifulSoup to read
        s = BeautifulSoup(line)
        loc = s.find('location')
        time = s.find('time')
        date = s.find('date')
        
        if loc:
            list_locations[i]=loc.text
        if date:
            list_date[i]=date.text
        if time:
            list_time[i]=time.text
        i+=1
    return list_locations,list_date,list_time
    

def Findlatlong(street,others):
    #call Google API
    gmaps = GoogleMaps('AIzaSyCg9iHfQ8YE1givl-cGKqkPPQUM3PVVg_o')
    lat,lng = gmaps.address_to_latlng(street+','+others)
    return lat,lng


def createDF():
    df = pd.DataFrame(columns=('file_index','street','date','city','state','country','long','lat'))
    #when populate the data frame
    #if the time is close
    return df
