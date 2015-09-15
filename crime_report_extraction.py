# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 03:42:11 2015

@author: Xiaoqian
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 21:24:53 2015

@author: Xiaoqian Liu
"""
import re
import glob
import numpy as np
import pandas as pd
from identifyLocation_regex import *
import geopy
from urllib import urlopen

#==============================================================================
#split a list of HTML documents into multiple documents (individual reports)
#==============================================================================
def splitHTMLFiles(directory):
    listCrimeReports = glob.glob(directory)
    dict_files = {}
    index = 1
    for file_ in listCrimeReports:
        #convert HTML to raw text 
        html = urlopen(file_).read()    
        raw_txt = nltk.clean_html(html) 
        #regex
        pattern = re.compile(r'[0-9]* DOCUMENTS') 
        find_ = pattern.finditer(raw_txt)
        locations = [m.span() for m in find_ if m]
        #split the document
        start,end = locations[0][1],0
        for loc in locations[1:]:
            end = loc[0]
            dict_files[index]=raw_txt[start:end]
            start = loc[1]
            index+=1
    return dict_files

#==============================================================================
# Location and date lookup
# 1. location (regex)
# 2. date (e.g. Oct 10) 
# 3. concatenate title (region name) and location
# 4. geocoding
#==============================================================================
def findDateLocation(file_text):
      #find date
    date_pattern_ = re.compile(r'[A-Z][a-z]*(\.) [0-9][0-9]*') 
    date_loc = date_pattern_.finditer(file_text)
    date_location = [m.span() for m in date_loc if m]
    return date_location

def findDateTimeLocation(file_text):
    time_pattern = re.compile(r"\d{1,2}:\d{1,2} *(?:a|p)")
    time_loc = time_pattern.finditer(file_text)
    time_location = [m.span() for m in time_loc if m]
    return time_location

def getLocations(file_text):
    p=re.compile(r'[A-Z][a-z]{1,}(?= State Police)')
    loc_find = p.findall(file_text)
    state = []
    if len(loc_find) > 0:
        state =  [loc_find[0]]
    
    #uppercase:
    result= set(state+re.compile('\r?\n *([A-Z][a-z]{1,}(?= *\r?\n))').findall(file_text)
    +re.compile(r'[A-Z][a-z]{1,}(?= County)').findall(file_text))
    return result

#if any locations are mentioned
def findText1(file_text):
    return re.compile(r'SEXUAL [A-Z]{1,} \r?\n .* (?=\r?\n *[A-Z]{1,})').findall(file_text)


def formatAddress(address_info):
    if len(address_info) > 1 and address_info[-1]!='':
        address_info = address_info[1].strip()+','+address_info[0].strip()
        address_info = address_info.split(',',2)[1]
    else:
        address_info = address_info[0].strip()
    return address_info.strip('. ')
    
#==============================================================================
# Location look up
#==============================================================================
def info_lookup(file_text):
    #the format in crime report is like:
    #address, block number, time, date
    regions = getLocations(file_text)
    col_infotext = []
    col_date = []
    col_time = []
    col_address = []
    col_info_ = []
    
    if len(regions) > 0:
        col_infotext = findText1(file_text)
        for infotext in col_infotext:
            
            infotext = infotext.split('\n ')[1]
            #get time location
            date_location = findDateLocation(infotext)
            time_location  = findDateTimeLocation(infotext)

            #previous location: address + block(optional) + time(optional) +time
            address_info = ''
            date = ''
            time = ''
            if len(time_location) > 0:
                time_loc = time_location[0]
                time = infotext[(time_loc[0]):(time_loc[1]+2)].strip()
                address_info = (infotext[:time_loc[0]]).strip().split(',')
            if len(date_location) >0:
                date_loc = date_location[0]
                date = infotext[date_loc[0]:(date_loc[1])].strip()
                address_info = (infotext[:date_loc[1]]).strip().split(',')
                infotext = infotext[date_loc[1]+1:].strip()

            
            #append
            col_date.append(date)
            col_time.append(time)
            #remove all the empty space
            if address_info != '':
                address_info = formatAddress(address_info)
                
            col_address.append(address_info)
            col_info_.append(infotext)

                    
    return regions,col_info_,col_date,col_time,col_address

#==============================================================================
# geocoding
#==============================================================================
#testaddr = 'Solomons Island Rd, calvert county'
#types of geocodes
# (so if the address can't be found, then use another kind of geocoder)
def geocoder(col_address,col_regions):
    col_lat = []
    col_long = []
    col_addr_correct = []
    for pair in zip(col_address,col_regions):
#        geocoder_google = geopy.geocoders.GoogleV3(api_key=None, 
#          domain='maps.googleapis.com', scheme='https', client_id=None, 
#          secret_key=None, timeout=1, proxies=None) #faster than nominatim
#        
        geocoder = geopy.geocoders.Nominatim()
        result = FindGeocode(testaddr,'',geocoder)
        col_lat.append(result[0])
        col_long.append(result[1])
        col_addr_correct.append(result[2])
    return col_lat,col_long,col_addr_correct
#        FindGeocode(testaddr,'',geocoder_google)

#==============================================================================
# Store results to pandas dataframe
#==============================================================================
def createDF(file_text):
    output = info_lookup(file_text)
    if output:
        col_regions,col_infotext,col_date,col_time,col_address= output
        col_lat,col_long,col_addr_correct = geocoder(col_address,col_regions)
        if len(col_lat) > 0 & len(col_long) >0 & len(col_addr_correct) >0:
            #sexual assaults data frame (sa_df)
            sa_df = pd.DataFrame({'Date': col_date,'Time':col_time,'Region':col_regions,
                               'Text':col_infotext,'Address':col_address,
                               'Lat':col_lat, 'Long':col_long,'Corrected_Address':col_addr_correct})
            return sa_df
    else:
        return pd.DataFrame()

def createDF1(file_text,index):
    col_regions,col_infotext,col_date,col_time,col_address= info_lookup(file_text)
    if len(col_regions)>0:
        index_ = np.ones(len(col_date))*index
        sa_df = pd.DataFrame({'Doc_Index':index_,'Date': col_date,'Time':col_time,'Region':','.join(r for r in col_regions),
                           'Text':col_infotext,'Address':col_address})
        return sa_df
    else:
        return pd.DataFrame(columns=['Doc_Index','Date','Time','Region','Text','Address'])    

def mergeDFs(diretory):
    dict_files = splitHTMLFiles(directory)
    list_df = []
    for key,file_text in dict_files.items():
        df_ = createDF1(file_text,key)
        if len(df_) > 0:
            list_df.append(df_)
    return pd.concat(list_df)
    

if __name__ == '__main__':
    directory = "/crime reports/selected/*.html"
    df = mergeDFs(directory)
    print df.head(10)
    df.to_csv('/output/sexual_assaults_dataframe.csv',
    sep=',',index=False)
