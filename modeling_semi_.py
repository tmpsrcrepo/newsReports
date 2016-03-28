import pystruct.models
from pystruct.models import GraphCRF,ChainCRF
import numpy as np
import pandas as pd
#from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from model import predictive_model,predictive_model_CRF


def loadData():
    import cPickle
    trainX = cPickle.load(open('WP_trainX_CRF_window_1.p','rb'))
    trainY = cPickle.load(open('WP_trainY_CRF_window_1.p','rb'))
    types = cPickle.load(open('WP_types_CRF_window_1.p'))

loadData()

def selectEntities(trainX,trainY,types,sample_factor):
    ne = [i for i,t in enumerate(types) if t>1]
    o = [i for i,t in enumerate(types) if t==1]
    #select o: first neglect sentence w/ less than two words
    o = [i for i in o if len(trainY[i])>2]
    #then sample 2*count of ne
    import random
    o = random.sample(range(len(o)), len(ne)/sample_factor)
    return [trainX[i] for i in ne+o],[trainY[i] for i in ne+o]


trainX,trainY =selectEntities(trainX,trainY,types,1)


xlen = len(trainX[0][0])
trainX = selectFeatures()



est = predictive_model_CRF(np.array(trainX),np.array(trainY),2)




#trainX = cPickle.load(open('WP_trainX_window_1.p','rb'))
#trainY = cPickle.load(open('WP_trainY_window_1.p','rb'))

est = predictive_model(np.hstack(trainX),np.hstack(trainY))


trainX_unlabel = 'WP_1516_trainX_window_1.p'

k = 3 #partition numbers
