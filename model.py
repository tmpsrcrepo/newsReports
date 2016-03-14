import numpy as np
import pandas as pd
#from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.pipeline import _name_estimators
from sklearn.ensemble import VotingClassifier
from ensemble import *
import xgboost as xgb
from sklearn.semi_supervised import label_propagation
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import cPickle
import pystruct.models
from pystruct.models import GraphCRF
from pystruct.learners import NSlackSSVM
import itertools
from pystruct.learners import OneSlackSSVM
from sklearn import preprocessing

def cv(clfs_,X,Y):
    df = pd.DataFrame(columns=('w1', 'w2', 'mean', 'std'))
    i = 0
    for w1 in [4,2,1]:
        for w2 in [1,4,2]:
            if w1==w2:
                continue
            eclf = EnsembleClassifier(clfs=clfs_,voting='soft',weights=[w1,w2])
            scores = cross_validation.cross_val_score(estimator=eclf,X=X,y=Y,cv=5,scoring='log_loss',n_jobs=2)
            print i,scores
            df.loc[i] = [w1,w2,scores.mean(),scores.std()]
            i+=1
    df.sort(columns =['mean','std'],ascending=False)
    print df
    return df['w1'][0],df['w2'][0]



#def gbm_model(trainX,trainY):
#    gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
#    predictions = gbm.predict(test_X)

def xgb_model(x_train, x_test, y_train, y_test):
    dtrain = xgb.DMatrix( x_train, label=y_train)
    del x_train
    dtest = xgb.DMatrix( x_test, label=y_test)
    del x_test
    param = {}
    param['eta'] = 0.1
    param['max_depth'] = 10
    param['silent'] = 1
    param['num_class'] = 4
    param['objective'] = 'multi:softmax'
    param['nthread'] = 2
    param['n_estimators']=100
    #param['eval_metric'] = 'auc'
    plst = param.items()
    
    watchlist = [ (dtrain,'train'), (dtest, 'test') ]
    
    #evallist  = [(dtest,'eval'), (dtrain,'train')]
    num_round = 237
    bst = xgb.train( param, dtrain, num_round)
    #xgb.plot_importance(bst)
    
    #rf = RandomForestClassifier(n_estimators=3000, max_depth=10,n_jobs=2)
    #pipe1 = Pipeline([('sel',ColumnSelector(range(col_count))),('clf',bst)])
    #pipe2 = Pipeline([('sel',ColumnSelector(range(col_count[:-4]))),('clf',rf)])
    
    y_pred = bst.predict(dtest,ntree_limit=bst.best_ntree_limit)
    target_names = ['Start','Mid','End','Others']
    #eclf = EnsembleClassifier(clfs=[pipe1, pipe2],voting='soft',weights=[0.5,0.2])
    #eclf.fit(x_train,y_train)
    #y_pred = eclf.predict(x_test)
    print classification_report(y_test, y_pred, target_names=target_names)
    return bst

def xgb_semi_supervised(trainX,trainY,X_unlabeled,Y_unlabeled):
    row_count =trainX.shape[0]
    trainX = np.hstack((trainX,np.array(word_dist_list).reshape(row_count,1)))
    trainX = np.hstack((trainX,np.array(time_dist_list).reshape(row_count,1)))
    
    row_count =X_unlabeled.shape[0]
    X_unlabeled = np.hstack((X_unlabeled,np.array(word_dist_list_unlabeled).reshape(row_count,1)))
    X_unlabeled = np.hstack((X_unlabeled,np.array(time_dist_list_unlabeled).reshape(row_count,1)))
    
    X_unlabeled,_,Y_unlabeled,_ = train_test_split(X_unlabeled, Y_unlabeled, test_size=0.85, random_state=20)
    
    x_train, x_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.25, random_state=20)
    
    #concatenate x_train,y_train w/ x_unlabeled and y_unlabeled repectively
    x_ = np.concatenate((x_train,X_unlabeled),axis=0)
    x_ = sparse.csr_matrix(x_)
    y_ = np.concatenate((y_train,Y_unlabeled),axis=0)
    #y_ = sparse.csr_matrix(y_)
    
    #unlabeled_indices = np.arange(x_shape[0])[x_train.shape[0]:]
    
    label_prop_model = label_propagation.LabelSpreading(kernel='knn', alpha=1.0)
    label_prop_model.fit(x_.toarray(),y_)
    y_pred = label_prop_model.transduction_
    
    #y_ = label_prop_model.predict(x_)
    xgb_model(x_,x_test,y_pred,y_test)


def multiClf(x_train, x_test, y_train, y_test):
    #lb = preprocessing.LabelBinarizer()
    #y=y_train.reshape((1,y_train.shape[0]))
    #lb.fit(y_train)
    #y=lb.transform(y_train)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    #full = np.vstack([x for x in itertools.combinations(range(4), 2)])
    clf = pystruct.models.MultiClassClf(n_features=x_train.shape[1],n_classes=4)
    ssvm = NSlackSSVM(clf, C=.1, tol=0.01)
    
    ssvm.fit(x_train,y_train)
    y_pred = clf.predict(np.array(x_test))
    target_names = ['Start','Mid','End','Others']
    #eclf = EnsembleClassifier(clfs=[pipe1, pipe2],voting='soft',weights=[0.5,0.2])
    #eclf.fit(x_train,y_train)
    #y_pred = eclf.predict(x_test)
    print classification_report(y_test, y_pred, target_names=target_names)


def CRF_oneNode(x_train, x_test, y_train, y_test):
    pbl = GraphCRF(n_states = 4,n_features=20)
    svm = NSlackSSVM(pbl,max_iter=200, C=10,n_jobs=2)
    
    svm.fit(x_train,y_train)
    y_pred = svm.predict(x_test)
    target_names = ['Start','Mid','End','Others']
    #eclf = EnsembleClassifier(clfs=[pipe1, pipe2],voting='soft',weights=[0.5,0.2])
    #eclf.fit(x_train,y_train)
    #y_pred = eclf.predict(x_test)
    print classification_report(y_test, y_pred, target_names=target_names)


def predictive_model(trainX,trainY):
    #est = RandomForestClassifier(n_estimators=1000, max_depth=10)
    from sklearn.decomposition import PCA
    comp = int(trainX.shape[1]*0.7)
    print comp
    trainX = np.array(PCA(n_components=comp).fit_transform(trainX))
    #PCA:
    #pca
    #75% train set and 25% trainset
    x_train, x_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.25, random_state=20)
    
    
    xgb_model(x_train,x_test,y_train,y_test)
    #multiClf(x_train,x_test,y_train,y_test)
    
    
    #x_train = x_train[:,xrange(col_count-2)]
    
    #print est.feature_importances_
    
    target_names = ['Start','Mid','End','Others']


    #combo = [(trainX,est),(trainX_,est),(trainX_,svmCLF)]
    #for pair in combo:
    #scores = cross_validation.cross_val_score(pair[1], pair[0], trainY, cv=5, scoring='accuracy')
    #print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    #x_train = sparse.csr_matrix(x_train).toarray()
    #y_train = sparse.csr_matrix(y_train).toarray()
    
    #gbm = xgb.XGBClassifier(**param).fit(x_train, y_train)
    #svm = SVC(C=10,probability=True,kernel='linear')
    #svm = SVC(C=10,probability=True,kernel='linear',decision_function_shape='ovr')
    #pipe1 = Pipeline([('sel',ColumnSelector(range(col_count))),('clf',svm)])
    #pipe2 = Pipeline([('sel',ColumnSelector(range(col_count))),('clf',gbm)])
    #pipe3 = Pipeline([('sel',ColumnSelector(range(col_count-3)+[col_count-1])),('clf',est)])
    #est.fit(x_train,y_train)
    #y_pred = est.predict(x_test)
    
    #w1,w2 = cv([pipe1, pipe2],trainX,trainY)
    #eclf = EnsembleClassifier(clfs=[pipe1, pipe2],voting='soft',weights=[0.2,0.5])
    #eclf.fit(x_train,y_train)
    #eclf = gbm
    #eclf.fit(x_train,y_train)
    #y_pred = eclf.predict(x_test)
    #print classification_report(y_test, y_pred, target_names=target_names)

    #fit the full model
    #est.fit(trainX,trainY)
    #scores = cross_validation.cross_val_score(est, trainX, trainY, cv=10)
    #print scores.mean(),scores.std()*2
    #return est


#print np.array(X).shape
#trainX_path='trainX.p'
#trainY_path='trainY.p'
trainX_path = 'trainX_window_1.p'
trainY_path = 'trainY_window_1.p'
est = predictive_model(np.array(cPickle.load(open(trainX_path,'rb'))),np.array(cPickle.load(open(trainY_path,'rb'))))
#est = xgb_semi_supervised(np.array(outX),np.array(Y),np.array(outX_unlabeled),np.array(Y_unlabeled))
#multiclassCRF((np.array(cPickle.load(open('trainX.p','rb'))),np.array(cPickle.load(open('trainY.p','rb'))))



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



    return result

#then check if street abbr is contained in the retrieved addresses
#out = est.predict(outX)
#out_result = TokenPredsequence(out,bigramList)

#print set(out_result)

