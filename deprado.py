# general libraries
import pandas as pd
import numpy as np
from tqdm import tqdm


# ------------------------------------------------------------------- UTILS

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


# ------------------------------------------------------------------- FRACDIFF

# this is the same weights function as for expanding window FFD??
# needs to be changed to fixed width window or use standard fracdiff in this thesis
def getWeights_FFD(d,thres):
    w,k=[1.],1
    while True:
        w_=-w[-1]/k*(d-k+1)
        if abs(w_)<thres:break
        w.append(w_);k+=1
    return np.array(w[::-1]).reshape(-1,1)

def fracDiff_FFD(series,d,thres=1e-5):
    w = getWeights_FFD(d,thres)
    width = len(w)-1
    df = {}
    for name in series.columns:
        seriesF,df_=series[[name]].fillna(method='ffill').dropna(),pd.Series(dtype=float)
        for iloc1 in range(width,seriesF.shape[0]):
            loc0,loc1=seriesF.index[iloc1-width],seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]):continue # exclude NAs
            df_[loc1]=np.dot(w.T,seriesF.loc[loc0:loc1])[0,0]
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df

def getMinFFD(df0):
    '''
    Input: df0 = dataframe with columns of time series
    '''

    from statsmodels.tsa.stattools import adfuller

    statsList = []
    dList = []
    i = 0

    for col in df0.columns:
        i += 1
        print(i,": ", col)
        
        # temp dataframe to store results for adf stats
        stats = pd.DataFrame(columns=['adfStat','pVal','lags','nObs','95% conf','corr'])
        
        # copy column of input df
        df1=df0[col].copy(deep=True)
        df1 = pd.DataFrame(df1)
        df1.columns=['Close']

        # loop over values for differentiation and compute adf stats
        for d in tqdm(np.linspace(0,1,11)):

            # compute fracdiff for given d in loop
            df2=fracDiff_FFD(df1,d,thres=.01)

            # compute correlation between original and differentiated series
            corr=np.corrcoef(df1.loc[df2.index,'Close'],df2['Close'])[0,1] # just for information

            # compute adf stats
            df2=adfuller(df2['Close'],maxlag=1,regression='c',autolag=None) # test if last fracdiff is stationary

            stats.loc[d]=list(df2[:4])+[df2[4]['5%']]+[corr] # with critical value

            if df2[4]['5%'] > stats.iloc[0, 0]:
                break

        # break loop if first adf stat is smaller than 95% conf
        if df2[4]['5%'] > stats.iloc[0, 0]:
            dList.append(0)
            continue
        
        # get integer indeces of intersection space and differences 
        idx0 = np.argwhere(np.array(stats['adfStat']) < stats['95% conf'][0])[0][0]
        idx1 = idx0 - 1
        d0 = stats.index[idx0]
        d1 = stats.index[idx1]

        # line of 95% conf interval
        A = (d0, stats['95% conf'][0])
        B = (d1, stats['95% conf'][0])

        # line of adfStat
        C = (d0, stats['adfStat'].iloc[idx0])
        D = (d1, stats['adfStat'].iloc[idx1])

        # compute intersection of lines
        dOpt = line_intersection((A,B),(C,D))[0]
    
        # append results to list
        statsList.append(stats)
        dList.append(dOpt)

    return statsList, dList


# ------------------------------------------------------------------- CLUSTERING

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

def clusterKMeansBase(corr0, maxNumClusters=10, n_init=10):
    x, silh = ((1 - corr0.fillna(0)) / 2.) ** .5, pd.Series(dtype=np.float64) # eval observations matrix from corr0
    # try different initializations
    for init in range(n_init):
         # try different cluster numbers
        for i in range(2,maxNumClusters+1):
            # perform kmeans
            kmeans_ = KMeans(n_clusters=i, n_init=1)
            kmeans_ = kmeans_.fit(x)
            # compute silhouette score and quality measure
            silh_= silhouette_samples(x,kmeans_.labels_)
            stat = (silh_.mean() / silh_.std(), silh.mean() / silh.std())    
            # keep fit if previous stat was better
            if np.isnan(stat[1]) or stat[0]>stat[1]:
                silh, kmeans = silh_, kmeans_  
    # order results
    newIdx=np.argsort(kmeans.labels_)
    corr1=corr0.iloc[newIdx] # reorder rows
    corr1=corr1.iloc[:,newIdx] # reorder columns
    # rename clusters
    clstrs = {i:corr0.columns[np.where(kmeans.labels_==i)[0]].tolist() for i in np.unique(kmeans.labels_)} # cluster members
    silh = pd.Series(silh, index=x.index)
    return corr1, clstrs, silh

def makeNewOutputs(corr0,clstrs,clstrs2):
    clstrsNew={}
    # build lists for clusters to compare
    for i in clstrs.keys():
        clstrsNew[len(clstrsNew.keys())]=list(clstrs[i])
    for i in clstrs2.keys():
        clstrsNew[len(clstrsNew.keys())]=list(clstrs2[i])
    # build new correlation matrix from clstrsNew
    newIdx = [j for i in clstrsNew for j in clstrsNew[i]]
    corrNew = corr0.loc[newIdx,newIdx]
    # compute distance matrix of new correlation matrix
    x = ((1 - corr0.fillna(0)) / 2.) ** .5
    # make array for new cluster labels
    kmeans_labels = np.zeros(len(x.columns))
    # assign labels of x to labels of kmeans_labels
    for i in clstrsNew.keys():
        idxs = [x.index.get_loc(k) for k in clstrsNew[i]]
        kmeans_labels[idxs] = i
    # compute silhouette scores within x using kmeans_labels
    silhNew = pd.Series(silhouette_samples(x,kmeans_labels), index=x.index)
    return corrNew, clstrsNew, silhNew


def clusterKMeansTop(corr0,maxNumClusters=None,n_init=10):
    if maxNumClusters == None: 
        maxNumClusters = corr0.shape[1] - 1
        # run base clustering on corr0
    corr1, clstrs, silh = clusterKMeansBase(corr0, maxNumClusters=min(maxNumClusters, corr0.shape[1]-1), n_init=n_init)
    # get quality score of each cluster from base clustering
    clusterTstats = {i:np.mean(silh[clstrs[i]]) / np.std(silh[clstrs[i]]) for i in clstrs.keys()}
    # compute mean of quality scores
    tStatMean = sum(clusterTstats.values()) / len(clusterTstats)
        # find subset of clusters with quality score less than mean
    redoClusters = [i for i in clusterTstats.keys() if clusterTstats[i] < tStatMean]
    if len(redoClusters) <= 1:
        return corr1, clstrs, silh # no clusters to redo, return previous base clustering results
    else:
        # build new correlation matrix from clusters to redo
        keysRedo = [j for i in redoClusters for j in clstrs[i]]
        corrTmp = corr0.loc[keysRedo,keysRedo]
        # get stats of actual clusters to redo
        tStatMean = np.mean([clusterTstats[i] for i in redoClusters])
        # run top clustering on new correlation matrix (recursive call)
        corr2, clstrs2, silh2 = clusterKMeansTop(corrTmp, maxNumClusters=min(maxNumClusters, corrTmp.shape[1]-1), n_init=n_init)
            # Make new outputs, if necessary
        corrNew,clstrsNew,silhNew = makeNewOutputs(corr0, {i:clstrs[i] for i in clstrs.keys() if i not in redoClusters}, clstrs2)
                # get new quality scores for redone clusters
        newTstatMean = np.mean([np.mean(silhNew[clstrsNew[i]]) / np.std(silhNew[clstrsNew[i]]) for i in clstrsNew.keys()])
        if newTstatMean <= tStatMean:
            return corr1, clstrs, silh # return previous base clustering results if quality score is worse
        else:
            return corrNew, clstrsNew, silhNew # return new clustering results if quality score is better


# ------------------------------------------------------------------- CLUSTERING METRIC

import numpy as np,scipy.stats as ss
from sklearn.metrics import mutual_info_score

def numBins(nObs,corr=None):
    # Optimal number of bins for discretization
    if corr is None: # univariate case
        z=(8+324*nObs+12*(36*nObs+729*nObs**2)**.5)**(1/3.)
        b=round(z/6.+2./(3*z)+1./3)
    else: # bivariate case
        b=round(2**-.5*(1+(1+24*nObs/(1.-corr**2))**.5)**.5)
    return int(b)

def varInfo(x,y,norm=False):
    # variation of information
    bXY=numBins(x.shape[0],corr=np.corrcoef(x,y)[0,1])
    
    cXY=np.histogram2d(x,y,bXY)[0]
    iXY=mutual_info_score(None,None,contingency=cXY)
    hX=ss.entropy(np.histogram(x,bXY)[0]) # marginal
    hY=ss.entropy(np.histogram(y,bXY)[0]) # marginal
    vXY=hX+hY-2*iXY # variation of information
    if norm:
        hXY=hX+hY-iXY # joint
        vXY/=hXY # normalized variation of information
    return vXY

def varInfoMat(X, norm=False):
    '''Compute VarInfo on whole matrix X'''
    l = X.shape[1]
    metric = np.full([l,l], np.nan)
    for i in range(l):
        for j in range(l):
            if not i == j: 
                metric[i,j] = varInfo(X.iloc[:,i].values, X.iloc[:,j].values, norm=norm)
            else:
                metric[i,j] = 0
    return pd.DataFrame(metric, index=X.columns, columns=X.columns)


# ------------------------------------------------------------------- DENOISING

def mpPDF(var,q,pts):
    # Marcenko-Pastur pdf
    # q=T/N
    eMin,eMax = var * (1 - (1. / q)**.5)**2, var * (1 + (1. / q)**.5)**2
    eVal = np.linspace(eMin, eMax, pts)
    pdf = q / (2 * np.pi * var * eVal) * ((eMax - eVal) * (eVal - eMin))**.5
    pdf = pd.Series(pdf, index=eVal)
    return pdf

from sklearn.neighbors import KernelDensity
def fitKDE(obs, bWidth=.25, kernel='gaussian', x=None):
    # Fit kernel to a series of obs, and derive the prob of obs
    # x is the array of values on which the fit KDE will be evaluated
    if len(obs.shape) == 1:
        obs=obs.reshape(-1,1)
    
    kde = KernelDensity(kernel=kernel,bandwidth=bWidth).fit(obs)
    
    if x is None:
        x = np.unique(obs).reshape(-1,1)
    
    if len(x.shape) == 1:
        x = x.reshape(-1,1)
    logProb = kde.score_samples(x) # log(density)
    pdf = pd.Series(np.exp(logProb), index=x.flatten())
    return pdf

def getPCA(matrix):
    # Get eVal, eVec from a Hermitian matrix
    eVal, eVec = np.linalg.eigh(matrix)
    indices = eVal.argsort()[::-1] # arguments for sorting eVal desc
    eVal, eVec = eVal[indices], eVec[:,indices]
    eVal = np.diagflat(eVal)
    return eVal, eVec

def cov2corr(cov):
    # Derive the correlation matrix from a covariance matrix
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std,std)
    corr[corr<-1], corr[corr>1]=-1, 1 # numerical error
    return corr

def denoisedCorr(eVal,eVec,nFacts):
    # Remove noise from corr by fixing random eigenvalues
    eVal_=np.diag(eVal).copy()
    eVal_[nFacts:]=eVal_[nFacts:].sum() / float(eVal_.shape[0]-nFacts)
    eVal_=np.diag(eVal_)
    corr1=np.dot(eVec,eVal_).dot(eVec.T)
    corr1=cov2corr(corr1)
    return corr1

def errPDFs(var,eVal,q,bWidth,pts=1000):
    # Fit error

    # scipy minimize puts all vars in a vector, so define a function to unpack 
    if type(var) == np.ndarray:
        var = var[0]

    pdf0 = mpPDF(var, q, pts) # theoretical pdf
    pdf1 = fitKDE(eVal, bWidth, x=pdf0.index.values) # empirical pdf
    sse = np.sum((pdf1 - pdf0)**2)
    return sse

from scipy.optimize import minimize
def findMaxEval(eVal, q, bWidth):
    
    # Find max random eVal by fitting Marcenko’s dist
    out = minimize(lambda *x: errPDFs(*x), .5, args=(eVal, q, bWidth), bounds=((1E-5, 1-1E-5),))
    
    if out['success']:
        var = out['x'][0]
    else:
        var = 1
    eMax = var * (1 + (1. / q)**.5)**2
    return eMax, var

# ------------------------------------------------------------------- MULTI PROCESSING

import sys
import time
import numpy as np
import multiprocessing as mp
import datetime as dt

def nestedParts(numAtoms,numThreads,upperTriang=False):
    # partition of atoms with an inner loop
    parts,numThreads_=[0],min(numThreads,numAtoms)
    for num in range(numThreads_):
        part=1 + 4*(parts[-1]**2+parts[-1]+numAtoms*(numAtoms+1.)/numThreads_)
        part=(-1+part**.5)/2.
        parts.append(part)
    parts=np.round(parts).astype(int)
    if upperTriang: # the first rows are the heaviest
        parts=np.cumsum(np.diff(parts)[::-1])
        parts=np.append(np.array([0]),parts)
    return parts

def linParts(numAtoms,numThreads):
    # partition of atoms with a single loop
    parts=np.linspace(0,numAtoms,min(numThreads,numAtoms)+1)
    parts=np.ceil(parts).astype(int)
    return parts

def expandCall(kargs):
    # Expand the arguments of a callback function, kargs[’func’]
    func=kargs['func']
    del kargs['func']
    out=func( ** kargs)
    return out

def processJobs_(jobs):
    # Run jobs sequentially, for debugging
    out=[]
    for job in jobs:
        out_=expandCall(job)
        out.append(out_)
    return out

def reportProgress(jobNum,numJobs,time0,task):
    # Report progress as asynch jobs are completed
    msg=[float(jobNum)/numJobs,(time.time()-time0)/60.]
    msg.append(msg[1]*(1/msg[0]-1))
    timeStamp=str(dt.datetime.fromtimestamp(time.time()))
    msg=timeStamp+' '+str(round(msg[0]*100,2))+'% '+task+' done after '+ str(round(msg[1],2))+' minutes. Remaining '+str(round(msg[2],2))+' minutes.'
    if jobNum<numJobs:sys.stderr.write(msg+'\r')
    else:sys.stderr.write(msg+'\n')
    return

def processJobs(jobs,task=None,numThreads=24):
    # Run in parallel.
    # jobs must contain a ’func’ callback, for expandCall
    if task is None:task=jobs[0]['func'].__name__
    pool=mp.Pool(processes=numThreads)
    outputs,out,time0=pool.imap_unordered(expandCall,jobs),[],time.time()
    # Process asynchronous output, report progress
    for i,out_ in enumerate(outputs,1):
        out.append(out_)
        reportProgress(i,len(jobs),time0,task)
    pool.close();pool.join() # this is needed to prevent memory leaks
    return out

def barrierTouch(r,width=.5):
    # find the index of the earliest barrier touch
    t,p={},np.log((1+r).cumprod(axis=0))
    for j in range(r.shape[1]): # go through columns
        for i in range(r.shape[0]): # go through rows
            if p[i,j]>=width or p[i,j]<=-width:
                t[j]=i
                break
    return t

def mpPandasObj(func,pdObj,numThreads=24,mpBatches=1,linMols=True,**kargs):
    '''
    Parallelize jobs, return a DataFrame or Series
    + func: function to be parallelized. Returns a DataFrame
    + pdObj[0]: Name of argument used to pass the molecule
    + pdObj[1]: List odf0 = mpPandasObj(func=applyPtSlOnT1, pdObj=('molecule',events.index), numThreads=numThreads, close=close, events=events, ptSl=[ptSl,ptSl])f atoms that will be grouped into molecules
    + kargs: any other argument needed by func
    Example: df1=mpPandasObj(func,('molecule',df0.index),24,**kargs)
    '''

    if linMols: parts = linParts(len(pdObj[1]), numThreads * mpBatches)
    else: parts = nestedParts(len(pdObj[1]), numThreads * mpBatches)
    jobs=[]
    for i in range(1,len(parts)):
        job={pdObj[0]:pdObj[1][parts[i-1]:parts[i]],'func':func}
        job.update(kargs)
        jobs.append(job)
    if numThreads==1:out=processJobs_(jobs)
    else: out=processJobs(jobs,numThreads=numThreads)
    if isinstance(out[0],pd.DataFrame):df0=pd.DataFrame()
    elif isinstance(out[0],pd.Series):df0=pd.Series(dtype='float') # not orgininal code: dtype='float'
    else: return out
    # for i in out: df0=df0.append(i) # append is deprecated !
    for i in out: df0 = pd.concat([df0,i],axis=0) # added own code
    return df0.sort_index()

# ------------------------------------------------------------------- LABELING


def applyPtSlOnT1(close, events, ptSl, molecule):
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)

    if ptSl[0] > 0:pt = ptSl[0] * events_['trgt']
    else: pt = pd.Series(index=events.index, dtype=float) # NaNs

    if ptSl[1] > 0:sl = -ptSl[1] * events_['trgt']
    else: sl = pd.Series(index=events.index, dtype=float) # NaNs

    for loc, t1 in events_['t1'].fillna(close.index[-1]).items():
        df0 = close[loc:t1] # path prices
        # df0 = (df0/close[loc]-1) * events_.at[loc,'side'] # for relative returns
        df0 = df0 - close[loc] # for absolute moves
        out.loc[loc,'sl'] = df0[df0<sl[loc]].index.min() # earliest stop loss.
        out.loc[loc,'pt'] = df0[df0>pt[loc]].index.min() # earliest profit taking.
    return out


def vertBar(close, tEvents, numDays):
    t1=close.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
    t1=t1[t1<close.shape[0]]
    t1=pd.Series(close.index[t1],index=tEvents[:t1.shape[0]]) # NaNs at end
    return t1


def getEvents(close: pd.Series, tEvents: pd.Series, ptSl, trgt: pd.Series, minRet: float, numThreads: int, t1=False):
    
    #1) get target
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt>minRet] # minRet

    #2) get t1 (max holding period)
    if t1 is False: t1 = pd.Series(pd.NaT, index=tEvents)

    #3) form events object, apply stop loss on t1
    side_ = pd.Series(1.,index=trgt.index) # arbitrarily set side to 1, since it is not needed to learn the side

    events = pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis=1).dropna(subset=['trgt'])
    # df0 = mpPandasObj(func=applyPtSlOnT1, pdObj=('molecule',events.index), numThreads=numThreads, close=close, events=events, ptSl=[ptSl,ptSl]) # [ptSl, ptSl] does not work
    df0 = mpPandasObj(func=applyPtSlOnT1, pdObj=('molecule',events.index), numThreads=numThreads, close=close, events=events, ptSl=ptSl)

    # get timestamps of earliest hit of either stop loss or profit taking 
    events['t1'] = df0.dropna(how='all').min(axis=1) # pd.min ignores nan
    events = events.drop('side',axis=1)
    return events


def getBins(events, close, trgt):
    #1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    px=close.reindex(px,method='bfill')
    #2) create out object
    out=pd.DataFrame(index=events_.index)
    # out['move']=px.loc[events_['t1'].values].values / px.loc[events_.index] - 1 # for relative returns
    out['move'] = px.loc[events_['t1'].values].values - px.loc[events_.index]  # for absolute moves
    out['bin'] = np.sign(out['move'])
    out['bin'].loc[np.abs(out['move']) <= trgt] = 0 # time barrier touch
    return out

# ------------------------------------------------------------------- AVERAGE UNIQUENESS

def mpNumCoEvents(closeIdx,t1,molecule):
    '''
    Compute the number of concurrent events per bar.
    +molecule[0] is the date of the first event on which the weight will be computed
    +molecule[-1] is the date of the last event on which the weight will be computed
    Any event that starts before t1[molecule].max() impacts the count.
    '''
    #1) find events that span the period [molecule[0],molecule[-1]]
    t1 = t1.fillna(closeIdx[-1]) # unclosed events still must impact other weights
    t1 = t1[t1>=molecule[0]] # events that end at or after molecule[0]
    t1 = t1.loc[:t1[molecule].max()] # events that start at or before t1[molecule].max()
    #2) count events spanning a bar
    iloc = closeIdx.searchsorted(np.array([t1.index[0],t1.max()]))
    count = pd.Series(0, index=closeIdx[iloc[0]:iloc[1]+1], dtype='float64')
    for tIn, tOut in t1.items(): count.loc[tIn:tOut] += 1.
    return count.loc[molecule[0]:t1[molecule].max()]

def mpSampleTW(t1,numCoEvents,molecule):
    # Derive average uniqueness over the event's lifespan
    wght=pd.Series(index=molecule, dtype='float64')
    for tIn,tOut in t1.loc[wght.index].items():
        wght.loc[tIn]=(1./numCoEvents.loc[tIn:tOut]).mean()
    return wght



# ------------------------------------------------------------------- PURGING CROSS VALIDATION

from sklearn.metrics import log_loss
from sklearn.model_selection._split import KFold

class PurgedKFold(KFold):
    '''
    Extend KFold to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training examples in between
    '''
    def __init__(self,n_splits=3,t1=None,pctEmbargo=0.):
        if not isinstance(t1,pd.Series):
            raise ValueError('Label Through Dates must be a pandas series')
        super(PurgedKFold,self).__init__(n_splits,shuffle=False,random_state=None)
        self.t1=t1
        self.pctEmbargo=pctEmbargo

    def split(self,X,y=None,groups=None):
        if (X.index==self.t1.index).sum()!=len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')
        indices=np.arange(X.shape[0])
        mbrg=int(X.shape[0]*self.pctEmbargo)
        test_starts=[(i[0],i[-1]+1) for i in np.array_split(np.arange(X.shape[0]),self.n_splits)]
        for i,j in test_starts:
            t0=self.t1.index[i] # start of test set
            test_indices=indices[i:j]
            maxT1Idx=self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices=self.t1.index.searchsorted(self.t1[self.t1<=t0].index)
            train_indices=np.concatenate((train_indices,indices[maxT1Idx+mbrg:]))
            yield train_indices,test_indices

# ------------------------------------------------------------------- FEATURE IMPORTANCE


def featImpMDI(ﬁt,featNames):
    # feat importance based on IS mean impurity reduction
    # get importances for each tree
    df0 = {i: tree.feature_importances_ for i,tree in enumerate(ﬁt.estimators_)}
    # convert to DF
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = featNames
    df0 = df0.replace(0, np.nan) # because max_features=1

    # compute mean and std of importances
    imp = pd.concat({'mean':df0.mean(), 'std':df0.std()*df0.shape[0]**-.5}, axis=1) # CLT
    imp /= imp['mean'].sum()
    return imp


def featImpMDA(clf, X, y, n_splits, t1):
    # feat importance based on OOS score reduction
    # cvGen = KFold(n_splits=n_splits)
    cvGen = PurgedKFold(n_splits=n_splits,t1=t1) # purged
    scr0, scr1 = pd.Series(dtype=float), pd.DataFrame(columns=X.columns)
    for i,(train,test) in tqdm(enumerate(cvGen.split(X=X, y=y)), total=n_splits):
        X0, y0 = X.iloc[train,:], y.iloc[train] # get train set
        X1, y1 = X.iloc[test,:], y.iloc[test] # get test set
        fit = clf.fit(X=X0, y=y0) # the fit occurs here
        prob = fit.predict_proba(X1) # prediction before shuffling
        pred = fit.predict(X1) # prediction before shuffling
        # compute logloss before shuffling
        scr0.loc[i] = -log_loss(y1, prob, labels=clf.classes_)
        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values) # shuffle one column
            prob = fit.predict_proba(X1_) # prediction after shuffling
            # compute logloss after shuffling
            scr1.loc[i,j] = -log_loss(y1, prob, labels=clf.classes_)
    # compute importance for each feature
    imp = (-1 * scr1).add(scr0, axis=0)
    imp = imp / (-1 * scr1) # normalize
    imp = pd.concat({'mean' : imp.mean(), 'std' : imp.std() * imp.shape[0] ** -.5}, axis=1) # CLT
    return imp

def groupMeanStd(df0,clstrs):
    out = pd.DataFrame(columns=['mean','std'])
    for i, j in clstrs.items():
        df1 = df0[j].sum(axis=1)  # sum of each MDI in the cluster
        out.loc['C_'+str(i), 'mean'] = df1.mean()  # mean 
        out.loc['C_'+str(i), 'std'] = df1.std() * df1.shape[0] ** -.5  # std * sqrt(n)
    return out

def featImpMDI_Clustered(fit, featNames, clstrs):
    # get importances of each tree
    df0 = {i:tree.feature_importances_ for i,tree in enumerate(fit.estimators_)}
    # convert to dataframe
    df0 = pd.DataFrame.from_dict(df0,orient='index')
    df0.columns = featNames
    df0 = df0.replace(0,np.nan) # because max_features=1
    # get mean and std
    imp = groupMeanStd(df0,clstrs)
    imp /= imp['mean'].sum()
    return imp

def featImpMDA_Clustered(clf,X,y,clstrs,n_splits, t1):
    # cvGen = KFold(n_splits=n_splits)
    cvGen = PurgedKFold(n_splits=n_splits,t1=t1) # purged
    scr0, scr1 = pd.Series(dtype=np.float64), pd.DataFrame(columns=clstrs.keys())  # make empty scrorer
    for i,(train,test) in tqdm(enumerate(cvGen.split(X=X, y=y)), total=n_splits):
        # train and test by cv folds
        X0, y0 = X.iloc[train,:], y.iloc[train]
        X1, y1 = X.iloc[test,:], y.iloc[test]
        # fit classifier and compute score
        fit = clf.fit(X=X0,y=y0)
        prob = fit.predict_proba(X1)
        scr0.loc[i] = -log_loss(y1, prob, labels=clf.classes_)
        for j in scr1.columns:
            X1_ = X1.copy(deep=True) 
            # shuffle cluster
            for k in clstrs[j]:
                np.random.shuffle(X1_[k].values) 
            # fit and compute score after 1 cluster shuffled
            prob = fit.predict_proba(X1_)
            scr1.loc[i,j] = -log_loss(y1, prob, labels=clf.classes_)
    # compute importances as difference between scores
    imp = (-1 * scr1).add(scr0, axis=0)
    imp = imp / (-1*scr1)
    # mean and std
    imp = pd.concat({'mean' : imp.mean(),'std' : imp.std() * imp.shape[0] ** -.5}, axis=1)
    imp.index = ['C_'+str(i) for i in imp.index]
    return imp

