import sys
import scipy.io as spio
from random import randrange
from MsWave import MsWave
from Site import Site
from numpy import *

import time as time
import datetime as dtime

lenSeg = 500
nSeg = 1000
N = lenSeg*nSeg

#data = (spio.loadmat('../LabelMe'))['data']
data = (spio.loadmat('../ANNsift_base'))['data'].T

t = time.time()
dstr = dtime.datetime.fromtimestamp(t).strftime('%Y-%m-%d,%H:%M')

WeightName = 'origin';
LogFile = open('./mlogs/Weights/' + str(lenSeg) + '_' + str(nSeg) + '/' + WeightName + '/' + dstr,'w+');

TimeLen = len(data[0])
MatrixCost = TimeLen*(TimeLen-1)/2;
AccCost = nSeg * MatrixCost;
AccNaiveCost = 0;

Q = 200;
k = 1;
mtimes = 30;
WPath = '../trans_ANN/Weights/' + str(lenSeg) + '_' + str(nSeg) + '/' + WeightName + '/';
WList = dict()
for i in range(nSeg):
    WList[i] = matrix( (spio.loadmat(WPath + 'X_'+str(lenSeg)+'_'+str(i+1)))['X'] ).T

for t in range(mtimes):
    print t
    print >>LogFile, 't=' + str(t)

    q = [randrange(N) for i in range(Q)]
    print '(#instance, #machine, #total)=(%d, %d, %d), k=%d, #Q = %d' % (lenSeg,nSeg,N,k,Q)
    print >> LogFile, '(#instance, #machine, #total)=(%d, %d, %d), k=%d, #Q = %d' % (lenSeg,nSeg,N,k,Q)
    print >> LogFile, 'query id = ' + str(q)
    _query = matrix(data[q])

    sites = dict()
    query = dict()
    for i in range(nSeg):
        W = WList[i];
        query[i] = _query*W

        s = i*lenSeg;
        e = s+lenSeg;
        cand = dict()
        for j in range(s,e):
            if j in q:
                continue
            cand[j] = ((data[j]*W).tolist())[0]

        sites[i] = Site(i, cand.keys(), cand)


    ans, cost, level_rs, qcost =  MsWave(k, query, sites)
    naive = size(_query)*nSeg + nSeg*k + k
    AccCost += cost;
    AccNaiveCost += naive
    print level_rs
    print 'ans = ' + str(ans)
    print 'cost = '+str(cost)+'/'+str(naive)+'('+str(float(cost)/naive)+')'
    print 'qcost = '+str(qcost)+'/'+str(cost)+'('+str(float(qcost)/cost)+')'
    print 'matrix+cost = '+str(cost+nSeg*MatrixCost)+'/'+str(naive)+'('+str(float(cost+nSeg*MatrixCost)/naive)+')'
    print 'AccCost = '+str(AccCost)+'/'+str(AccNaiveCost)+'('+str(float(AccCost)/AccNaiveCost)+')'
    print >> LogFile, level_rs
    print >> LogFile, 'ans = ' + str(ans)
    print >> LogFile, 'cost = '+str(cost)+'/'+str(naive)+'('+str(float(cost)/naive)+')'
    print >> LogFile, 'qcost = '+str(qcost)+'/'+str(cost)+'('+str(float(qcost)/cost)+')'
    print >> LogFile, 'matrix+cost = '+str(cost+nSeg*MatrixCost)+'/'+str(naive)+'('+str(float(cost+nSeg*MatrixCost)/naive)+')'
    print >> LogFile, 'AccCost = '+str(AccCost)+'/'+str(AccNaiveCost)+'('+str(float(AccCost)/AccNaiveCost)+')'
    t = time.time()
    dstr = dtime.datetime.fromtimestamp(t).strftime('%Y-%m-%d,%H:%M:%S')
    print >> LogFile, dstr
    print >> LogFile, '#### End of (#instance, #machine, #total)=(%d, %d, %d), k=%d, #Q = %d ####\n' % (nSeg,lenSeg,N,k,Q)
#print >> 'cost = '+str(cost)+'/'+str(naive)+'('+str(float(cost)/naive)+')'
#print 'qcost = '+str(qcost)
LogFile.close()
