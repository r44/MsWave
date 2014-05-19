import sys
import scipy.io as spio
from random import randrange
from MsWave import MsWave
from Site import Site
from numpy import *

#main
k = 1
lenSeg = 1000
nSeg = 20

lenSeg = 500
nSeg = 200
N = lenSeg*nSeg

#data = (spio.loadmat('../LabelMe'))['data']
data = (spio.loadmat('../ANNsift_base'))['data'].T

LogFile = open('./Log','w+')

Q = 10
q = [randrange(N) for i in range(Q)]
#q = [4,9,10,442,123]
#q = [4]
print '(#instance, #machine, #total)=(%d, %d, %d), k=%d, #Q = %d' % (nSeg,lenSeg,N,k,Q)
print >> LogFile, '|Q|=' + str(Q) 
print >> LogFile, '(#instance, #machine, #total)=(%d, %d, %d), k=%d, #Q = %d' % (nSeg,lenSeg,N,k,Q)
#print 'query id = ' + str(q)
print >> LogFile, 'query id = ' + str(q)
_query = matrix(data[q])

sites = dict()
query = dict()
for i in range(nSeg):
    #W = matrix( (spio.loadmat('../trans/X_'+str(lenSeg)+'_'+str(i+1)))['X'] ).T
    W = matrix( (spio.loadmat('../trans_ANN/X_'+str(lenSeg)+'_'+str(i+1)))['X'] ).T
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
TimeLen = query[0].shape[1]
MatrixCost = TimeLen*(TimeLen-1)/2;
print >> LogFile, level_rs
print >> LogFile, 'ans = ' + str(ans)
print >> LogFile, 'matrix+cost = '+str(cost+nSeg*MatrixCost)+'/'+str(naive)+'('+str(float(cost+nSeg*MatrixCost)/naive)+')'
print level_rs
print 'ans = ' + str(ans)
print 'matrix+cost = '+str(cost+nSeg*MatrixCost)+'/'+str(naive)+'('+str(float(cost+nSeg*MatrixCost)/naive)+')'
print >> LogFile, '#### End of (#instance, #machine, #total)=(%d, %d, %d), k=%d, #Q = %d ####' % (nSeg,lenSeg,N,k,Q)
#print >> 'cost = '+str(cost)+'/'+str(naive)+'('+str(float(cost)/naive)+')'
#print 'qcost = '+str(qcost)

'''
d = []
for i in xrange(N):
    tmp = 0;
    for l in xrange(Q):
        tmp += sum( (k-j)**2 for (k,j) in zip(_query[l,:].tolist()[0], data[i]) )**0.5
    d.append(tmp)

a = sorted( xrange(N), key=lambda k: d[k] )
sys.stderr.write('\n')
a = [i for i in a if i not in q]
print a[:k]
'''
