import numpy
class Site:

    def __init__(self, siteid, candlist, cand):
        self.siteid = siteid
        self.candlist = candlist
        self.cand = cand
        self.accdist = dict()
        self.qssum = 0
        self.k = 0
        self.M = 0
        self.ub = 0
        self.lb = 0
        self.T = len(cand[candlist[0]])

    def prp1_first(self, query, s, e, qssum, k, M):
        self.k = k
        self.M = M
        self.qssum = qssum
        for sid in self.candlist:
            self.accdist[sid] = dict()
            for q in range(query.shape[0]):
                self.accdist[sid][q] = sum( (i-j)**2 for (i,j) in zip(query[q,:].tolist()[0], self.cand[sid][s:e]))
        self.ub, self.lb = self.cal_bound( query, e)
        return sorted(self.ub.values())[:self.k/self.M+1]

    def prp1_later(self, query, s, e):
        for q in range(query.shape[0]):
            self.qssum[q] -= sum( i**2 for i in query[q,:].tolist()[0])
            if e == self.T:
                self.qssum[q] = 0

        for sid in self.candlist:
            for q in range(query.shape[0]):
                self.accdist[sid][q] += sum( (i-j)**2 for (i,j) in zip(query[q,:].tolist()[0], self.cand[sid][s:e]))
        self.ub, self.lb = self.cal_bound( query, e)
        return sorted(self.ub.values())[:self.k/self.M+1]
    
    def prp2(self, prp_th):
        tmp = sorted(self.ub.values())[:self.k/self.M+1]
        return [b for b in self.ub.values() if b < prp_th and b > tmp[-1]]

    def cal_bound(self,query,e):
        ub = dict()
        lb = dict()
        for sid in self.candlist:
            ub_tmp = dict()
            lb_tmp = dict()
            cssum = sum( v**2 for v in self.cand[sid][e:] )
            for q in range(query.shape[0]):
                lb_tmp[q] = self.accdist[sid][q]
                ub_tmp[q] = self.accdist[sid][q] + self.qssum[q] + cssum
                ub_tmp[q] += 2*(self.qssum[q] * cssum )**0.5
            ub[sid] = sum( v**0.5 for v in ub_tmp.values() )
            lb[sid] = sum( v**0.5 for v in lb_tmp.values() )
        return ub, lb

    def prune(self, th):
        self.candlist = [sid for sid in self.candlist if self.lb[sid] <= th]
        return len(self.candlist)
    
    def get_ans(self):
        return self.candlist

