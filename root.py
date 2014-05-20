from math import *
import heapq
import numpy

class root:

    def __init__(self, k, query):
        self.done = False

        self.ub = []
        self.rc = 0
        self.rs = dict()

        self.k = k
        self.M = len(query)
        self.query = query

        T = query[0].shape[1]

        #self.pivot = [T/4+T/8*i for i in range(20) if T/4+T/8*i <= T]
        self.pivot = [T/2+T/4*i for i in range(20) if T/2+T/4*i <= T]
        self.pivot = [T*3/4+T/8*i for i in range(20) if T*3/4+T/8*i <= T]
        self.pivot = [T*3/4+T/16*i for i in range(20) if T*3/4+T/16*i <= T]
        #self.pivot = [2**i for i in range(20) if 2**i <= T]

        self.pivot[-1] = T

        self.level = 0
        self.maxlevel = len(self.pivot)-1

    def send_first(self, siteid):
        self.rs[siteid] = 0
        s = 0
        e = self.pivot[0]

        qssum = [numpy.linalg.norm(self.query[siteid][i,e:], ord='fro')**2 for i in range(self.query[siteid].shape[0])]
        return self.query[siteid][:,0:e], 0, e, qssum, self.k, self.M

    def send_later(self, siteid):
        lev = self.level
        s = self.pivot[lev-1]
        e = self.pivot[lev]

        return self.query[siteid][:,s:e], s, e

    def prp1(self, ub):
        self.ub += ub

    def prp2(self):
        if len(self.ub) > self.k:
            th = sorted(self.ub)[self.k-1]
        else:
            th = sorted(self.ub)[-1]
        return th

    def check1(self,siteid,rc):
        if rc == 0:
            del self.rs[siteid]
        self.rc += rc

    def check2(self):
        if self.rc <= self.k or self.level == self.maxlevel:
            self.done = True
        self.rc = 0
        self.level += 1
        self.ub = []

    def remainsite(self):
        return self.rs

    def isdone(self):
        return self.done

    def get_answer(self):
        return []


