import sys
from root import root
def MsWave( k, query, sites):
    
    cost = 0
    qcost = 0
    rt = root(k, query)
    for site in sites.values():
        levelq, s, e, qssum, k, M = rt.send_first(site.siteid)
        ub = site.prp1_first(levelq, s, e, qssum, k, M)
        rt.prp1(ub)
        cost += levelq.size + 1 + len(qssum)+ len(ub)
        qcost += levelq.size
    
    for site in sites.values():
        prp_th = rt.prp2()
        ub = site.prp2(prp_th)
        rt.prp1(ub)
        cost += 1 + len(ub)

    for site in sites.values():
        th = rt.prp2()
        rc = site.prune(th)
        rt.check1(site.siteid,rc)
        cost += 2
    rt.check2()
    
    level_rs = []
    while( not rt.isdone() ):
        rs = rt.remainsite()
        level_rs += [len(rs)]
        for site in sites.values():
            if site.siteid not in rs:
                continue
            levelq, s, e  = rt.send_later(site.siteid)
            ub = site.prp1_later(levelq, s, e)
            rt.prp1(ub)
            cost += levelq.size + len(ub)
            qcost += levelq.size
        
        for site in sites.values():
            if site.siteid not in rs:
                continue
            prp_th = rt.prp2()
            ub = site.prp2(prp_th)
            rt.prp1(ub)
            cost += 1 + len(ub)

        for site in sites.values():
            if site.siteid not in rs:
                continue
            th = rt.prp2()
            rc = site.prune(th)
            rt.check1(site.siteid,rc)
            cost += 2
        rt.check2()
        
    rs = rt.remainsite()
    level_rs += [len(rs)]
    
    ans = []
    for site in sites.values():
        if site.siteid not in rs:
            continue
        ans += site.get_ans()
    
    cost += len(ans)

    return ans, cost, level_rs, qcost
