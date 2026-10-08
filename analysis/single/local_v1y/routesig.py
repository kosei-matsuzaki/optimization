import numpy as np, itertools, json, sys
from multiprocessing import Pool
from core.benchmarks import BENCHMARKS_5D_BY_NAME as B5, BENCHMARKS_10D_BY_NAME as B10
from core.optimizers.mceso import MultiChannelEpidemicOptimizer as M
def csig(C):
    w,V=np.linalg.eigh(C); w=np.maximum(w,1e-300)
    return float(np.log10(w[-1]/w[0])), float(np.mean(np.max(np.abs(V),axis=0)))
class T(M):
    def _channel_ratios(self, st):
        before=st.channel_route
        r=super()._channel_ratios(st)
        frac=len(st.history_f)/st.max_evals
        if before is None and st.channel_route is not None and 'commit' not in self.rec:
            lc,la=csig(st.cc_C) if st.cc_C is not None else (np.nan,np.nan)
            self.rec['commit']=dict(route=st.channel_route,gen=len(st.history_sigma_global),frac=frac,cond=st.cc_logratio_ema,algA=st.cc_align_ema,mgap=st.cc_mgap_ema,lc_cond=lc,lc_algA=la)
        for tf in (0.15,0.30):
            k=f'at{tf}'
            if frac>=tf and k not in self.rec:
                lc,la=csig(st.cc_C) if st.cc_C is not None else (np.nan,np.nan)
                self.rec[k]=dict(cond=st.cc_logratio_ema,algA=st.cc_align_ema,mgap=st.cc_mgap_ema,lc_cond=lc,lc_algA=la)
        return r
def job(a):
    d,f,s=a; B=B5 if d==5 else B10
    o=T(B[f],seed=s*100); o.rec={}; o.optimize(2500*d); return d,f,s,o.rec
if __name__=='__main__':
    fs=sorted(B5)
    tasks=list(itertools.product([5,10],fs,range(5)))
    with Pool(10) as p: R=p.map(job,tasks,chunksize=2)
    json.dump(R,open(sys.argv[1],'w'))
    print('done',len(R))
