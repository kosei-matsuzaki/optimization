import sys, numpy as np, cma
from core.benchmarks import BENCHMARKS_10D_BY_NAME as B10
from core.optimizers.mceso import MultiChannelEpidemicOptimizer as M
b=B10['F12-BentCigar']; lo,hi=b.bounds if hasattr(b,'bounds') else (-5,5)
CK=[2500,5000,10000,15000,20000,25000]
def mceso(kw,seed):
    class T(M):
        def _update_cc_cov(self, st, survived):
            pf=st.gen_parent_f; nl=st.gen_n_local
            beat=sum(1 for k,x,f in survived if k<nl and pf is not None and k<len(pf) and f<pf[k])
            cl=sum(1 for k,x,f in survived if k<nl)
            self.ns.append((beat,cl))
            super()._update_cc_cov(st, survived)
        def _record_generation(self, st):
            super()._record_generation(st)
            self.log.append((len(st.history_f), np.linalg.cond(st.cc_C), st.sigma/st.span, st.best_so_far))
    o=T(b,seed=seed,**kw); o.log=[]; o.ns=[]; r=o.optimize(max_evals=25000)
    out=[]
    for c in CK:
        row=[l for l in o.log if l[0]<=c][-1]; out.append((row[1],row[3]))
    ns=np.array(o.ns) if o.ns else np.zeros((1,2))
    return out, ns[:,0].mean(), ns[:,1].mean(), len(o.log)
def cmaes(seed):
    rng=np.random.default_rng(seed); x0=rng.uniform(-5,5,10)
    es=cma.CMAEvolutionStrategy(x0,1.0,{'seed':seed+1,'verbose':-9,'maxfevals':25000,'bounds':[-5,5]})
    rec={}; best=np.inf
    while not es.stop():
        X=es.ask(); F=[b(np.array(x)) if callable(b) else b.func(np.array(x)) for x in X]; es.tell(X,F); best=min(best,min(F))
        n=es.countevals; c=float((es.D.max()/es.D.min())**2)
        for ck in CK:
            if ck not in rec and n>=ck: rec[ck]=(c,best)
        if n>=25000: break
    return [rec.get(ck,(float('nan'),best)) for ck in CK], es.sp.popsize
mode=sys.argv[1]
for seed in [int(x) for x in sys.argv[2].split(',')]:
    if mode=='cma':
        out,lam=cmaes(seed); print('CMA-ES seed',seed,'lambda',lam)
    else:
        kw=eval(mode); out,beat,cl,g=mceso(kw,seed); print(mode,'seed',seed,f'gens {g} close-children/gen {cl:.2f} beat-parent/gen {beat:.2f}')
    print('   evals: '+'  '.join(f'{ck}:cond={c:.1e},f={f:.1e}' for ck,(c,f) in zip(CK,out)))
