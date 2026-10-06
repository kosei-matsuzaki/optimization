import numpy as np, itertools
from multiprocessing import Pool
from core.benchmarks import BENCHMARKS_10D_BY_NAME as B
from core.optimizers.mceso import MultiChannelEpidemicOptimizer as M
FUN=['F10-EllipsoidalRot','F11-Discus','F12-BentCigar','F14-DiffPowers']
WIN=[150,300,600]
CFG={'base':{},'frz50':{'cc_spill_freeze_gens':50},'gate2':{'cc_gate_mahal':2.0},'gate3':{'cc_gate_mahal':3.0},'gate5':{'cc_gate_mahal':5.0}}
def job(a):
    fn,w,c,s=a
    r=M(B[fn],seed=s*100,restart_no_improve_threshold=w,**CFG[c]).optimize(max_evals=25000)
    return (fn,w,c,s,r.best_f)
if __name__=='__main__':
    tasks=list(itertools.product(FUN,WIN,CFG,range(10)))
    with Pool(9) as p: R=p.map(job,tasks,chunksize=4)
    import collections; d=collections.defaultdict(list)
    for fn,w,c,s,f in R: d[(fn,w,c)].append(f)
    print('SR@1e-10 (n=10) per window 150/300/600   | median best_f per window')
    for fn in FUN:
        for c in CFG:
            sr=[np.mean(np.array(d[(fn,w,c)])<=1e-10) for w in WIN]; md=[np.median(d[(fn,w,c)]) for w in WIN]
            print(f"{fn[:13]:13s} {c:6s} SR {'/'.join(f'{x:.1f}' for x in sr)}   med {'/'.join(f'{x:.1e}' for x in md)}")
    print('mean SR over 4 funcs per window:')
    for c in CFG:
        m=[np.mean([np.mean(np.array(d[(fn,w,c)])<=1e-10) for fn in FUN]) for w in WIN]
        print(f"  {c:6s} {'/'.join(f'{x:.3f}' for x in m)}  spread {max(m)-min(m):.3f}")
