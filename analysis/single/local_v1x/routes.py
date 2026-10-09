import collections
from core.benchmarks import BENCHMARKS_5D_BY_NAME as B5, BENCHMARKS_10D_BY_NAME as B10, BENCHMARKS_BY_NAME as B2
from core.optimizers.mceso import MultiChannelEpidemicOptimizer as M
class T(M):
    def _run_generation(self, st):
        super()._run_generation(st); self.route=st.channel_route
for B,lab,me in [(B2,'2D',5000),(B5,'5D',12500),(B10,'10D',25000)]:
    row=[]
    for f in ['F01-Sphere','F02-EllipsoidalSep','F03-RastriginSep','F04-BucheRastrigin','F15-RastriginRot','F17-SchafferF7','F20-Schwefel','F21-Gallagher101','F10-EllipsoidalRot','F12-BentCigar']:
        c=collections.Counter()
        for s in range(5):
            o=T(B[f],seed=s*100); o.route=None; o.optimize(me); c[o.route]+=1
        row.append(f"{f[:7]} {dict(c)}")
    print(lab,' | '.join(row))
