import csv, gzip, os, statistics as st

METHODS = ["MC-ESO","v2_scan2d","ELSHADE-SPACMA","LSHADE-SPACMA","EBOwithCMAR","IPOP-CMA-ES","L-SHADE"]
LEVELS = ["sr_1e-2","sr_1e-4","sr_1e-7","sr_1e-10"]

def rd(p):
    f = gzip.open(p,'rt') if p.endswith('.gz') else open(p)
    return list(csv.DictReader(f))

def pct(s):
    return float(s.rstrip('%')) if s and s!='N/A' else float('nan')

def load(run, dim=2):
    base=f"{run}/dim{dim}/summary.csv"
    if not os.path.exists(base): base+='.gz'
    rows=rd(base)
    d={}
    for r in rows:
        d[(r['function'], r['method'])] = r
    return d

def load_w(run, dim=2):
    base=f"{run}/dim{dim}/wilcoxon.csv"
    if not os.path.exists(base): base+='.gz'
    return rd(base)

RUNS = {
 'i2':'results/20261012_194245_j37d2i2_quick',
 'i3':'results/20261012_200359_j37d2i3_quick',
 'i4':'results/20261012_202547_j37d2i4_quick',
}
REF1_BASE = 'runs/20261012_015021_allmethods2_d2_quick'   # instance 1, seed-offset 0, 34 methods
REF1_SCAN = 'analysis/single/j36'                          # v2_scan2d instance 1

data = {k: load(v) for k,v in RUNS.items() if os.path.exists(v+'/dim2/summary.csv')}
FUNCS = sorted({f for d in data.values() for (f,m) in d})

# instance 1 reference
ref1 = load(REF1_BASE)
scan_rows = rd(REF1_SCAN+'/summary_dim2.csv')
for r in scan_rows:
    ref1[(r['function'], r['method'])] = r

def overall(d, method):
    vals = {}
    for lv in LEVELS:
        xs=[pct(d[(f,method)][lv]) for f in FUNCS if (f,method) in d]
        vals[lv]=st.mean(xs) if xs else float('nan')
    ev=[float(d[(f,method)]['evals_succ_mean']) for f in FUNCS
        if (f,method) in d and d[(f,method)]['evals_succ_mean'] not in ('','N/A','---')
        and pct(d[(f,method)]['sr_1e-10'])>0]
    vals['evals']=st.mean(ev) if ev else float('nan')
    vals['nf']=len(ev)
    return vals

print("## [1] OVERALL per instance and 3-instance mean (2D BBOB-24, n=20, 5000 evals)\n")
hdr=f"{'method':16s} " + " ".join(f"{k:>8s}" for k in ['SR1e-2','SR1e-4','SR1e-7','SR1e-10','evals','nf'])
per_inst={}
for key in ['i2','i3','i4']:
    if key not in data: continue
    print(f"### instance {key[1]}")
    print(hdr); print('-'*len(hdr))
    per_inst[key]={}
    for m in METHODS:
        v=overall(data[key],m); per_inst[key][m]=v
        print(f"{m:16s} "+" ".join(f"{v[l]:8.2f}" for l in LEVELS)+f" {v['evals']:8.1f} {v['nf']:8d}")
    print()

print("### 3-instance mean (i2,i3,i4)")
print(hdr); print('-'*len(hdr))
mean3={}
for m in METHODS:
    row={}
    for l in LEVELS+['evals']:
        xs=[per_inst[k][m][l] for k in per_inst]
        row[l]=st.mean(xs)
    row['nf']=st.mean([per_inst[k][m]['nf'] for k in per_inst])
    mean3[m]=row
    print(f"{m:16s} "+" ".join(f"{row[l]:8.2f}" for l in LEVELS)+f" {row['evals']:8.1f} {row['nf']:8.1f}")
print()

print("### instance 1 reference (tuning instance; MC-ESO/5 baselines from runs/20261012_015021_allmethods2_d2_quick, v2_scan2d from analysis/single/j36)")
print(hdr); print('-'*len(hdr))
ref1_overall={}
for m in METHODS:
    v=overall(ref1,m); ref1_overall[m]=v
    print(f"{m:16s} "+" ".join(f"{v[l]:8.2f}" for l in LEVELS)+f" {v['evals']:8.1f} {v['nf']:8d}")
print()

print("## [2] SR@1e-10 ranking")
for label,tbl in [('instance 1 (tuning)',{m:ref1_overall[m]['sr_1e-10'] for m in METHODS}),
                  ('instance 2',{m:per_inst['i2'][m]['sr_1e-10'] for m in METHODS} if 'i2' in per_inst else None),
                  ('instance 3',{m:per_inst['i3'][m]['sr_1e-10'] for m in METHODS} if 'i3' in per_inst else None),
                  ('instance 4',{m:per_inst['i4'][m]['sr_1e-10'] for m in METHODS} if 'i4' in per_inst else None),
                  ('mean of i2-i4',{m:mean3[m]['sr_1e-10'] for m in METHODS})]:
    if tbl is None: continue
    order=sorted(tbl.items(), key=lambda kv:-kv[1])
    print(f"  {label:20s}: " + "  ".join(f"{i+1}.{m} {v:.2f}" for i,(m,v) in enumerate(order)))
print()

print("## [3] Wilcoxon (ref=MC-ESO, p_two<0.05, direction by A12) win/loss counts per baseline")
for key in ['i2','i3','i4']:
    if key not in data: continue
    w=load_w(RUNS[key])
    print(f"### instance {key[1]}")
    for m in METHODS[1:]:
        win=loss=0; wl=[]; ll=[]
        for r in w:
            if r['method']!=m: continue
            try: p=float(r['p_value_two_sided']); a=float(r['a12'])
            except ValueError: continue
            if p!=p or p>=0.05: continue
            if a>0.5: win+=1; wl.append(f"{r['function'].split('-')[0]}({a:.2f},{r['a12_magnitude']})")
            elif a<0.5: loss+=1; ll.append(f"{r['function'].split('-')[0]}({a:.2f},{r['a12_magnitude']})")
        print(f"  {m:16s} MC-ESO better {win:2d} / worse {loss:2d}")
        if wl: print(f"      better: {' '.join(wl)}")
        if ll: print(f"      worse : {' '.join(ll)}")
    print()

print("## [4] PER-FUNCTION SR@1e-10 (%) — all 7 methods, per instance")
for m in METHODS:
    print(f"### {m}")
    print(f"  {'function':24s} {'i1(ref)':>8s} {'i2':>6s} {'i3':>6s} {'i4':>6s} {'mean i2-4':>10s} {'diff vs i1':>11s}")
    for f in FUNCS:
        r1 = pct(ref1[(f,m)]['sr_1e-10']) if (f,m) in ref1 else float('nan')
        vs=[]
        for k in ['i2','i3','i4']:
            vs.append(pct(data[k][(f,m)]['sr_1e-10']) if k in data and (f,m) in data[k] else float('nan'))
        ok=[v for v in vs if v==v]
        mn=st.mean(ok) if ok else float('nan')
        diff=mn-r1
        mark='' if (diff==diff and abs(diff)<1e-9) else ('  <-- changed' if diff==diff else '')
        print(f"  {f:24s} {r1:8.1f} "+" ".join(f"{v:6.1f}" for v in vs)+f" {mn:10.2f} {diff:+11.2f}{mark}")
    print()
