"""HMHH — Heterogeneous Meta-Hyper-Heuristic (Grobler, Engelbrecht, Kendall &
Yadavalli).

Sources:
  [G10] "Alternative hyper-heuristic strategies for multi-method global
        optimization", IEEE CEC 2010 (framework, Algorithm 1; tabu-search
        selection, Algorithm 2; GA self-adaptive mutation, GCPSO update).
  [G15] "Heuristic space diversity control for improved meta-hyper-heuristic
        performance", Information Sciences 300:49-62, 2015 (the four-method
        HMHH of Algorithm 1 / Fig. 1, Q_δm of Eq. (1), Table 1 parameters,
        heuristic-space-diversity (HSD) strategies).
  [vdB] van den Bergh & Engelbrecht, GCPSO (ρ schedule, s_c = 15, f_c = 5).
  [Y08] Yang, Tang & Yao, "Self-adaptive differential evolution with
        neighborhood search" (SaNSDE), CEC 2008.
  [AH05] Auger & Hansen, IPOP-CMA-ES, CEC 2005.

Algorithm ([G15] Algorithm 1): a common population of n_s entities; each entity
is allocated to one of four constituent algorithms (GA, GCPSO, SaNSDE,
CMA-ES), which update it in the context of the common population (donors,
tournament partners and the global best are drawn from everyone). Every k
iterations Q_δm (total fitness improvement of the entities each algorithm
owned over the period, Eq. (1)) is computed and entities are re-allocated by
the rank-based tabu search of Burke, Kendall & Soubeiga (2003). One iteration
= one new candidate per entity (n_s evaluations).

Parameters (source):
  n_s = 100, k = 5, tabu size 2 for 4 algorithms (1 for 3, 0 for ≤2)   [G15 Table 1]
  GCPSO: c1 2.0→0.7, c2 0.7→2.0, w 0.9→0.4 (linear over 95 % of I_max) [G15 Table 1]
  GA: p_c 0.6→0.4 (95 % of I_max), p_m = 0.1, BLX-α α = 0.5, N_t = 13  [G15 Table 1]
  GA self-adaptive mutation τ1 = 1/sqrt(2D), τ2 = 1/sqrt(2 sqrt D)    [G10 Eq. (1)]
  GCPSO ρ0 = 1.0, s_c = 15, f_c = 5                                    [vdB defaults]
  SaNSDE: p, fp learned every 50 gens, CR ~ N(CRm, 0.1) resampled every
          5 gens, CRm re-estimated every 25 gens, F ~ N(0.5, 0.3) w.p. fp
          else Cauchy(0, 1); strategies DE/rand/1/bin and
          DE/current-to-best/2/bin                                     [Y08]
  CMA-ES: default strategy parameters, σ0 = (B−A)/2, restarts at a
          uniform random mean (IPOP convention)                        [AH05]

Interpretations / guesses (the papers leave these open):
  G1. Entity fitness used in Q_δm is the entity's best-so-far ("parent")
      value, so Q ≥ 0. GA, SaNSDE and CMA-ES replace an entity only by a
      better candidate; GCPSO moves a non-elitist position and updates the
      entity's personal best (= the parent).
  G2. Tabu selection ([G10] Alg. 2 per entity + [G15] Q_δm): each entity i
      keeps a rank r_il per algorithm and a FIFO tabu list. After a period,
      improvement > 0 → r_il += 1; = 0 → r_il −= 1 and l becomes tabu for i.
      The next algorithm is the highest-r_il non-tabu available one, ties
      broken by the algorithms' Q_δm of the period, then at random.
  G3. CMA-ES is one shared distribution whose λ each iteration equals the
      number of entities it owns (μ = ⌊λ/2⌋, learning rates recomputed for
      λ); sample j is the candidate for its j-th entity. Samples are clipped
      to the box and the clipped point is used in the update. Restart when
      σ·sqrt(λ_max(C)) < 1e-12·(B−A), cond(C) > 1e14, or the best sampled f
      has not changed by > 1e-12 over 10 + ⌈30n/4⌉ generations.
  G4. GA: two parents by tournament (N_t = 13) over the common population;
      BLX-0.5 with prob. p_c else a copy of parent 1; each gene mutated with
      prob. p_m by the entity's own self-adaptive step ς_ij (initial ς =
      0.1·(B−A)); the mutated ς is kept only if the child is accepted.
  G5. GCPSO: velocity clamped to ±(B−A), positions clipped to the box; an
      entity entering GCPSO starts from its parent position, keeping its last
      velocity (zero at first). The global-best particle (if GCPSO owns it)
      uses the GCPSO update around ŷ.
  G6. SaNSDE counters advance once per iteration in which it owns entities;
      CR is clipped to [0, 1].
  G7. HSD control: default ``hsd="none"`` is the baseline HMHH of [G15].
      ``hsd="eihh2"`` (the best no-a-priori strategy in [G15]) starts with one
      randomly ordered algorithm and adds the others at 10.4 %, 23.8 % and
      42.7 % of the budget — the exponential time points [G15] quotes for
      EDHH, reused for EIHH (the EIHH points are not given).
  ``allocation="random"`` re-allocates each entity uniformly at random every k
  iterations ([G10] "Random" selection) instead of the tabu search.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult

GA, PSO, DE, CMA = 0, 1, 2, 3
_NAMES = ("GA", "GCPSO", "SaNSDE", "CMA-ES")


class _BudgetExhausted(Exception):
    pass


class HMHHOptimizer(BaseOptimizer):
    """Heterogeneous meta-hyper-heuristic (Grobler et al. 2010/2015): GA,
    GCPSO, SaNSDE and CMA-ES on a common population, entity-to-algorithm
    allocation every k iterations by rank-based tabu search."""

    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        n_entities: int = 100,       # [G15] Table 1
        k: int = 5,                  # [G15] Table 1
        allocation: str = "tabu",    # "tabu" | "random"
        hsd: str = "none",           # "none" | "eihh2"
        # GCPSO [G15] Table 1, [vdB]
        c1: tuple[float, float] = (2.0, 0.7),
        c2: tuple[float, float] = (0.7, 2.0),
        w: tuple[float, float] = (0.9, 0.4),
        rho0: float = 1.0,
        s_c: int = 15,
        f_c: int = 5,
        # GA [G15] Table 1
        pc: tuple[float, float] = (0.6, 0.4),
        pm: float = 0.1,
        blx_alpha: float = 0.5,
        tournament: int = 13,
        ga_sigma0_frac: float = 0.1,  # guess G4
        # SaNSDE [Y08]
        de_lp: int = 50,
        de_cr_reset: int = 5,
        de_crm_update: int = 25,
    ):
        super().__init__(benchmark, seed)
        if allocation not in ("tabu", "random"):
            raise ValueError("allocation must be 'tabu' or 'random'")
        if hsd not in ("none", "eihh2"):
            raise ValueError("hsd must be 'none' or 'eihh2'")
        self.n_entities = n_entities
        self.k = k
        self.allocation = allocation
        self.hsd = hsd
        self.c1, self.c2, self.w = c1, c2, w
        self.rho0, self.s_c, self.f_c = rho0, s_c, f_c
        self.pc, self.pm, self.blx_alpha, self.tournament = pc, pm, blx_alpha, tournament
        self.ga_sigma0_frac = ga_sigma0_frac
        self.de_lp, self.de_cr_reset, self.de_crm_update = de_lp, de_cr_reset, de_crm_update

    # ---------------------------------------------------------------- utils
    def _eval(self, x: np.ndarray) -> float:
        if len(self._hf) >= self._max_evals:
            raise _BudgetExhausted()
        x = np.asarray(x, dtype=float).copy()
        f = float(self.func(x))
        self._hx.append(x)
        self._hf.append(f)
        return f

    def _accept(self, i: int, x: np.ndarray, f: float) -> bool:
        if f < self.pf[i]:
            self.P[i] = x
            self.pf[i] = f
            return True
        return False

    @staticmethod
    def _lin(pair: tuple[float, float], frac: float) -> float:
        return pair[0] + (pair[1] - pair[0]) * min(frac, 1.0)

    # ----------------------------------------------------------------- CMA
    def _cma_init(self, mean: np.ndarray) -> None:
        n = self.dim
        self.cma = {
            "m": mean.copy(), "sigma": 0.5 * self._span,
            "C": np.eye(n), "B": np.eye(n), "D": np.ones(n),
            "pc": np.zeros(n), "ps": np.zeros(n), "gen": 0,
            "hist": deque(maxlen=10 + int(math.ceil(30 * n / 4))),
        }

    def _cma_step(self, rng, ents: list[int]) -> None:
        s, n = self.cma, self.dim
        lam = len(ents)
        z = rng.standard_normal((lam, n))
        y = (z * s["D"]) @ s["B"].T
        x = np.clip(s["m"] + s["sigma"] * y, self._lo, self._hi)
        f = np.empty(lam)
        for j, i in enumerate(ents):
            f[j] = self._eval(x[j])
            self._accept(i, x[j], f[j])
        s["gen"] += 1
        mu = max(lam // 2, 1)
        order = np.argsort(f)
        wts = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        wts /= wts.sum()
        mueff = 1.0 / (wts ** 2).sum()
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        ysel = (x[order[:mu]] - s["m"]) / s["sigma"]   # clipped points (G3)
        ymean = wts @ ysel
        s["m"] = s["m"] + s["sigma"] * ymean
        invsqrtC = (s["B"] / s["D"]) @ s["B"].T
        s["ps"] = (1 - cs) * s["ps"] + math.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ ymean)
        hsig = (np.linalg.norm(s["ps"]) / math.sqrt(1 - (1 - cs) ** (2 * s["gen"])) / chiN
                < 1.4 + 2 / (n + 1))
        s["pc"] = (1 - cc) * s["pc"] + hsig * math.sqrt(cc * (2 - cc) * mueff) * ymean
        s["C"] = ((1 - c1 - cmu) * s["C"] + c1 * (np.outer(s["pc"], s["pc"])
                  + (1 - hsig) * cc * (2 - cc) * s["C"])
                  + cmu * (ysel.T * wts) @ ysel)
        s["sigma"] *= math.exp(min(1.0, (cs / ds) * (np.linalg.norm(s["ps"]) / chiN - 1)))
        C = (s["C"] + s["C"].T) / 2
        restart = not (np.all(np.isfinite(C)) and np.isfinite(s["sigma"]))
        if not restart:
            d2, B = np.linalg.eigh(C)
            d2 = np.maximum(d2, 1e-300)
            s["C"], s["B"], s["D"] = C, B, np.sqrt(d2)
            s["hist"].append(float(f.min()))
            h = s["hist"]
            restart = (s["sigma"] * s["D"].max() < 1e-12 * self._span
                       or d2.max() > 1e14 * d2.min()
                       or (len(h) == h.maxlen and max(h) - min(h) < 1e-12))
        if restart:  # G3: IPOP-style restart at a uniform random mean
            self._cma_init(rng.uniform(self._lo, self._hi))

    # ----------------------------------------------------------- optimize
    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        n, ns = self.dim, self.n_entities
        lo, hi = self.bounds
        self._lo, self._hi = np.full(n, float(lo)), np.full(n, float(hi))
        self._span = float(hi - lo)
        self._hx, self._hf = [], []
        self._max_evals = max_evals
        history_pop: list[np.ndarray] = []
        n_alg = 4
        I_max = max(max_evals // ns, 1)
        t_ramp = 0.95 * I_max

        # HSD schedule (G7)
        if self.hsd == "eihh2":
            seq = list(rng.permutation(n_alg))
            add_at = [0.0, 0.104, 0.238, 0.427]
        else:
            seq, add_at = list(range(n_alg)), [0.0] * n_alg

        def available() -> list[int]:
            frac = len(self._hf) / max_evals
            return [seq[j] for j in range(n_alg) if frac >= add_at[j]]

        def tabu_size(na: int) -> int:
            return 2 if na >= 4 else (1 if na == 3 else 0)   # [G15] Table 1

        # entity state
        self.P = np.zeros((ns, n))
        self.pf = np.full(ns, np.inf)
        X = np.zeros((ns, n))
        V = np.zeros((ns, n))
        gsig = np.full((ns, n), self.ga_sigma0_frac * self._span)
        rank = np.zeros((ns, n_alg))
        tabu = [deque() for _ in range(ns)]
        # GCPSO state
        rho, n_succ, n_fail = self.rho0, 0, 0
        # SaNSDE state
        de = {"p": 0.5, "fp": 0.5, "CRm": 0.5, "gen": 0,
              "ns1": 0, "nf1": 0, "ns2": 0, "nf2": 0,
              "fs1": 0, "ff1": 0, "fs2": 0, "ff2": 0,
              "CR": np.clip(rng.normal(0.5, 0.1, ns), 0, 1),
              "crrec": [], "dfrec": []}
        tau1, tau2 = 1 / math.sqrt(2 * n), 1 / math.sqrt(2 * math.sqrt(n))

        try:
            init = rng.uniform(self._lo, self._hi, (ns, n))
            for i in range(ns):
                self.pf[i] = self._eval(init[i])
                self.P[i] = init[i]
            X[:] = self.P
            history_pop.append(self.P.copy())
            av = available()
            alloc = rng.choice(av, ns)
            self._cma_init(self.P[alloc == CMA].mean(axis=0) if np.any(alloc == CMA)
                           else rng.uniform(self._lo, self._hi))
            prev_alloc = np.full(ns, -1)

            t = 0
            while len(self._hf) < max_evals:
                pf_start = self.pf.copy()
                # entities newly given to GCPSO start from their parent
                newp = (alloc == PSO) & (prev_alloc != PSO)
                X[newp] = self.P[newp]
                prev_alloc = alloc.copy()
                for _ in range(self.k):
                    frac = t / t_ramp
                    gb = int(np.argmin(self.pf))
                    # ---------------- GA ----------------
                    pc = self._lin(self.pc, frac)
                    for i in np.where(alloc == GA)[0]:
                        par = []
                        for _p in range(2):
                            cand = rng.choice(ns, min(self.tournament, ns), replace=False)
                            par.append(cand[np.argmin(self.pf[cand])])
                        p1, p2 = self.P[par[0]], self.P[par[1]]
                        if rng.random() < pc:
                            lo_ = np.minimum(p1, p2)
                            I_ = np.abs(p1 - p2)
                            child = rng.uniform(lo_ - self.blx_alpha * I_, lo_ + I_ + self.blx_alpha * I_)
                        else:
                            child = p1.copy()
                        sig = gsig[i] * np.exp(tau1 * rng.standard_normal() + tau2 * rng.standard_normal(n))
                        mm = rng.random(n) < self.pm
                        child[mm] += sig[mm] * rng.standard_normal(mm.sum())
                        child = np.clip(child, self._lo, self._hi)
                        if self._accept(i, child, self._eval(child)):
                            gsig[i] = sig
                            X[i] = child
                    # ---------------- GCPSO ----------------
                    ents = np.where(alloc == PSO)[0]
                    if ents.size:
                        wv = self._lin(self.w, frac)
                        c1v, c2v = self._lin(self.c1, frac), self._lin(self.c2, frac)
                        gb = int(np.argmin(self.pf))
                        yhat = self.P[gb].copy()
                        fhat = self.pf[gb]
                        for i in ents:
                            if i == gb:
                                r = rng.random(n)
                                V[i] = -X[i] + yhat + wv * V[i] + rho * (1 - 2 * r)
                                V[i] = np.clip(V[i], -self._span, self._span)
                                X[i] = np.clip(yhat + wv * V[i] + rho * (1 - 2 * r), self._lo, self._hi)
                            else:
                                r1, r2 = rng.random(n), rng.random(n)
                                V[i] = (wv * V[i] + c1v * r1 * (self.P[i] - X[i])
                                        + c2v * r2 * (yhat - X[i]))
                                V[i] = np.clip(V[i], -self._span, self._span)
                                X[i] = np.clip(X[i] + V[i], self._lo, self._hi)
                            f = self._eval(X[i])
                            self._accept(i, X[i].copy(), f)
                            if i == gb:
                                if f < fhat:
                                    n_succ, n_fail = n_succ + 1, 0
                                else:
                                    n_succ, n_fail = 0, n_fail + 1
                                if n_succ > self.s_c:
                                    rho *= 2.0
                                elif n_fail > self.f_c:
                                    rho *= 0.5
                    # ---------------- SaNSDE ----------------
                    ents = np.where(alloc == DE)[0]
                    if ents.size:
                        de["gen"] += 1
                        if de["gen"] % self.de_cr_reset == 0:
                            de["CR"] = np.clip(rng.normal(de["CRm"], 0.1, ns), 0, 1)
                        gb = int(np.argmin(self.pf))
                        for i in ents:
                            others = np.delete(np.arange(ns), i)
                            r = rng.choice(others, 3, replace=False)
                            use_f_normal = rng.random() < de["fp"]
                            F = rng.normal(0.5, 0.3) if use_f_normal else rng.standard_cauchy()
                            strat1 = rng.random() < de["p"]
                            if strat1:   # DE/rand/1
                                v = self.P[r[0]] + F * (self.P[r[1]] - self.P[r[2]])
                            else:        # DE/current-to-best/2
                                v = (self.P[i] + F * (self.P[gb] - self.P[i])
                                     + F * (self.P[r[0]] - self.P[r[1]]))
                            jr = rng.integers(n)
                            m = rng.random(n) < de["CR"][i]
                            m[jr] = True
                            u = np.where(m, v, self.P[i])
                            u = np.clip(u, self._lo, self._hi)
                            old = self.pf[i]
                            ok = self._accept(i, u, self._eval(u))
                            key = "1" if strat1 else "2"
                            fk = "1" if use_f_normal else "2"
                            if ok:
                                X[i] = u
                                de["ns" + key] += 1
                                de["fs" + fk] += 1
                                de["crrec"].append(de["CR"][i])
                                de["dfrec"].append(old - self.pf[i])
                            else:
                                de["nf" + key] += 1
                                de["ff" + fk] += 1
                        if de["gen"] % self.de_crm_update == 0 and de["crrec"]:
                            dfa = np.asarray(de["dfrec"])
                            wcr = dfa / dfa.sum() if dfa.sum() > 0 else np.full(dfa.size, 1 / dfa.size)
                            de["CRm"] = float(wcr @ np.asarray(de["crrec"]))
                            de["crrec"], de["dfrec"] = [], []
                        if de["gen"] % self.de_lp == 0:
                            for pk, a, b, c, d in (("p", "ns1", "nf1", "ns2", "nf2"),
                                                   ("fp", "fs1", "ff1", "fs2", "ff2")):
                                s1, f1, s2, f2 = de[a], de[b], de[c], de[d]
                                den = s2 * (s1 + f1) + s1 * (s2 + f2)
                                de[pk] = s1 * (s2 + f2) / den if den > 0 else 0.5
                                de[a] = de[b] = de[c] = de[d] = 0
                    # ---------------- CMA-ES ----------------
                    ents = list(np.where(alloc == CMA)[0])
                    if ents:
                        self._cma_step(rng, ents)
                        for i in ents:
                            X[i] = self.P[i]
                    t += 1
                    history_pop.append(self.P.copy())

                # -------- re-allocation (Algorithm 1, lines 15-19) --------
                delta = pf_start - self.pf
                Q = np.zeros(n_alg)
                for m in range(n_alg):
                    Q[m] = delta[alloc == m].sum()        # Eq. (1)
                av = available()
                if self.allocation == "random":
                    alloc = rng.choice(av, ns)
                    continue
                ts = tabu_size(len(av))
                for i in range(ns):
                    l = alloc[i]
                    if delta[i] > 0:
                        rank[i, l] += 1
                    else:
                        rank[i, l] -= 1
                        if ts > 0:
                            if l in tabu[i]:
                                tabu[i].remove(l)
                            tabu[i].append(l)
                    while len(tabu[i]) > ts:
                        tabu[i].popleft()
                    cand = [m for m in av if m not in tabu[i]] or list(av)
                    key = np.array([[rank[i, m], Q[m], rng.random()] for m in cand])
                    best = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))[-1]
                    alloc[i] = cand[best]
        except _BudgetExhausted:
            pass

        history_pop.append(self.P.copy())
        return self._make_result(self._hx, self._hf, history_pop)
