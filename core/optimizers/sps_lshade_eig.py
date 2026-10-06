"""SPS-L-SHADE-EIG — numpy port (Guo, Tsai, Yang & Hsu; CEC 2015 winner).

    S.-M. Guo, J. S.-H. Tsai, C.-C. Yang and P.-H. Hsu, "A self-optimization
    approach for L-SHADE incorporated with eigenvector-based crossover and
    successful-parent-selecting framework on CEC 2015 benchmark set",
    IEEE CEC 2015, pp. 1003-1010.

Sources (short tags used below):

* [CODE] ``SPS_L_SHADE_EIG.m`` by C.-C. Yang (co-author), repository
  github.com/ChinChangYang/RobustOptimizer, ``solver/soea/SPS_L_SHADE_EIG/``.
  The competition copy redistributed in github.com/TBU-AILab/
  ResourceFiles_TEVC2021 ``algorithms/SPS-L-SHADE-EIG/`` is the same algorithm
  (only bookkeeping / bound-handling switches differ). Every rule below was
  transcribed from [CODE].
* [DEF]  ``default_de_D{10,30,50,100}.mat`` + ``utils/setDEoptions.m`` of the
  competition copy: the *default* (un-tuned) parameter vector, used for any
  CEC 2015 function without a tuned file.

The CEC 2015 paper itself was not available to the implementer.

What the "self-optimization" is: CEC 2015 was the *learning-based* track, and
the authors tuned the 15 parameters of the algorithm *per test function*
offline, using SPS-L-SHADE-EIG itself as the meta-optimizer
(``gen_sps_l_shade_eig_optim_params.m`` lists the tuned vectors, e.g. D=10 f1:
NP=459, H=422, CR range [0.60, 1.17]...). That per-problem offline tuning is a
property of the competition protocol, not of the optimizer, and is NOT
reproduced; the defaults here are [DEF], i.e. what the authors' pipeline uses
on a function it has not tuned for.

Algorithm (one generation, population sorted by f at its end):
  * per individual a memory slot r ~ U{1..H}; ER ~ N(M_ER,r, erw) clipped to
    [0,1]; F = M_F,r + Cauchy(0, fw) redrawn while F<=0, capped at 1; CR ~
    N(M_CR,r, crw), CR<=CRmin -> CRmin, CR>CRmax -> CRmax.
  * current-to-pbest/1 with archive, pbest from the best max(2, round(p NP)).
  * SPS: individual i whose consecutive-failure counter FC_i > Q is
    reproduced from the successful-parent pool SP (the last NP successful
    trial vectors, a ring buffer) instead of the population: base, pbest
    (by f_SP rank), r1 and r2 (from SP ∪ A) all come from SP; the crossover
    partner and the bound-repair anchor are SP_i too.
  * EIG: with probability ER_i the binomial crossover is done in the
    eigenbasis B of C, where C <- (1-cw) C + cw cov(P) every generation
    (C_0 = cov(P_0)); cw decays linearly cw = (1 - FES/maxFES) cw_init.
    The covariance is of the *whole* current population.
  * bound repair: midpoint between the violated bound and the base parent.
  * strict selection (u < x). Successful: (ER, F, CR, |Δf|) stored; FC_i = 0;
    the trial is appended to SP. Failed: FC_i += 1.
  * memory: M_ER, M_CR = Δf-weighted arithmetic mean; M_F = weighted Lehmer
    mean; H-slot ring.
  * LPSR: NP = round(NP_init - (NP_init - NP_min) FES/maxFES); the worst are
    dropped; archive capacity max(2, round(Ar NP)), trimmed to its first
    entries; SP trimmed to NP entries.

Parameters (defaults from [DEF] unless noted):
  n_init_factor 19      NP_init = 19 D ([DEF]: x1 = 19D-4, NP = x1 + NPmin)
  n_min 4               NP_min
  memory_size H 6
  p_best 0.11
  arc_rate Ar 2.6       [CODE] default. setDEoptions rounds it -> 3 in the
                        competition pipeline; kept at 2.6 (= L-SHADE).
  q_stagnation Q 64     FC threshold for switching to SP
  f_init 0.5, cr_init 0.5, er_init 1.0   memory initial values
  cw 0.3                covariance learning rate (initial)
  fw 0.1, crw 0.1, erw 0.2  scale of the F / CR / ER samplers
  cr_min 0.05, cr_max 0.30  ([DEF]: CRmin = 0.05, CRmax = CRmin + 0.25)

Quirks of [CODE] reproduced literally (switchable):
  * ``fixed_cauchy_table=True``: the Cauchy noise for F is drawn ONCE
    (NP_init + 10 samples) and cycled through for the whole run.
  * ``archive_parent=False``: after a success [CODE] stores X(:,i) *after* it
    was overwritten by the trial, i.e. the archive receives successful
    offspring, not the replaced parents (unlike L-SHADE). True restores the
    L-SHADE behaviour.
  * r2 is only required to differ from r1 (it may equal i).
  * the SP trim after LPSR keeps the columns at positions j with
    argsort(f_SP)[j] < NP (not the NP best), and the surviving sort indices
    are reused as the new rank order — transcribed as is.
  * the archive starts as Ar NP random points of which none is used until
    real entries arrive (nA = 0), so it is represented as empty here.

Deviations: the budget is checked per evaluation (the last generation is cut
at ``max_evals``; [CODE] stops when FES > maxFES - NP). [CODE]'s optional
'auto' early stop (std of fitness below 10 eps) is not used — the run always
spends the budget. Non-finite covariance -> previous C is kept (guess).
"""
from __future__ import annotations

import math

import numpy as np

from ..benchmarks import BenchmarkFunction
from .base import BaseOptimizer, OptimizeResult


def _mround(x: float) -> int:
    """MATLAB ``round`` (half away from zero) for non-negative arguments."""
    return int(math.floor(x + 0.5))


class SPSLSHADEEIGOptimizer(BaseOptimizer):
    def __init__(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 42,
        n_init_factor: float = 19.0,
        n_min: int = 4,
        memory_size: int = 6,
        p_best: float = 0.11,
        arc_rate: float = 2.6,
        q_stagnation: int = 64,
        f_init: float = 0.5,
        cr_init: float = 0.5,
        er_init: float = 1.0,
        cw: float = 0.3,
        fw: float = 0.1,
        crw: float = 0.1,
        erw: float = 0.2,
        cr_min: float = 0.05,
        cr_max: float = 0.30,
        fixed_cauchy_table: bool = True,
        archive_parent: bool = False,
    ):
        super().__init__(benchmark, seed)
        self.n_init = max(int(n_min), _mround(n_init_factor * self.dim))
        self.n_min = int(n_min)
        self.memory_size = int(memory_size)
        self.p_best = p_best
        self.arc_rate = arc_rate
        self.q = q_stagnation
        self.f_init, self.cr_init, self.er_init = f_init, cr_init, er_init
        self.cw_init = cw
        self.fw, self.crw, self.erw = fw, crw, erw
        self.cr_min = cr_min
        self.cr_max = max(cr_max, cr_min)  # [CODE] "Fix conflict parameters"
        self.fixed_cauchy_table = fixed_cauchy_table
        self.archive_parent = archive_parent

    def optimize(self, max_evals: int = 5000) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds
        lo = np.broadcast_to(np.asarray(lo, dtype=float), (self.dim,))
        hi = np.broadcast_to(np.asarray(hi, dtype=float), (self.dim,))
        d = self.dim
        hx: list[np.ndarray] = []
        hf: list[float] = []
        hpop: list[np.ndarray] = []

        def evaluate(x: np.ndarray) -> float:
            f = float(self.func(x))
            hx.append(x.copy())
            hf.append(f)
            return f

        NPinit = min(self.n_init, max_evals)
        NPmin = min(self.n_min, NPinit)
        NP = NPinit
        X = lo + (hi - lo) * rng.random((NP, d))
        fx = np.array([evaluate(x) for x in X])
        o = np.argsort(fx, kind="stable")
        X, fx = X[o], fx[o]
        hpop.append(X.copy())

        H = self.memory_size
        MF = np.full(H, self.f_init)
        MCR = np.full(H, self.cr_init)
        MER = np.full(H, self.er_init)
        iM = 0
        FC = np.zeros(NP, dtype=int)
        C = np.cov(X, rowvar=False) if NP > 1 else np.eye(d)
        C = np.atleast_2d(C)
        SP, fSP = X.copy(), fx.copy()
        iSP = 0
        Asize = max(2, _mround(self.arc_rate * NP))
        A = np.empty((0, d))
        cw = self.cw_init
        Q = self.q

        n_chy = NPinit + 10
        chy = self.fw * np.tan(np.pi * (rng.random(n_chy) - 0.5))
        iChy = 0
        sortidx_fSP = np.argsort(fSP, kind="stable")

        def cauchy_noise() -> float:
            nonlocal iChy
            if not self.fixed_cauchy_table:
                return self.fw * math.tan(math.pi * (rng.random() - 0.5))
            v = chy[iChy]
            iChy = (iChy + 1) % n_chy
            return float(v)

        while len(hf) < max_evals and NP >= 2:
            r = rng.integers(0, H, NP)
            ER = np.clip(MER[r] + self.erw * rng.standard_normal(NP), 0.0, 1.0)
            F = np.zeros(NP)
            for i in range(NP):
                while F[i] <= 0.0:
                    F[i] = MF[r[i]] + cauchy_noise()
                F[i] = min(F[i], 1.0)
            CR = MCR[r] + self.crw * rng.standard_normal(NP)
            CR[CR <= self.cr_min] = self.cr_min
            CR[CR > self.cr_max] = self.cr_max

            n_pb = max(2, _mround(self.p_best * NP))
            pbest = rng.integers(0, n_pb, NP)
            nA = len(A)
            XA = np.vstack([X, A]) if nA else X
            SPA = np.vstack([SP, A]) if nA else SP
            r1 = np.empty(NP, dtype=int)
            r2 = np.empty(NP, dtype=int)
            for i in range(NP):
                r1[i] = rng.integers(0, NP)
                while r1[i] == i:
                    r1[i] = rng.integers(0, NP)
                r2[i] = rng.integers(0, NP + nA)
                while r2[i] == r1[i]:
                    r2[i] = rng.integers(0, NP + nA)

            sps = FC > Q
            base = np.where(sps[:, None], SP, X)
            V = np.empty((NP, d))
            for i in range(NP):
                if not sps[i]:
                    V[i] = X[i] + F[i] * (X[pbest[i]] - X[i]) + F[i] * (X[r1[i]] - XA[r2[i]])
                else:
                    V[i] = (SP[i] + F[i] * (SP[sortidx_fSP[pbest[i]]] - SP[i])
                            + F[i] * (SP[r1[i]] - SPA[r2[i]]))

            Cn = (1.0 - cw) * C + cw * np.atleast_2d(np.cov(X, rowvar=False))
            if np.all(np.isfinite(Cn)):
                C = Cn
            try:
                _, B = np.linalg.eigh((C + C.T) / 2.0)
            except np.linalg.LinAlgError:
                B = np.eye(d)

            U = np.empty((NP, d))
            for i in range(NP):
                jrand = rng.integers(0, d)
                if rng.random() < ER[i]:
                    xt = B.T @ base[i]
                    vt = B.T @ V[i]
                    m = rng.random(d) < CR[i]
                    m[jrand] = True
                    U[i] = B @ np.where(m, vt, xt)
                else:
                    m = rng.random(d) < CR[i]
                    m[jrand] = True
                    U[i] = np.where(m, V[i], base[i])
            U = np.where(U < lo, 0.5 * (lo + base), U)
            U = np.where(U > hi, 0.5 * (hi + base), U)

            n_eval = min(NP, max_evals - len(hf))
            S_er, S_f, S_cr, S_df = [], [], [], []
            for i in range(n_eval):
                fu = evaluate(U[i])
                if fu < fx[i]:
                    S_er.append(ER[i]); S_f.append(F[i]); S_cr.append(CR[i])
                    S_df.append(abs(fu - fx[i]))
                    arc_entry = X[i].copy() if self.archive_parent else U[i].copy()
                    X[i] = U[i]
                    fx[i] = fu
                    if len(A) < Asize:
                        A = np.vstack([A, arc_entry[None]])
                    else:
                        A[rng.integers(0, Asize)] = arc_entry
                    FC[i] = 0
                    SP[iSP] = U[i]
                    fSP[iSP] = fu
                    iSP = (iSP + 1) % NP
                else:
                    FC[i] += 1
            if n_eval < NP:
                hpop.append(X.copy())
                break

            if S_f:
                w = np.array(S_df)
                w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
                sf = np.array(S_f)
                MER[iM] = float(np.sum(w * np.array(S_er)))
                MCR[iM] = float(np.sum(w * np.array(S_cr)))
                MF[iM] = float(np.sum(w * sf ** 2) / np.sum(w * sf))
                iM = (iM + 1) % H

            nfe = len(hf)
            cw = (1.0 - nfe / max_evals) * self.cw_init

            o = np.argsort(fx, kind="stable")
            X, fx, FC = X[o], fx[o], FC[o]

            NP_new = _mround(NPinit - (NPinit - NPmin) * nfe / max_evals)
            NP_new = max(NPmin, min(NP, NP_new))
            X, fx, FC = X[:NP_new], fx[:NP_new], FC[:NP_new]
            Asize = max(2, _mround(self.arc_rate * NP_new))
            if len(A) > Asize:
                A = A[:Asize]
            order = np.argsort(fSP, kind="stable")
            keep = order < NP_new
            SP, fSP = SP[keep], fSP[keep]
            sortidx_fSP = order[keep]
            NP = NP_new
            iSP = iSP % NP
            hpop.append(X.copy())

        return self._make_result(hx, hf, hpop)
