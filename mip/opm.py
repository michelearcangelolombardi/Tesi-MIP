# Passaggio 2: OPM con il metodo dei breakpoint.
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy import integrate
from scipy.optimize import brentq

from .bs import call, digitale
from .input import Caso
from .waterfall import payoff, soglie


@dataclass
class Mercato:
    data: str
    E0: float
    T: float
    r: float
    sigma: float


def mercato(caso: Caso, data: str = "valutazione", r: float | None = None, sigma: float | None = None) -> Mercato:
    if data == "valutazione":
        m = Mercato(data, caso.equity_valutazione, caso.T, caso.rf_cont_valutazione, caso.vol_valutazione)
    elif data == "ingresso":
        m = Mercato(data, caso.equity_ingresso, caso.t_exit, caso.rf_cont_ingresso, caso.vol_ingresso)
    else:
        raise ValueError(data)
    return replace(m, r=m.r if r is None else r, sigma=m.sigma if sigma is None else sigma)


def _f(caso: Caso, t: float, variante=None, quota_unica=None, campo="mip"):
    def f(E):
        return float(payoff(caso, E, t, variante=variante, quota_unica=quota_unica)[campo])
    return f


def _senior(caso: Caso, t: float) -> list[float]:
    shl, pref = caso.strumento("Shareholder loan"), caso.strumento("Strumento privilegiato")
    a = shl.importo * (1 + shl.pik) ** t if shl else 0.0
    b = pref.importo * (1 + pref.pik) ** t if pref else 0.0
    return [x for x in (a, a + b) if x > 0]


def breakpoint(caso: Caso, t: float, variante=None, quota_unica=None) -> list[float]:
    f = _f(caso, t, variante, quota_unica)
    cand = sorted({float(v) for v in soglie(caso, t, variante).values()} | set(_senior(caso, t)))
    cand = [c for c in cand if c > 0]
    estremi = [0.0] + cand + [max(cand) * 4 + 1]
    extra = []
    pend = lambda x1, x2: (f(x2) - f(x1)) / (x2 - x1)
    for a, b in zip(estremi[:-1], estremi[1:]):
        L = b - a
        sa, sb = pend(a + 0.001 * L, a + 0.002 * L), pend(b - 0.002 * L, b - 0.001 * L)
        if abs(sa - sb) > 1e-9:
            lo, hi = a + 0.002 * L, b - 0.002 * L
            for _ in range(200):
                mid = (lo + hi) / 2
                if abs(pend(mid - 1e-6 * L, mid) - sa) < 1e-9:
                    lo = mid
                else:
                    hi = mid
            extra.append((lo + hi) / 2)
    return sorted(set(cand + extra))


def scomposizione_opm(caso: Caso, m: Mercato, t: float | None = None, variante=None, quota_unica=None,
                      campo="mip") -> tuple[float, list[dict]]:
    t = caso.t_exit if t is None else t
    f = _f(caso, t, variante, quota_unica, campo)
    bp = breakpoint(caso, t, variante, quota_unica)
    pend = lambda x1, x2: (f(x2) - f(x1)) / (x2 - x1)
    estremi = [0.0] + bp + [bp[-1] * 4 + 1]
    s_prec = pend(bp[0] * 0.25, bp[0] * 0.75)
    valore = s_prec * m.E0 + f(0.0) * math.exp(-m.r * m.T)
    righe = []
    for k, b in enumerate(bp):
        a, c_ = estremi[k], estremi[k + 2]
        s_dopo = pend(b + (c_ - b) * 0.25, b + (c_ - b) * 0.75)
        d_sx = (b - a) * 0.25
        sinistra = f(b - d_sx) + s_prec * d_sx
        salto = f(b) - sinistra
        salto = 0.0 if abs(salto) < 1e-9 * max(1.0, b) else salto
        c, d = call(m.E0, b, m.T, m.r, m.sigma), digitale(m.E0, b, m.T, m.r, m.sigma)
        contributo = (s_dopo - s_prec) * c + salto * d
        valore += contributo
        righe.append(dict(breakpoint=b, pendenza_prima=s_prec, pendenza_dopo=s_dopo, salto=salto,
                          call=c, digitale=d, contributo=contributo))
        s_prec = s_dopo
    return valore, righe


def valore_opm(caso: Caso, m: Mercato, t: float | None = None, variante=None, quota_unica=None) -> float:
    if variante == "V5":
        return valore_integrale(caso, m, t, variante)
    return scomposizione_opm(caso, m, t, variante, quota_unica)[0]


def componente_ratchet(caso: Caso, m: Mercato, t: float | None = None) -> float:
    if caso.codice != "A":
        return float("nan")
    senza = replace(caso, parametri={**caso.parametri, "q_ratchet": caso.parametri["q_base"]})
    return valore_opm(caso, m, t) - valore_opm(senza, m, t)


def valore_integrale(caso: Caso, m: Mercato, t: float | None = None, variante=None, quota_unica=None) -> float:
    t = caso.t_exit if t is None else t
    f = _f(caso, t, variante, quota_unica)
    mu, sq = (m.r - m.sigma ** 2 / 2) * m.T, m.sigma * math.sqrt(m.T)
    E = lambda z: m.E0 * math.exp(mu + sq * z)
    punti = sorted((math.log(b / m.E0) - mu) / sq for b in breakpoint(caso, t, variante, quota_unica)
                   + [x * 1.05 for x in soglie(caso, t, variante).values()]
                   + [x * 0.95 for x in soglie(caso, t, variante).values()])
    punti = [z for z in punti if -12 < z < 12]
    estremi = [-12.0] + punti + [12.0]
    tot = 0.0
    for a, b in zip(estremi[:-1], estremi[1:]):
        v, _ = integrate.quad(lambda z: f(E(z)) * math.exp(-z * z / 2), a, b, epsabs=1e-12, epsrel=1e-12, limit=200)
        tot += v
    return math.exp(-m.r * m.T) * tot / math.sqrt(2 * math.pi)


def valore_forma_chiusa(caso: Caso, m: Mercato, t: float | None = None) -> float:
    t = caso.t_exit if t is None else t
    sg, p = soglie(caso, t), caso.parametri
    C = lambda K: call(m.E0, float(K), m.T, m.r, m.sigma)
    D = lambda K: digitale(m.E0, float(K), m.T, m.r, m.sigma)
    if caso.codice == "A":
        H, R = sg["hurdle"], sg["ratchet"]
        return p["q_base"] * C(H) + (p["q_ratchet"] - p["q_base"]) * (C(R) + (R - H) * D(R))
    if caso.codice == "B":
        return p["c"] * C(sg["hurdle"]) - (p["c"] - p["s"]) * C(sg["fine_catch_up"])
    return p["p"] * C(sg["hurdle"])
