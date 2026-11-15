# Passaggio 3: alberi binomiale e trinomiale.
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, replace

import numpy as np

from .input import Caso, InputMancante, anni, apri
from .opm import mercato, valore_opm
from .waterfall import payoff


COLONNA = {"A": 4, "B": 5, "C": 6}


def _termine(etichetta: str, caso: str) -> float:
    ws = apri()["Termini_MIP"]
    for r in range(1, ws.max_row + 1):
        if str(ws.cell(r, 3).value or "").startswith(etichetta):
            return float(ws.cell(r, COLONNA[caso]).value)
    raise InputMancante(f"'{etichetta}' non trovato in Termini_MIP")


@dataclass
class Leaver:
    prob_annua: float
    quota_good: float
    vesting_anni: float


def leaver(caso: Caso) -> Leaver:
    k = caso.codice
    lv = Leaver(_termine("Probabilità annua di uscita", k), _termine("Quota di uscite qualificate come good", k),
                _termine("Durata del vesting", k))
    if "prob_leaver" in caso.extra:
        lv = replace(lv, prob_annua=float(caso.extra["prob_leaver"]))
    return lv


def distribuzione_exit(caso: Caso) -> list[tuple[dt.date, float]]:
    ws = apri()["Exit_Distribuzione"]
    out = []
    for r in range(1, ws.max_row + 1):
        d = ws.cell(r, 3).value
        if isinstance(d, dt.datetime):
            p = float(ws.cell(r, COLONNA[caso.codice]).value or 0.0)
            if p > 0:
                out.append((d.date(), p))
    tot = sum(p for _, p in out)
    mesi = int(caso.extra.get("sposta_exit_mesi", 0))
    if mesi:
        from dateutil.relativedelta import relativedelta
        out = [(d + relativedelta(months=mesi, day=31), p) for d, p in out]
    return [(d, p / tot) for d, p in out]


def passi_albero() -> int:
    ws = apri()["Parametri_Mercato"]
    for r in range(1, ws.max_row + 1):
        if str(ws.cell(r, 3).value or "").startswith("Step dell'albero"):
            return int(ws.cell(r, 4).value)
    raise InputMancante("Step dell'albero non trovato in Parametri_Mercato")


def valore_albero(caso: Caso, data: str = "valutazione", tipo: str = "binomiale", N: int | None = None,
                  exit_incerta: bool = False, con_leaver: bool = False, variante=None, quota_unica=None,
                  sigma: float | None = None, parametri: dict | None = None, prob_leaver: float | None = None) -> float:
    if parametri is not None:
        caso = replace(caso, parametri={**caso.parametri, **parametri})
    m = mercato(caso, data, sigma=sigma)
    data_rif = caso.data_valutazione if data == "valutazione" else caso.data_ingresso
    offset = anni(caso.data_ingresso, data_rif)

    if exit_incerta:
        dist = [(d, p) for d, p in distribuzione_exit(caso) if d > data_rif]
        giorni = [(d - data_rif).days for d, _ in dist]
        K, dt_ = giorni[-1], 1 / 365
        sopravv, exit_al_passo = 1.0, {}
        for g, (_, p) in zip(giorni, dist):
            exit_al_passo[g] = min(1.0, p / sopravv) if sopravv > 1e-15 else 1.0
            sopravv -= p
        exit_al_passo[K] = 1.0
    else:
        K = N or passi_albero()
        dt_ = m.T / K
        exit_al_passo = {K: 1.0}

    r, s = m.r, m.sigma
    sconto = math.exp(-r * dt_)
    if tipo == "binomiale":
        u = math.exp(s * math.sqrt(dt_))
        d = 1 / u
        pu = (math.exp(r * dt_) - d) / (u - d)
        prob = (pu, 1 - pu)
        nodi = lambda k: m.E0 * u ** (2 * np.arange(k + 1) - k)
    elif tipo == "trinomiale":
        dx = s * math.sqrt(3 * dt_)
        pu = (math.exp(r * dt_) - 2 / 3 - math.exp(-dx) / 3) / (math.exp(dx) - math.exp(-dx))
        prob = (pu, 2 / 3, 1 / 3 - pu)
        nodi = lambda k: m.E0 * np.exp(dx * np.arange(-k, k + 1))
    else:
        raise ValueError(tipo)

    lv = leaver(caso) if con_leaver else None
    if lv and prob_leaver is not None:
        lv = replace(lv, prob_annua=prob_leaver)
    q_lv = 1 - (1 - lv.prob_annua) ** dt_ if lv else 0.0
    costo = caso.importo_mip

    def mip(E, k):
        return np.asarray(payoff(caso, E, offset + k * dt_, variante=variante, quota_unica=quota_unica)["mip"], dtype=float)

    V = mip(nodi(K), K)
    for k in range(K - 1, -1, -1):
        if tipo == "binomiale":
            cont = sconto * (prob[0] * V[1:] + prob[1] * V[:-1])
        else:
            cont = sconto * (prob[0] * V[2:] + prob[1] * V[1:-1] + prob[2] * V[:-2])
        if lv:
            E = nodi(k)
            I = mip(E, k)
            maturato = min(1.0, (offset + k * dt_) / lv.vesting_anni)
            good = maturato * I + (1 - maturato) * np.minimum(costo, I)
            pagato = lv.quota_good * good + (1 - lv.quota_good) * np.minimum(costo, I)
            cont = q_lv * pagato + (1 - q_lv) * cont
        q = exit_al_passo.get(k, 0.0)
        V = q * mip(nodi(k), k) + (1 - q) * cont if q > 0 else cont
    return float(V[0])


def valore_opm_exit_incerta(caso: Caso, data: str = "valutazione", variante=None, quota_unica=None) -> float:
    data_rif = caso.data_valutazione if data == "valutazione" else caso.data_ingresso
    tot = 0.0
    for d, p in [(d, p) for d, p in distribuzione_exit(caso) if d > data_rif]:
        m = mercato(caso, data)
        m.T = anni(data_rif, d)
        tot += p * valore_opm(caso, m, t=anni(caso.data_ingresso, d), variante=variante, quota_unica=quota_unica)
    return tot
