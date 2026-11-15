# Passaggio 4: Monte Carlo.
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, replace

import numpy as np

from .alberi import distribuzione_exit, leaver
from .input import Caso, InputMancante, anni, apri
from .bs import call
from .opm import mercato
from .waterfall import payoff, soglie


def impostazioni(caso: Caso) -> dict:
    ws = apri()["Parametri_Mercato"]
    col = {"A": 4, "B": 5, "C": 6}[caso.codice]
    trova = {"n": "Numero di simulazioni Monte Carlo", "seed": "Seed del generatore", "step": "Step temporali Monte Carlo"}
    out = {}
    for chiave, et in trova.items():
        r = next((r for r in range(1, ws.max_row + 1) if str(ws.cell(r, 3).value or "").startswith(et)), None)
        if r is None:
            raise InputMancante(f"'{et}' non trovato in Parametri_Mercato")
        out[chiave] = int(ws.cell(r, col).value)
    return out


@dataclass
class RisultatoMC:
    valore: float
    errore_standard: float
    n_percorsi: int
    valore_semplice: float = float("nan")
    errore_standard_semplice: float = float("nan")

    @property
    def intervallo_95(self) -> tuple[float, float]:
        return self.valore - 1.96 * self.errore_standard, self.valore + 1.96 * self.errore_standard


def _griglia(data_rif: dt.date, fine: dt.date, step_anno: int, date_exit: list[dt.date]) -> np.ndarray:
    nodi = sorted({d for d in date_exit if d > data_rif} | {fine})
    tempi, prec = [0.0], 0.0
    for d in nodi:
        t = anni(data_rif, d)
        n = max(1, math.ceil((t - prec) * step_anno - 1e-9))
        tempi.extend(prec + (t - prec) * np.arange(1, n + 1) / n)
        prec = t
    return np.array(tempi)


@dataclass
class Processo:
    log_x0: float
    r: float
    drift: np.ndarray
    vol: np.ndarray
    equity: callable
    strike_controllo: callable
    valore_controllo: float
    salti: np.ndarray | None = None
    controllo_lineare: bool = False


def processo_equity(caso: Caso, m, tempi, offset, date_idx, prob, variante=None) -> Processo:
    dt_ = np.diff(tempi)
    hurdle = lambda k: float(np.asarray(soglie(caso, offset + tempi[k], variante)["hurdle"]))
    valore = sum(p_ * call(m.E0, hurdle(ke), tempi[ke], m.r, m.sigma) for p_, ke in zip(prob, date_idx))
    return Processo(math.log(m.E0), m.r, (m.r - m.sigma ** 2 / 2) * dt_, np.full(len(dt_), m.sigma),
                    lambda x, k: x, hurdle, valore)


def fair_value_vettoriale(caso: Caso, E, tau: float, r: float, s: float, t_exit: float) -> np.ndarray:
    from scipy.stats import norm
    E = np.maximum(np.asarray(E, dtype=float), 1e-12)
    sq = s * math.sqrt(tau)

    def C(K):
        d1 = (np.log(E / K) + (r + s * s / 2) * tau) / sq
        return E * norm.cdf(d1) - K * math.exp(-r * tau) * norm.cdf(d1 - sq)

    def D(K):
        return math.exp(-r * tau) * norm.cdf((np.log(E / K) + (r - s * s / 2) * tau) / sq)

    sg, p = soglie(caso, t_exit), caso.parametri
    if caso.codice == "A":
        H, R = float(sg["hurdle"]), float(sg["ratchet"])
        return p["q_base"] * C(H) + (p["q_ratchet"] - p["q_base"]) * (C(R) + (R - H) * D(R))
    if caso.codice == "B":
        return p["c"] * C(float(sg["hurdle"])) - (p["c"] - p["s"]) * C(float(sg["fine_catch_up"]))
    return p["p"] * C(float(sg["hurdle"]))


def valore_mc(caso: Caso, data: str = "valutazione", exit_incerta: bool = False, con_leaver: bool = False,
              n: int | None = None, seed: int | None = None, step_anno: int | None = None,
              parametri: dict | None = None, variante=None, quota_unica=None, antitetiche: bool = True,
              controllo: bool = True, modello: str = "equity", sigma_attivo: float | None = None,
              leaver_fair_value: bool = False) -> RisultatoMC:
    imp = impostazioni(caso)
    n = n or imp["n"]
    seed = imp["seed"] if seed is None else seed
    step_anno = step_anno or imp["step"]
    if parametri is not None:
        caso = replace(caso, parametri={**caso.parametri, **parametri})
    m = mercato(caso, data)
    data_rif = caso.data_valutazione if data == "valutazione" else caso.data_ingresso
    offset = anni(caso.data_ingresso, data_rif)
    rng = np.random.default_rng(seed)

    if exit_incerta:
        dist = [(d, p) for d, p in distribuzione_exit(caso) if d > data_rif]
        date, prob = [d for d, _ in dist], np.array([p for _, p in dist])
        prob = prob / prob.sum()
    else:
        date, prob = [caso.data_exit], np.array([1.0])
    punti = list(date)
    if modello == "strutturale":
        punti += [dt.date(y, 12, 31) for y in range(data_rif.year, date[-1].year + 1)]
    tempi = _griglia(data_rif, date[-1], step_anno, punti)
    indice_exit = np.array([int(np.argmin(np.abs(tempi - anni(data_rif, d)))) for d in date])
    K = len(tempi) - 1
    if modello == "equity":
        pr = processo_equity(caso, m, tempi, offset, indice_exit, prob, variante)
    elif modello == "strutturale":
        from .strutturale import processo_strutturale
        pr = processo_strutturale(caso, data, tempi, indice_exit, prob, variante, sigma_attivo)
    else:
        raise ValueError(modello)

    meta = n // 2 if antitetiche else n
    Z = rng.standard_normal((K, meta))
    if antitetiche:
        Z = np.concatenate([Z, -Z], axis=1)
    n_eff = Z.shape[1]
    exit_k = indice_exit[rng.choice(len(date), size=meta, p=prob)]
    if antitetiche:
        exit_k = np.concatenate([exit_k, exit_k])

    lv = leaver(caso) if con_leaver else None
    if lv and lv.prob_annua > 0:
        u = rng.random(meta)
        tau = np.log(u) / np.log(1 - lv.prob_annua)
        if antitetiche:
            tau = np.concatenate([tau, tau])
        esce_k = np.searchsorted(tempi, tau, side="right") - 1
        esce_k = np.where(esce_k < exit_k, esce_k, K + 1)
    else:
        esce_k = np.full(n_eff, K + 1)

    pagamento = np.zeros(n_eff)
    ctrl = np.zeros(n_eff)
    logX = np.full(n_eff, pr.log_x0)
    livello = np.exp(logX)
    dt_ = np.diff(tempi)
    costo = caso.importo_mip
    t_exit = np.array([tempi[i] for i in indice_exit])
    for k in range(K + 1):
        if k > 0:
            if pr.salti is None:
                logX += pr.drift[k - 1] + pr.vol[k - 1] * math.sqrt(dt_[k - 1]) * Z[k - 1]
                livello = np.exp(logX)
            else:
                livello = livello * np.exp(pr.drift[k - 1] + pr.vol[k - 1] * math.sqrt(dt_[k - 1]) * Z[k - 1])
        if pr.salti is not None and pr.salti[k] != 0.0:
            livello = livello - pr.salti[k]
        t_ing = offset + tempi[k]
        sconto = math.exp(-pr.r * tempi[k])
        sel = exit_k == k
        sel_esce = esce_k == k
        if sel.any():
            X = livello[sel]
            ctrl[sel] = sconto * (X if pr.controllo_lineare else np.maximum(X - pr.strike_controllo(k), 0.0))
            mask = sel & (esce_k > K)
            E = pr.equity(livello[mask], k)
            pagamento[mask] = sconto * payoff(caso, E, t_ing, variante=variante, quota_unica=quota_unica)["mip"]
        if sel_esce.any():
            E = pr.equity(livello[sel_esce], k)
            if leaver_fair_value:
                rimasti = t_exit > tempi[k] + 1e-12
                pr_r = prob[rimasti] / prob[rimasti].sum()
                I = sum(p_ * fair_value_vettoriale(caso, E, te - tempi[k], m.r, m.sigma, offset + te)
                        for p_, te in zip(pr_r, t_exit[rimasti]))
            else:
                I = np.asarray(payoff(caso, E, t_ing, variante=variante, quota_unica=quota_unica)["mip"], dtype=float)
            maturato = min(1.0, t_ing / lv.vesting_anni)
            good = maturato * I + (1 - maturato) * np.minimum(costo, I)
            bad = np.minimum(costo, I)
            pagamento[sel_esce] = sconto * (lv.quota_good * good + (1 - lv.quota_good) * bad)

    if antitetiche:
        Y, X, m_ = (pagamento[:meta] + pagamento[meta:]) / 2, (ctrl[:meta] + ctrl[meta:]) / 2, meta
    else:
        Y, X, m_ = pagamento, ctrl, n_eff
    semplice, se_semplice = float(Y.mean()), float(Y.std(ddof=1) / math.sqrt(m_))
    if controllo and X.var() > 0:
        beta = float(np.cov(Y, X)[0, 1] / X.var(ddof=1))
        Yc = Y - beta * (X - pr.valore_controllo)
        return RisultatoMC(float(Yc.mean()), float(Yc.std(ddof=1) / math.sqrt(m_)), n_eff, semplice, se_semplice)
    return RisultatoMC(semplice, se_semplice, n_eff, semplice, se_semplice)


def componente_ratchet_mc(caso: Caso, **kw) -> RisultatoMC:
    base = valore_mc(caso, **kw)
    senza = valore_mc(caso, parametri={"q_ratchet": caso.parametri["q_base"]}, **kw)
    return RisultatoMC(base.valore - senza.valore, float("nan"), base.n_percorsi)
