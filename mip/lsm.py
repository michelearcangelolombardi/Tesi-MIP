# Passaggio 6: exit endogena (regola obiettivi e Longstaff-Schwartz).
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np

from .input import Caso, InputMancante, anni, apri
from .montecarlo import impostazioni
from .opm import mercato
from .strutturale import piano_bp
from .waterfall import payoff, soglie

COLONNA = {"A": 5, "B": 6, "C": 7}


@dataclass
class ParametriLSM:
    applicare: bool
    prima: int
    ultima: int
    holding_min: float
    irr_obiettivo: float
    mom_obiettivo: float
    costi: float
    n: int
    grado: int = 3


def parametri_lsm(caso: Caso, finestre_foglio: bool = False) -> ParametriLSM:
    ws = apri()["LSM"]
    col = COLONNA[caso.codice]

    def v(et):
        for r in range(1, ws.max_row + 1):
            if str(ws.cell(r, 3).value or "").startswith(et):
                return ws.cell(r, col).value
        raise InputMancante(f"'{et}' non trovato nel foglio LSM")

    prima, ultima = int(v("Prima finestra")), int(v("Ultima finestra"))
    if not finestre_foglio:
        from .alberi import distribuzione_exit
        anni_dist = [d.year for d, _ in distribuzione_exit(caso)]
        prima, ultima = min(anni_dist), max(anni_dist)
    return ParametriLSM(str(v("Applicare l'estensione")).strip().lower().startswith("s"), prima, ultima,
                        float(v("Holding minimo")), float(v("IRR obiettivo")),
                        float(v("MoM obiettivo")), float(v("Costi di transazione")), int(v("Numero di simulazioni")))


@dataclass
class Percorsi:
    date: list[dt.date]
    t: np.ndarray
    t_ing: np.ndarray
    E: np.ndarray
    mip: np.ndarray
    netto_fondo: np.ndarray
    costi: np.ndarray
    sconto: np.ndarray


def simula(caso: Caso, data: str, par: ParametriLSM, n: int, seed: int, misura: str = "Q") -> Percorsi:
    m = mercato(caso, data)
    ke = caso.ke_valutazione if data == "valutazione" else caso.ke_ingresso
    drift = m.r if misura == "Q" else math.log(1 + ke)
    data_rif = caso.data_valutazione if data == "valutazione" else caso.data_ingresso
    piano = piano_bp(caso)
    date = [dt.date(y, 12, 31) for y in range(par.prima, par.ultima + 1)
            if dt.date(y, 12, 31) > data_rif and anni(caso.data_ingresso, dt.date(y, 12, 31)) >= par.holding_min - 1e-9]
    t = np.array([anni(data_rif, d) for d in date])
    t_ing = np.array([anni(caso.data_ingresso, d) for d in date])
    rng = np.random.default_rng(seed)
    meta = n // 2
    Z = rng.standard_normal((len(date), meta))
    Z = np.concatenate([Z, -Z], axis=1)
    dt_ = np.diff(np.concatenate([[0.0], t]))
    logE = math.log(m.E0) + np.cumsum(((drift - m.sigma ** 2 / 2) * dt_)[:, None] + m.sigma * np.sqrt(dt_)[:, None] * Z, axis=0)
    E = np.exp(logE)
    mip = np.vstack([np.asarray(payoff(caso, E[j], t_ing[j])["mip"], dtype=float) for j in range(len(date))])
    debito = np.array([piano.debito_totale(d) for d in date])[:, None]
    costi = par.costi * (E + debito)
    return Percorsi(date, t, t_ing, E, mip, E - mip, costi, np.exp(-m.r * t))


def _base(x: np.ndarray, grado: int, scala: float) -> np.ndarray:
    y = x / scala
    return np.vstack([y ** g for g in range(grado + 1)]).T


def regola_obiettivi(caso: Caso, p: Percorsi, par: ParametriLSM) -> np.ndarray:
    inv = caso.investimento_fondo
    mom = p.netto_fondo / inv
    irr = np.where(mom > 0, mom ** (1 / p.t_ing[:, None]), 0.0) - 1
    ok = (mom >= par.mom_obiettivo) & (irr >= par.irr_obiettivo)
    ok[-1] = True
    return np.argmax(ok, axis=0)


def stima_lsm(caso: Caso, p: Percorsi, par: ParametriLSM) -> list[np.ndarray | None]:
    flusso = p.sconto[:, None] * (p.netto_fondo - p.costi)
    scala = float(np.median(p.E[0]))
    valore = flusso[-1].copy()
    coeff = [None] * len(p.date)
    for j in range(len(p.date) - 2, -1, -1):
        X = _base(p.E[j], par.grado, scala)
        b, *_ = np.linalg.lstsq(X, valore, rcond=None)
        coeff[j] = (b, scala)
        continua = X @ b
        esci = flusso[j] >= continua
        valore = np.where(esci, flusso[j], valore)
    return coeff


def regola_lsm(p: Percorsi, coeff, grado: int) -> np.ndarray:
    flusso = p.sconto[:, None] * (p.netto_fondo - p.costi)
    n = p.E.shape[1]
    scelta = np.full(n, len(p.date) - 1)
    deciso = np.zeros(n, dtype=bool)
    for j in range(len(p.date) - 1):
        b, scala = coeff[j]
        continua = _base(p.E[j], grado, scala) @ b
        esci = (~deciso) & (flusso[j] >= continua)
        scelta[esci] = j
        deciso |= esci
    return scelta


def conflitto(caso: Caso, p: Percorsi, scelta: np.ndarray, soglia: str, banda: float = 0.10) -> dict:
    vivo = np.ones(p.E.shape[1], dtype=bool)
    conti = {"sotto": [0, 0], "sopra": [0, 0]}
    for j in range(len(p.date) - 1):
        S = float(np.asarray(soglie(caso, p.t_ing[j])[soglia]))
        esce = scelta == j
        sotto = vivo & (p.E[j] >= (1 - banda) * S) & (p.E[j] < S)
        sopra = vivo & (p.E[j] >= S) & (p.E[j] < (1 + banda) * S)
        conti["sotto"][0] += int(sotto.sum()); conti["sotto"][1] += int((sotto & esce).sum())
        conti["sopra"][0] += int(sopra.sum()); conti["sopra"][1] += int((sopra & esce).sum())
        vivo &= ~esce
    pr = lambda k: conti[k][1] / conti[k][0] if conti[k][0] else float("nan")
    return {"soglia": soglia, "p_exit_appena_sotto": pr("sotto"), "p_exit_appena_sopra": pr("sopra"),
            "casi_sotto": conti["sotto"][0], "casi_sopra": conti["sopra"][0]}


@dataclass
class EsitoRegola:
    regola: str
    valore_mip: float
    errore_standard: float
    valore_fondo: float
    tempo_medio: float
    distribuzione: dict
    quota_sotto_ratchet: float
    quota_oltre_soglia: float
    conflitto: list


def _esito(caso: Caso, p: Percorsi, scelta: np.ndarray, nome: str) -> EsitoRegola:
    idx = np.arange(p.E.shape[1])
    mip = p.sconto[scelta] * p.mip[scelta, idx]
    fondo = p.sconto[scelta] * (p.netto_fondo[scelta, idx] - p.costi[scelta, idx])
    meta = len(idx) // 2
    coppie = (mip[:meta] + mip[meta:]) / 2
    E = p.E[scelta, idx]
    sg = {k: np.array([float(np.asarray(soglie(caso, t)[k])) for t in p.t_ing]) for k in soglie(caso, p.t_ing[0])}
    if caso.codice == "A":
        R = sg["ratchet"][scelta]
        sotto, oltre = float(np.mean((E >= 0.9 * R) & (E < R))), float(np.mean(E >= R))
    elif caso.codice == "B":
        sotto, oltre = float("nan"), float(np.mean(E >= sg["fine_catch_up"][scelta]))
    else:
        sotto, oltre = float("nan"), float(np.mean(E >= sg["hurdle"][scelta]))
    distr = {d.year: float(np.mean(scelta == j)) for j, d in enumerate(p.date)}
    confl = [conflitto(caso, p, scelta, s_) for s_ in sg]
    return EsitoRegola(nome, float(coppie.mean()), float(coppie.std(ddof=1) / math.sqrt(meta)), float(fondo.mean()),
                       float(p.t[scelta].mean()), distr, sotto, oltre, confl)


def valuta(caso: Caso, data: str = "valutazione", n: int | None = None, seed: int | None = None,
           finestre_foglio: bool = False) -> dict[str, EsitoRegola]:
    par = parametri_lsm(caso, finestre_foglio)
    imp = impostazioni(caso)
    n = n or par.n
    seed = imp["seed"] if seed is None else seed
    stima = simula(caso, data, par, n, seed)
    p = simula(caso, data, par, n, seed + 1)
    coeff = stima_lsm(caso, stima, par)
    out = {"Regola obiettivi": _esito(caso, p, regola_obiettivi(caso, p, par), "Regola obiettivi"),
           "LSM (ottimo del fondo)": _esito(caso, p, regola_lsm(p, coeff, par.grado), "LSM (ottimo del fondo)")}
    for j, d in enumerate(p.date):
        out[f"Exit fissa {d.year}"] = _esito(caso, p, np.full(p.E.shape[1], j), f"Exit fissa {d.year}")
    pP = simula(caso, data, par, n, seed + 2, misura="P")
    out["Regola obiettivi – misura reale"] = _esito(caso, pP, regola_obiettivi(caso, pP, par), "Regola obiettivi – misura reale")
    out["LSM – misura reale"] = _esito(caso, pP, regola_lsm(pP, coeff, par.grado), "LSM – misura reale")
    out["LSM (ottimo del fondo)"].regola = "Regola LSM"
    return out
