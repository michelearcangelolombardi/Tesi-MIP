# Passaggio 2: PWERM da prassi e da quantili.
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from scipy.stats import norm

from .config import FILE_INPUT
from .input import Caso, InputMancante, anni, apri
from .opm import Mercato, mercato, valore_opm
from .waterfall import payoff, soglie


@dataclass
class Scenario:
    nome: str
    tipo: str
    anno: int
    equity_exit: float
    probabilita: float
    anni_sconto: float
    valore_excel: float


def carica_scenari(caso: Caso, data: str = "valutazione", percorso: Path = FILE_INPUT) -> list[Scenario]:
    ws = apri(percorso)["Scenari_PWERM"]
    titolo = f"Caso {caso.codice} – alla data di {data}"
    r0 = next((r for r in range(1, ws.max_row + 1) if str(ws.cell(r, 3).value or "").startswith(titolo)), None)
    if r0 is None:
        raise InputMancante(f"Blocco '{titolo}' non trovato in Scenari_PWERM")
    out = []
    for r in range(r0 + 1, ws.max_row + 1):
        etichetta = str(ws.cell(r, 3).value or "")
        if etichetta.startswith("Totale probabilità"):
            break
        anno = ws.cell(r, 5).value
        if isinstance(anno, (int, float)) and ws.cell(r, 10).value is not None:
            out.append(Scenario(etichetta, ws.cell(r, 4).value, int(anno), float(ws.cell(r, 9).value),
                                float(ws.cell(r, 10).value), float(ws.cell(r, 11).value), float(ws.cell(r, 12).value)))
    return out


def numero_quantili(percorso: Path = FILE_INPUT) -> int:
    ws = apri(percorso)["Parametri_Mercato"]
    for r in range(1, ws.max_row + 1):
        if str(ws.cell(r, 3).value or "").startswith("Numero di quantili"):
            return int(ws.cell(r, 4).value)
    raise InputMancante("Numero di quantili non trovato in Parametri_Mercato")


def _riferimento(caso: Caso, data: str):
    if data == "valutazione":
        return caso.data_valutazione, caso.equity_valutazione, caso.T, caso.ke_valutazione, caso.vol_valutazione
    return caso.data_ingresso, caso.equity_ingresso, caso.t_exit, caso.ke_ingresso, caso.vol_ingresso


def regime(caso: Caso, E: float, t: float) -> str:
    sg = {k: float(v) for k, v in soglie(caso, t).items()}
    if E < sg["hurdle"]:
        return "sotto l'hurdle"
    if caso.codice == "A":
        return "oltre il ratchet" if E >= sg["ratchet"] else "tra hurdle e ratchet"
    if caso.codice == "B":
        return "oltre il catch-up" if E >= sg["fine_catch_up"] else "nel catch-up"
    return "oltre l'hurdle"


def pwerm_prassi(caso: Caso, data: str = "valutazione", percorso: Path = FILE_INPUT) -> tuple[float, list[dict]]:
    data_rif, E0, _, ke, _ = _riferimento(caso, data)
    righe, totale = [], 0.0
    for s in carica_scenari(caso, data, percorso):
        data_exit = dt.date(s.anno, 12, 31)
        t = anni(caso.data_ingresso, data_exit)
        k = anni(data_rif, data_exit)
        mip = float(payoff(caso, s.equity_exit, t)["mip"])
        valore = s.probabilita * mip * (1 + ke) ** (-k)
        totale += valore
        righe.append(dict(scenario=s.nome, anno_exit=s.anno, probabilita=s.probabilita, equity_exit=s.equity_exit,
                          regime=regime(caso, s.equity_exit, t), mip_exit=mip, anni_sconto=k,
                          valore_attuale_pesato=valore, equity_attuale_pesata=s.probabilita * s.equity_exit * (1 + ke) ** (-k),
                          controllo_excel=s.valore_excel))
    return totale, righe


def medie_condizionate(E0: float, ke: float, sigma: float, T: float, n: int) -> np.ndarray:
    sq = sigma * math.sqrt(T)
    z = norm.ppf(np.linspace(0, 1, n + 1))
    phi = norm.cdf(z - sq)
    return E0 * (1 + ke) ** T * (phi[1:] - phi[:-1]) * n


def pwerm_quantili(caso: Caso, data: str = "valutazione", n: int = 5) -> tuple[float, list[dict]]:
    _, E0, T, ke, sigma = _riferimento(caso, data)
    medie = medie_condizionate(E0, ke, sigma, T, n)
    t = caso.t_exit
    mip = np.asarray(payoff(caso, medie, t)["mip"], dtype=float)
    sconto = (1 + ke) ** (-T)
    righe = [dict(quantile=i + 1, probabilita=1 / n, equity_exit=float(medie[i]), regime=regime(caso, float(medie[i]), t),
                  mip_exit=float(mip[i]), valore_attuale_pesato=float(mip[i] / n * sconto)) for i in range(n)]
    return float(mip.mean() * sconto), righe


def limite_quantili(caso: Caso, data: str = "valutazione") -> float:
    _, _, _, ke, _ = _riferimento(caso, data)
    return valore_opm(caso, mercato(caso, data, r=math.log(1 + ke)))


def scomposizione(caso: Caso, data: str = "valutazione", n: int = 5) -> dict:
    prassi = pwerm_prassi(caso, data)[0]
    quant = pwerm_quantili(caso, data, n)[0]
    lim = limite_quantili(caso, data)
    opm = valore_opm(caso, mercato(caso, data))
    return {"PWERM da prassi": prassi, f"PWERM da quantili (N={n})": quant, "PWERM da quantili (N→∞)": lim, "OPM": opm,
            "Differenza totale (prassi − OPM)": prassi - opm,
            "  di cui soggettività degli scenari (prassi − quantili)": prassi - quant,
            "  di cui discretizzazione (quantili N − quantili ∞)": quant - lim,
            "  di cui misura e tasso di sconto (quantili ∞ − OPM)": lim - opm}
