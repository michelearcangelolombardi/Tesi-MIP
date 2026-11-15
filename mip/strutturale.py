# Passaggio 5: modello strutturale.
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np
from scipy import integrate
from scipy.optimize import brentq
from scipy.stats import norm

from .input import Caso, InputMancante, anni, apri
from .opm import mercato
from .waterfall import payoff, soglie

COLONNA_CASO = {"A": 4, "B": 5, "C": 6}


@dataclass
class PianoBP:
    anni: list[int]
    ev: dict[int, float]
    debito: dict[int, float]
    debt_like: dict[int, float]
    payout: dict[int, float]
    flussi: dict[int, float]

    def debito_totale(self, d: dt.date) -> float:
        y = d.year
        if d == dt.date(y, 12, 31):
            return self.debito[y] + self.debt_like[y]
        a, b = y - 1, y
        f = (d - dt.date(a, 12, 31)).days / 365
        k = lambda z: self.debito[z] + self.debt_like[z]
        return k(a) + (k(b) - k(a)) * f

    def payout_anno(self, y: int) -> float:
        return self.payout.get(y, self.payout[max(self.payout)])


def piano_bp(caso: Caso) -> PianoBP:
    ws = apri()["BP"]
    anni_ = {c: int(ws.cell(5, c).value) for c in range(4, 14) if isinstance(ws.cell(5, c).value, (int, float))}
    r0 = next((r for r in range(1, ws.max_row + 1) if str(ws.cell(r, 3).value or "").strip() == f"Caso {caso.codice}"), None)
    if r0 is None:
        raise InputMancante(f"Blocco 'Caso {caso.codice}' non trovato nel foglio BP")

    def riga(etichetta):
        for r in range(r0 + 1, r0 + 15):
            if str(ws.cell(r, 3).value or "").startswith(etichetta):
                return {y: float(ws.cell(r, c).value) for c, y in anni_.items() if ws.cell(r, c).value is not None}
        raise InputMancante(f"'{etichetta}' non trovato nel foglio BP, Caso {caso.codice}")

    return PianoBP(sorted(anni_.values()), riga("EV da business plan"), riga("Debito netto"),
                   riga("Debt-like items"), riga("Payout yield"), riga("Flussi al debito"))


def flussi_debito(piano: PianoBP, data_rif: dt.date, data_fine: dt.date) -> list[tuple[float, float]]:
    out = []
    for y in range(data_rif.year, data_fine.year + 1):
        d = dt.date(y, 12, 31)
        if d <= data_rif or d > data_fine:
            continue
        f = 1.0 if data_rif <= dt.date(y - 1, 12, 31) else (d - data_rif).days / 365
        out.append((anni(data_rif, d), piano.flussi[y] * f))
    return out


def debito_a_gradini(piano: PianoBP, data_rif: dt.date, d: dt.date) -> float:
    ultimo = max([x for x in (dt.date(y, 12, 31) for y in range(data_rif.year, d.year + 1)) if data_rif < x <= d],
                 default=None)
    return piano.debito_totale(data_rif if ultimo is None else ultimo)


def vol_attivo(caso: Caso, data: str) -> float:
    ws = apri()["Comps"]
    et = f"Volatilità dell'attivo – {data}"
    for r in range(1, ws.max_row + 1):
        if str(ws.cell(r, 3).value or "").startswith(et):
            return float(ws.cell(r, COLONNA_CASO[caso.codice]).value)
    raise InputMancante(f"'{et}' non trovato nel foglio Comps")


def call_yield(S, K, T, r, Q, s):
    if K <= 0:
        return S * math.exp(-Q) - K * math.exp(-r * T)
    d1 = (math.log(S / K) + r * T - Q + s * s * T / 2) / (s * math.sqrt(T))
    return S * math.exp(-Q) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d1 - s * math.sqrt(T))


N_CALIBRAZIONE, SEED_CALIBRAZIONE = 200_000, 42


def _normali(n_tempi: int, n: int = N_CALIBRAZIONE, seed: int = SEED_CALIBRAZIONE) -> np.ndarray:
    z = np.random.default_rng(seed).standard_normal((n_tempi, n // 2))
    return np.concatenate([z, -z], axis=1)


def _tempi_flussi(flussi, T):
    return sorted({round(t, 12) for t, _ in flussi if t <= T + 1e-9} | {round(T, 12)})


def equity_mc(V0, s, r, T, flussi, K, Z) -> np.ndarray:
    tempi = _tempi_flussi(flussi, T)
    cf = {}
    for t, c in flussi:
        cf[round(t, 12)] = cf.get(round(t, 12), 0.0) + c
    V, prec = np.full(Z.shape[1], float(V0)), 0.0
    for i, t in enumerate(tempi):
        dt_ = t - prec
        V = V * np.exp((r - s * s / 2) * dt_ + s * math.sqrt(dt_) * Z[i]) - cf.get(t, 0.0)
        prec = t
    return math.exp(-r * T) * np.maximum(V - K, 0.0)


@dataclass
class Calibrazione:
    data: str
    V0: float
    E0: float
    sigma_attivo: float
    T: float
    pv_flussi: float
    debito_exit: float
    ev_bp: float
    vol_equity_implicita: float
    vol_equity_gbm: float


_CACHE: dict = {}


def calibra(caso: Caso, data: str = "valutazione", sigma_attivo: float | None = None) -> Calibrazione:
    m = mercato(caso, data)
    piano = piano_bp(caso)
    data_rif = caso.data_valutazione if data == "valutazione" else caso.data_ingresso
    s = vol_attivo(caso, data) if sigma_attivo is None else sigma_attivo
    chiave = (caso.codice, data, round(s, 12), m.E0, m.r, caso.data_exit, data_rif)
    if chiave in _CACHE:
        return _CACHE[chiave]
    T = anni(data_rif, caso.data_exit)
    K = debito_a_gradini(piano, data_rif, caso.data_exit)
    fl = flussi_debito(piano, data_rif, caso.data_exit)
    Z = _normali(len(_tempi_flussi(fl, T)))
    eq = lambda V: float(equity_mc(V, s, m.r, T, fl, K, Z).mean())
    V0 = brentq(lambda V: eq(V) - m.E0, m.E0, m.E0 + 20 * K + sum(c for _, c in fl), xtol=1e-10)
    h = 0.01 * V0
    delta = (eq(V0 + h) - eq(V0 - h)) / (2 * h)
    sigma_E = s * V0 * delta / m.E0
    pv = sum(c * math.exp(-m.r * t) for t, c in fl)
    ev_bp = m.E0 + piano.debito_totale(data_rif)
    cal = Calibrazione(data, V0, m.E0, s, T, pv, K, ev_bp, sigma_E, m.sigma)
    _CACHE[chiave] = cal
    return cal


def sigma_attivo_coerente(caso: Caso, data: str = "valutazione") -> float:
    obiettivo = mercato(caso, data).sigma
    return brentq(lambda s: calibra(caso, data, s).vol_equity_implicita - obiettivo, 0.01, obiettivo, xtol=1e-10)


def processo_strutturale(caso: Caso, data: str, tempi: np.ndarray, date_idx, prob, variante=None,
                         sigma_attivo: float | None = None):
    from .montecarlo import Processo
    cal = calibra(caso, data, sigma_attivo)
    piano = piano_bp(caso)
    m = mercato(caso, data)
    data_rif = caso.data_valutazione if data == "valutazione" else caso.data_ingresso
    date = [data_rif + dt.timedelta(days=int(round(t * 365))) for t in tempi]
    dt_ = np.diff(tempi)
    s = cal.sigma_attivo
    fl = flussi_debito(piano, data_rif, max(date))
    salti = np.zeros(len(tempi))
    for t, c in fl:
        salti[int(np.argmin(np.abs(tempi - t)))] += c
    debito = np.array([debito_a_gradini(piano, data_rif, d) for d in date])
    valore = sum(p_ * math.exp(-m.r * tempi[ke]) * (cal.V0 * math.exp(m.r * tempi[ke])
                                                    - sum(c * math.exp(m.r * (tempi[ke] - t)) for t, c in fl
                                                          if t <= tempi[ke] + 1e-9))
                 for p_, ke in zip(prob, date_idx))
    return Processo(math.log(cal.V0), m.r, (m.r - s * s / 2) * dt_, np.full(len(dt_), s),
                    lambda x, k: np.maximum(0.0, x - debito[k]), lambda k: 0.0, valore,
                    salti=salti, controllo_lineare=True)


def dinamica_leva(caso: Caso, data: str = "valutazione", sigma_attivo: float | None = None) -> list[dict]:
    cal = calibra(caso, data, sigma_attivo)
    piano = piano_bp(caso)
    m = mercato(caso, data)
    s = cal.sigma_attivo
    data_rif = caso.data_valutazione if data == "valutazione" else caso.data_ingresso
    fl = flussi_debito(piano, data_rif, caso.data_exit)
    righe = [{"Data": data_rif, "EV forward (€m)": cal.V0, "Debito netto + debt-like (€m)": piano.debito_totale(data_rif),
              "Leva attesa": piano.debito_totale(data_rif) / cal.V0, "Volatilità equity implicita (strutturale)": cal.vol_equity_implicita,
              "Volatilità equity (GBM)": cal.vol_equity_gbm}]
    for y in range(data_rif.year, caso.data_exit.year + 1):
        d = dt.date(y, 12, 31)
        if d <= data_rif:
            continue
        t = anni(data_rif, d)
        V = cal.V0 * math.exp(m.r * t) - sum(c * math.exp(m.r * (t - tj)) for tj, c in fl if tj <= t + 1e-9)
        D = piano.debito_totale(d)
        tau = cal.T - t
        if tau > 1e-9:
            resto = [(tj - t, c) for tj, c in fl if tj > t + 1e-9]
            Z = _normali(len(_tempi_flussi(resto, tau)))
            eq = lambda x: float(equity_mc(x, s, m.r, tau, resto, cal.debito_exit, Z).mean())
            E, h = eq(V), 0.01 * V
            vol = s * V * (eq(V + h) - eq(V - h)) / (2 * h) / E
        else:
            vol = s * V / (V - D)
        righe.append({"Data": d, "EV forward (€m)": V, "Debito netto + debt-like (€m)": D, "Leva attesa": D / V,
                      "Volatilità equity implicita (strutturale)": vol, "Volatilità equity (GBM)": cal.vol_equity_gbm})
    return righe


def verifica_calibrazione_mc(caso: Caso, data: str = "valutazione", n: int = 200_000, seed: int = 7) -> tuple[float, float]:
    cal = calibra(caso, data)
    m = mercato(caso, data)
    piano = piano_bp(caso)
    data_rif = caso.data_valutazione if data == "valutazione" else caso.data_ingresso
    fl = flussi_debito(piano, data_rif, caso.data_exit)
    Z = _normali(len(_tempi_flussi(fl, cal.T)), n, seed)
    eq = equity_mc(cal.V0, cal.sigma_attivo, m.r, cal.T, fl, cal.debito_exit, Z)
    coppie = (eq[: n // 2] + eq[n // 2:]) / 2
    return float(coppie.mean()), float(coppie.std(ddof=1) / math.sqrt(n // 2))
