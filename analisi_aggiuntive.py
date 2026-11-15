# Analisi aggiuntive.
import datetime as dt
import math
import statistics

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

import mip.strutturale as S
from mip.config import CARTELLA_RISULTATI
from mip.input import anni, carica_casi
from mip.lsm import _esito, parametri_lsm, regola_lsm, simula, stima_lsm
from mip.montecarlo import Processo, impostazioni, valore_mc
from mip.opm import mercato
from mip.pwerm import pwerm_prassi
from mip.waterfall import payoff, soglie

R = CARTELLA_RISULTATI


def politica_esatta(caso, data, par, nx=20001, nq=160):
    m = mercato(caso, data)
    p0 = simula(caso, data, par, 2, 0)
    t, t_ing, date = p0.t, p0.t_ing, p0.date
    piano = S.piano_bp(caso)
    debito = np.array([piano.debito_totale(d) for d in date])
    sT = m.sigma * math.sqrt(t[-1])
    x = np.linspace(math.log(m.E0) - 8 * sT, math.log(m.E0) + 8 * sT, nx)
    E = np.exp(x)

    def flusso(j, Ev):
        mip = np.asarray(payoff(caso, Ev, t_ing[j])["mip"], dtype=float)
        return math.exp(-m.r * t[j]) * (Ev - mip - par.costi * (Ev + debito[j]))

    z, w = np.polynomial.hermite_e.hermegauss(nq)
    w = w / w.sum()
    V, C = flusso(len(t) - 1, E), [None] * len(t)
    for j in range(len(t) - 2, -1, -1):
        dtj = t[j + 1] - t[j]
        xn = x[:, None] + (m.r - m.sigma ** 2 / 2) * dtj + m.sigma * math.sqrt(dtj) * z[None, :]
        C[j] = (np.interp(xn, x, V) * w).sum(axis=1)
        V = np.maximum(flusso(j, E), C[j])

    def scelta(p):
        n = p.E.shape[1]
        s, deciso = np.full(n, len(t) - 1), np.zeros(n, dtype=bool)
        for j in range(len(t) - 1):
            esci = (~deciso) & (flusso(j, p.E[j]) >= np.interp(np.log(p.E[j]), x, C[j]))
            s[esci] = j
            deciso |= esci
        return s
    return scelta


def differenza_appaiata(p, s1, s2, campo):
    idx = np.arange(p.E.shape[1])
    if campo == "mip":
        v1, v2 = p.sconto[s1] * p.mip[s1, idx], p.sconto[s2] * p.mip[s2, idx]
    else:
        v1 = p.sconto[s1] * (p.netto_fondo[s1, idx] - p.costi[s1, idx])
        v2 = p.sconto[s2] * (p.netto_fondo[s2, idx] - p.costi[s2, idx])
    d = v1 - v2
    h = len(d) // 2
    dd = (d[:h] + d[h:]) / 2
    return float(dd.mean()), float(dd.std(ddof=1) / math.sqrt(h))


def exit_esatta(casi):
    righe = []
    for k in ("A", "B"):
        c = casi[k]
        par = parametri_lsm(c)
        seed = impostazioni(c)["seed"]
        for data in ("valutazione", "ingresso"):
            stima = simula(c, data, par, par.n, seed)
            p = simula(c, data, par, par.n, seed + 1)
            s_lsm = regola_lsm(p, stima_lsm(c, stima, par), par.grado)
            s_ex = politica_esatta(c, data, par)(p)
            s_0 = np.zeros(p.E.shape[1], dtype=int)
            e_lsm, e_ex, e_0 = (_esito(c, p, s, n) for s, n in ((s_lsm, "LSM"), (s_ex, "esatta"), (s_0, "prima")))
            d_mip, se_mip = differenza_appaiata(p, s_ex, s_lsm, "mip")
            d_f0, se_f0 = differenza_appaiata(p, s_lsm, s_0, "fondo")
            righe.append({"Caso": k, "Data": data,
                          "MIP LSM": e_lsm.valore_mip, "MIP esatta": e_ex.valore_mip, "MIP prima finestra": e_0.valore_mip,
                          "Esatta − LSM (MIP)": d_mip, "e.s.": se_mip,
                          "Fondo LSM": e_lsm.valore_fondo, "Fondo esatta": e_ex.valore_fondo, "Fondo prima finestra": e_0.valore_fondo,
                          "LSM − prima finestra (fondo)": d_f0, "e.s. ": se_f0,
                          "Exit alla prima finestra, LSM": list(e_lsm.distribuzione.values())[0],
                          "Exit alla prima finestra, esatta": list(e_ex.distribuzione.values())[0],
                          "Rapporto art. 60, LSM": e_lsm.valore_mip / c.equity_ingresso,
                          "Rapporto art. 60, esatta": e_ex.valore_mip / c.equity_ingresso})
    return pd.DataFrame(righe)


def _payout_cumulato(piano, data_rif, data_fine):
    Q, d = 0.0, data_rif
    while d < data_fine:
        fine = min(dt.date(d.year, 12, 31), data_fine)
        if fine == d:
            fine = min(dt.date(d.year + 1, 12, 31), data_fine)
            Q += piano.payout_anno(d.year + 1) * (fine - d).days / 365
        else:
            Q += piano.payout_anno(d.year) * (fine - d).days / 365
        d = fine
    return Q


def _calibra_yield(caso, data="valutazione"):
    m = mercato(caso, data)
    piano = S.piano_bp(caso)
    data_rif = caso.data_valutazione if data == "valutazione" else caso.data_ingresso
    s = S.vol_attivo(caso, data)
    T = anni(data_rif, caso.data_exit)
    Q = _payout_cumulato(piano, data_rif, caso.data_exit)
    K = piano.debito_totale(caso.data_exit)
    V0 = brentq(lambda V: S.call_yield(V, K, T, m.r, Q, s) - m.E0, m.E0 * 1.0001, m.E0 + 20 * K)
    d1 = (math.log(V0 / K) + m.r * T - Q + s * s * T / 2) / (s * math.sqrt(T))
    return V0, s, s * V0 * math.exp(-Q) * norm.cdf(d1) / m.E0


def _processo_yield(caso, data, tempi, date_idx, prob, variante=None, sigma_attivo=None):
    V0, s, _ = _calibra_yield(caso, data)
    piano = S.piano_bp(caso)
    m = mercato(caso, data)
    data_rif = caso.data_valutazione if data == "valutazione" else caso.data_ingresso
    offset = anni(caso.data_ingresso, data_rif)
    date = [data_rif + dt.timedelta(days=int(round(t * 365))) for t in tempi]
    q = np.array([piano.payout_anno(d.year if d != dt.date(d.year, 12, 31) else d.year + 1) for d in date[:-1]])
    debito = np.array([piano.debito_totale(d) for d in date])
    hurdle = lambda k: float(np.asarray(soglie(caso, offset + tempi[k], variante)["hurdle"]))
    strike = lambda k: debito[k] + hurdle(k)
    valore = sum(p_ * S.call_yield(V0, strike(ke), tempi[ke], m.r, _payout_cumulato(piano, data_rif, date[ke]), s)
                 for p_, ke in zip(prob, date_idx))
    return Processo(math.log(V0), m.r, (m.r - q - s * s / 2) * np.diff(tempi), np.full(len(tempi) - 1, s),
                    lambda x, k: np.maximum(0.0, x - debito[k]), strike, valore)


def debito_yield(casi):
    righe, originale = [], S.processo_strutturale
    try:
        for k, c in casi.items():
            mc = valore_mc(c, exit_incerta=True, con_leaver=True).valore
            S.processo_strutturale = _processo_yield
            v = valore_mc(c, exit_incerta=True, con_leaver=True, modello="strutturale").valore
            S.processo_strutturale = originale
            righe.append({"Caso": k, "Volatilità iniziale dell'equity": _calibra_yield(c)[2],
                          "MIP, strutturale a rendimento costante": v, "MIP, Monte Carlo completo": mc,
                          "Scarto": v / mc - 1})
    finally:
        S.processo_strutturale = originale
    return pd.DataFrame(righe)


def margine_comune(casi, margini=(0.0307, 0.0360), euribor=0.0267):
    originale = S.piano_bp

    def con_margine(mrg):
        def f(caso):
            p = originale(caso)
            anni_ = sorted(p.debito)
            p.flussi = {y: p.debito[y - 1] - p.debito[y] + (euribor + mrg) * p.debito[y - 1] for y in anni_[1:]}
            return p
        return f

    base = {k: valore_mc(c, exit_incerta=True, con_leaver=True, modello="strutturale").valore for k, c in casi.items()}
    righe = []
    try:
        for mrg in margini:
            S.piano_bp = con_margine(mrg)
            S._CACHE.clear()
            for k, c in casi.items():
                v = valore_mc(c, exit_incerta=True, con_leaver=True, modello="strutturale").valore
                righe.append({"Margine comune": mrg, "Caso": k, "Strutturale base": base[k], "Strutturale, margine comune": v,
                              "Variazione": v / base[k] - 1})
    finally:
        S.piano_bp = originale
        S._CACHE.clear()
    return pd.DataFrame(righe)


def volatilita_scenari(casi):
    righe = []
    for k, c in casi.items():
        sc = pwerm_prassi(c, "valutazione")[1]
        p = np.array([r["probabilita"] for r in sc], dtype=float)
        F = np.array([r["equity_exit"] for r in sc], dtype=float)
        lnF = np.log(F)
        mu = (p * lnF).sum()
        dev = math.sqrt((p * (lnF - mu) ** 2).sum())
        righe.append({"Caso": k, "Volatilità implicita negli scenari": dev / math.sqrt(c.T),
                      "Volatilità dei modelli a opzioni": mercato(c, "valutazione").sigma})
    return pd.DataFrame(righe)


def ricavate():
    mappa = pd.read_excel(R / "09_approfondimenti.xlsx", "Mappa_condizionata")
    mappa = mappa[mappa.Mappa == "Volatilità × soglie"]
    v = mappa[["OPM semplice (€m)", "Monte Carlo completo (€m)", "Strutturale completo (€m)"]]
    mappa = mappa.assign(**{"Scarto in euro": v.max(axis=1) - v.min(axis=1)})
    return mappa[["Caso", "Δ volatilità", "Fattore soglie", "Scarto in euro", "Dispersione gruppo 3 (+ strutturale)"]]


def main():
    casi = carica_casi()
    fogli = {}
    print("\n1. Exit ottima esatta contro LSM")
    fogli["Exit_esatta"] = exit_esatta(casi)
    print(fogli["Exit_esatta"].round(3).to_string(index=False))
    print("\n2. Strutturale con servizio del debito a rendimento costante")
    fogli["Debito_rendimento"] = debito_yield(casi)
    print(fogli["Debito_rendimento"].round(4).to_string(index=False))
    print("\n3. Strutturale con margine comune")
    fogli["Margine_comune"] = margine_comune(casi)
    print(fogli["Margine_comune"].round(4).to_string(index=False))
    print("\n4. Volatilità implicita negli scenari del PWERM di prassi")
    fogli["Volatilita_scenari"] = volatilita_scenari(casi)
    print(fogli["Volatilita_scenari"].round(4).to_string(index=False))
    print("\n5. Scarti in euro della mappa condizionata")
    fogli["Mappa_euro"] = ricavate()
    print(fogli["Mappa_euro"].round(3).to_string(index=False))
    with pd.ExcelWriter(R / "12_analisi_aggiuntive.xlsx") as xw:
        for n, d in fogli.items():
            d.to_excel(xw, sheet_name=n, index=False)


if __name__ == "__main__":
    main()
