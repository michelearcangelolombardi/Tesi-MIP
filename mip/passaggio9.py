# Passaggio 9: tasso di sconto del PWERM e mappa condizionata.
from __future__ import annotations

import datetime as dt
import math
import statistics
from dataclasses import replace

import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.optimize import brentq

from .input import Caso, anni
from .montecarlo import valore_mc
from .opm import mercato, valore_opm
from .pwerm import carica_scenari, limite_quantili, pwerm_prassi
from .sensitivity import _scala_soglie
from .strutturale import vol_attivo
from .waterfall import payoff, soglie

VOL = (-0.10, -0.05, 0.0, 0.05, 0.10)
SOGLIE = (0.8, 1.0, 1.2)
EXIT = (-1.0, -0.5, 0.0, 0.5, 1.0)
VOL_EXIT = (-0.05, 0.0, 0.05)


def tabella_tasso_sconto(casi: dict[str, Caso]) -> pd.DataFrame:
    righe = []
    for data in ("valutazione", "ingresso"):
        for k, c in casi.items():
            m = mercato(c, data)
            V = valore_opm(c, m)
            h = m.E0 * 1e-4
            delta = (valore_opm(c, replace(m, E0=m.E0 + h)) - valore_opm(c, replace(m, E0=m.E0 - h))) / (2 * h)
            omega = delta * m.E0 / V
            ke = c.ke_valutazione if data == "valutazione" else c.ke_ingresso
            rf_annuo = math.exp(m.r) - 1
            capm = rf_annuo + omega * (ke - rf_annuo)
            atteso = limite_quantili(c, data) * (1 + ke) ** m.T
            implicito = (atteso / V) ** (1 / m.T) - 1
            rif = c.data_valutazione if data == "valutazione" else c.data_ingresso
            scen = [(s.probabilita, float(payoff(c, s.equity_exit, anni(c.data_ingresso, dt.date(s.anno, 12, 31)))["mip"]),
                     anni(rif, dt.date(s.anno, 12, 31))) for s in carica_scenari(c, data)]
            f = lambda r: sum(p * x * (1 + r) ** (-t) for p, x, t in scen) - V
            try:
                prassi = brentq(f, -0.5, 2.0)
            except ValueError:
                prassi = float("nan")
            pq_ke = limite_quantili(c, data)
            righe.append({"Data": data, "Caso": k, "Valore OPM (€m)": V, "Delta (Δ)": delta, "Elasticità Ω = Δ·E/V": omega,
                          "Ke (equity)": ke, "Risk-free annuo": rf_annuo,
                          "Rendimento richiesto dal MIP (CAPM con Ω)": capm,
                          "Tasso implicito (distribuzione del modello)": implicito,
                          "Tasso implicito (scenari di prassi)": prassi,
                          "PWERM da prassi al Ke (€m)": pwerm_prassi(c, data)[0],
                          "PWERM da prassi al rendimento del MIP (€m)": sum(p * x * (1 + capm) ** (-t) for p, x, t in scen),
                          "PWERM da quantili (limite) al Ke (€m)": pq_ke,
                          "PWERM da quantili (limite) al rendimento del MIP (€m)": atteso * (1 + capm) ** (-m.T),
                          "Tasso implicito / Ke": implicito / ke})
    return pd.DataFrame(righe)


def _sposta_exit(c: Caso, anni_: float) -> Caso:
    mesi = int(round(12 * anni_))
    if not mesi:
        return c
    return replace(c, data_exit=c.data_exit + relativedelta(months=mesi, day=31), extra={**c.extra, "sposta_exit_mesi": mesi})


def _distanza(c: Caso) -> float:
    H = float(soglie(c, c.t_exit)["hurdle"])
    return H / (c.equity_valutazione * math.exp(c.rf_cont_valutazione * c.T))


def tabella_mappa(casi: dict[str, Caso]) -> pd.DataFrame:
    righe = []
    for k, c in casi.items():
        sa0 = vol_attivo(c, "valutazione")
        celle = [("Volatilità × soglie", d, f, 0.0) for d in VOL for f in SOGLIE] + \
                [("Volatilità × exit", d, 1.0, e) for d in VOL_EXIT for e in EXIT]
        for mappa, d, f, e in celle:
            x = _scala_soglie(c, f) if f != 1.0 else c
            x = replace(x, vol_valutazione=c.vol_valutazione + d)
            x = _sposta_exit(x, e)
            v = {"OPM semplice": valore_opm(x, mercato(x)),
                 "Monte Carlo completo": valore_mc(x, exit_incerta=True, con_leaver=True).valore}
            g2 = list(v.values())
            if e == 0.0:
                v["Strutturale completo"] = valore_mc(x, exit_incerta=True, con_leaver=True, modello="strutturale",
                                                      sigma_attivo=sa0 * (c.vol_valutazione + d) / c.vol_valutazione).valore
            d2 = (max(g2) - min(g2)) / statistics.median(g2)
            g3 = list(v.values())
            d3 = (max(g3) - min(g3)) / statistics.median(g3) if len(g3) == 3 else float("nan")
            righe.append({"Caso": k, "Mappa": mappa, "Δ volatilità": d, "Fattore soglie": f, "Δ exit (anni)": e,
                          "Volatilità equity": c.vol_valutazione + d, "Distanza hurdle / equity attesa": _distanza(x),
                          **{f"{m} (€m)": val for m, val in v.items()},
                          "Dispersione gruppo 2 (OPM vs MC completo)": d2,
                          "Dispersione gruppo 3 (+ strutturale)": d3})
    return pd.DataFrame(righe)


def tabella_regole(mappa: pd.DataFrame) -> pd.DataFrame:
    righe = []
    for k, g in mappa.groupby("Caso"):
        vs = g[g.Mappa == "Volatilità × soglie"]
        ve = g[g.Mappa == "Volatilità × exit"]
        for col, nome in (("Dispersione gruppo 2 (OPM vs MC completo)", "gruppo 2"), ("Dispersione gruppo 3 (+ strutturale)", "gruppo 3")):
            r = {"Caso": k, "Dispersione": nome,
                 "Base": vs[(vs["Δ volatilità"] == 0) & (vs["Fattore soglie"] == 1.0)][col].iloc[0],
                 "Volatilità −10 punti": vs[(vs["Δ volatilità"] == -0.10) & (vs["Fattore soglie"] == 1.0)][col].iloc[0],
                 "Volatilità +10 punti": vs[(vs["Δ volatilità"] == 0.10) & (vs["Fattore soglie"] == 1.0)][col].iloc[0],
                 "Soglie × 0,8 (più vicine)": vs[(vs["Δ volatilità"] == 0) & (vs["Fattore soglie"] == 0.8)][col].iloc[0],
                 "Soglie × 1,2 (più lontane)": vs[(vs["Δ volatilità"] == 0) & (vs["Fattore soglie"] == 1.2)][col].iloc[0],
                 "Minimo sulla mappa": vs[col].min(), "Massimo sulla mappa": vs[col].max()}
            if nome == "gruppo 2":
                r["Exit −1 anno"] = ve[(ve["Δ volatilità"] == 0) & (ve["Δ exit (anni)"] == -1.0)][col].iloc[0]
                r["Exit +1 anno"] = ve[(ve["Δ volatilità"] == 0) & (ve["Δ exit (anni)"] == 1.0)][col].iloc[0]
            righe.append(r)
    return pd.DataFrame(righe)


def tabella_griglia(mappa: pd.DataFrame, nome_mappa: str, colonne: str, valori: str) -> pd.DataFrame:
    g = mappa[mappa.Mappa == nome_mappa]
    t = g.pivot_table(index=["Caso", "Δ volatilità"], columns=colonne, values=valori)
    t.columns = [f"{colonne} = {x:g}" for x in t.columns]
    return t.reset_index()


def esegui(casi: dict[str, Caso]) -> dict[str, pd.DataFrame]:
    mappa = tabella_mappa(casi)
    return {"Tasso_sconto_PWERM": tabella_tasso_sconto(casi), "Sintesi_mappa": tabella_regole(mappa),
            "Griglia_vol_soglie_g3": tabella_griglia(mappa, "Volatilità × soglie", "Fattore soglie",
                                                     "Dispersione gruppo 3 (+ strutturale)"),
            "Griglia_vol_soglie_g2": tabella_griglia(mappa, "Volatilità × soglie", "Fattore soglie",
                                                     "Dispersione gruppo 2 (OPM vs MC completo)"),
            "Griglia_vol_exit_g2": tabella_griglia(mappa, "Volatilità × exit", "Δ exit (anni)",
                                                   "Dispersione gruppo 2 (OPM vs MC completo)"),
            "Mappa_condizionata": mappa}
