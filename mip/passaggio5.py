# Passaggio 5: tabelle.
from __future__ import annotations

import statistics

import pandas as pd

from .input import Caso
from .montecarlo import componente_ratchet_mc, valore_mc
from .passaggio3 import _config, tabella_valori
from .strutturale import calibra, dinamica_leva, sigma_attivo_coerente, verifica_calibrazione_mc

MODELLI = ("Monte Carlo sull'equity", "Strutturale (σ attivo dei peer)", "Strutturale (σ attivo coerente)")


def tabella_calibrazione(casi: dict[str, Caso]) -> pd.DataFrame:
    righe = []
    for data in ("valutazione", "ingresso"):
        for k, c in casi.items():
            for nome, sa in (("peer", None), ("coerente", sigma_attivo_coerente(c, data))):
                cal = calibra(c, data, sa)
                media, se = verifica_calibrazione_mc(c, data) if sa is None else (float("nan"), float("nan"))
                righe.append({"Data": data, "Caso": k, "σ attivo": nome, "Volatilità dell'attivo": cal.sigma_attivo,
                              "V0 calibrato (€m)": cal.V0, "EV da business plan (€m)": cal.ev_bp,
                              "V0 / EV − 1": cal.V0 / cal.ev_bp - 1, "Debito all'exit (€m)": cal.debito_exit,
                              "Valore attuale dei flussi al debito fino all'exit (€m)": cal.pv_flussi,
                              "Volatilità equity implicita iniziale": cal.vol_equity_implicita,
                              "Volatilità equity del GBM": cal.vol_equity_gbm,
                              "Equity osservato (€m)": cal.E0, "Verifica MC: equity media scontata (€m)": media,
                              "Errore standard verifica (€m)": se})
    return pd.DataFrame(righe)


def tabella_valori_strutturale(casi: dict[str, Caso], data: str) -> pd.DataFrame:
    righe = []
    for k, c in casi.items():
        E0 = c.equity_valutazione if data == "valutazione" else c.equity_ingresso
        sa = sigma_attivo_coerente(c, data)
        for nome, incerta, lv in _config():
            for modello, kw in zip(MODELLI, ({"modello": "equity"}, {"modello": "strutturale"},
                                             {"modello": "strutturale", "sigma_attivo": sa})):
                r = valore_mc(c, data, exit_incerta=incerta, con_leaver=lv, **kw)
                rat = (componente_ratchet_mc(c, data=data, exit_incerta=incerta, con_leaver=lv, **kw).valore
                       if k == "A" else None)
                righe.append({"Data": data, "Caso": k, "Configurazione": nome, "Modello": modello,
                              "Valore del MIP (€m)": r.valore, "Errore standard (€m)": r.errore_standard,
                              "Valore / equity": r.valore / E0, "Componente ratchet (€m)": rat})
    df = pd.DataFrame(righe)
    base = df[df.Modello == MODELLI[0]].set_index(["Caso", "Configurazione"])["Valore del MIP (€m)"]
    df["Δ% dal MC sull'equity"] = df.apply(lambda r: r["Valore del MIP (€m)"] / base[(r.Caso, r.Configurazione)] - 1, axis=1)
    return df


def tabella_scomposizione(valori: pd.DataFrame) -> pd.DataFrame:
    righe = []
    for (data, k, conf), g in valori.groupby(["Data", "Caso", "Configurazione"], sort=False):
        v = dict(zip(g.Modello, g["Valore del MIP (€m)"]))
        gbm, peer, coer = v[MODELLI[0]], v[MODELLI[1]], v[MODELLI[2]]
        righe.append({"Data": data, "Caso": k, "Configurazione": conf, "MC sull'equity (€m)": gbm,
                      "Strutturale, σ peer (€m)": peer, "Differenza totale (€m)": peer - gbm,
                      "  di cui volatilità di partenza (σ peer − σ coerente)": peer - coer,
                      "  di cui dinamica di leva (σ coerente − GBM)": coer - gbm})
    return pd.DataFrame(righe)


def tabella_dispersione(casi: dict[str, Caso], valori: pd.DataFrame) -> pd.DataFrame:
    equity = tabella_valori(casi, "valutazione")
    righe = []
    for (k, conf), g in equity.groupby(["Caso", "Configurazione"], sort=False):
        s = valori[(valori.Data == "valutazione") & (valori.Caso == k) & (valori.Configurazione == conf)]
        v = dict(zip(s.Modello, s["Valore del MIP (€m)"]))
        base = g["Valore del MIP (€m)"].tolist() + [v[MODELLI[0]]]
        con = base + [v[MODELLI[1]]]
        d0 = (max(base) - min(base)) / statistics.median(base)
        d1 = (max(con) - min(con)) / statistics.median(con)
        righe.append({"Caso": k, "Configurazione": conf, "Dispersione senza strutturale": d0,
                      "Dispersione con strutturale (σ peer)": d1})
    return pd.DataFrame(righe)


def tabella_dinamica(casi: dict[str, Caso]) -> pd.DataFrame:
    righe = []
    for k, c in casi.items():
        for r in dinamica_leva(c, "valutazione"):
            righe.append({"Caso": k, **r})
    return pd.DataFrame(righe)


def tabella_art60(valori: pd.DataFrame, casi: dict[str, Caso]) -> pd.DataFrame:
    v = valori[valori.Data == "ingresso"].copy()
    v["Soglia 1% rispettata"] = v["Valore / equity"].map(lambda x: "Sì" if x >= 0.01 else "No")
    v["Fair value − prezzo (€m)"] = v.apply(lambda r: r["Valore del MIP (€m)"] - casi[r.Caso].importo_mip, axis=1)
    return v[["Caso", "Configurazione", "Modello", "Valore del MIP (€m)", "Valore / equity", "Soglia 1% rispettata",
              "Fair value − prezzo (€m)"]]


def esegui(casi: dict[str, Caso]) -> dict[str, pd.DataFrame]:
    valori = pd.concat([tabella_valori_strutturale(casi, "valutazione"), tabella_valori_strutturale(casi, "ingresso")],
                       ignore_index=True)
    return {
        "Calibrazione": tabella_calibrazione(casi),
        "Valori": valori,
        "Scomposizione": tabella_scomposizione(valori),
        "Dispersione": tabella_dispersione(casi, valori),
        "Dinamica_leva": tabella_dinamica(casi),
        "Art60_ingresso": tabella_art60(valori, casi),
    }
