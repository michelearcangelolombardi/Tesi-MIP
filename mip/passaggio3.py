# Passaggio 3: tabelle.
from __future__ import annotations

import statistics

import pandas as pd

from .alberi import passi_albero, valore_albero, valore_opm_exit_incerta
from .input import Caso, InputMancante, apri, quota_V8
from .opm import componente_ratchet, mercato, valore_opm
from .waterfall import VARIANTI

PASSI_CONVERGENZA = (25, 50, 100, 200, 500, 1000)


def _config():
    return [("Exit fissa, senza leaver", False, False), ("Exit incerta, senza leaver", True, False),
            ("Exit fissa, con leaver", False, True), ("Exit incerta, con leaver", True, True)]


def tabella_valori(casi: dict[str, Caso], data: str) -> pd.DataFrame:
    N = passi_albero()
    righe = []
    for k, c in casi.items():
        E0 = c.equity_valutazione if data == "valutazione" else c.equity_ingresso
        for nome, incerta, lv in _config():
            modelli = {}
            if not lv:
                modelli["OPM (Black-Scholes)"] = (valore_opm_exit_incerta(c, data) if incerta else valore_opm(c, mercato(c, data)))
            for tipo in ("binomiale", "trinomiale"):
                modelli[f"Albero {tipo}"] = valore_albero(c, data, tipo, N=N, exit_incerta=incerta, con_leaver=lv)
            for modello, v in modelli.items():
                rat = None
                if k == "A":
                    if modello.startswith("OPM") and not incerta:
                        rat = componente_ratchet(c, mercato(c, data))
                    elif modello.startswith("Albero"):
                        tipo = modello.split()[-1]
                        rat = v - valore_albero(c, data, tipo, N=N, exit_incerta=incerta, con_leaver=lv,
                                                parametri={"q_ratchet": c.parametri["q_base"]})
                righe.append({"Data": data, "Caso": k, "Configurazione": nome, "Modello": modello,
                              "Valore del MIP (€m)": v, "Valore per azione (€)": v * 1e6 / c.n_mip,
                              "Valore / equity": v / E0, "Componente ratchet (€m)": rat})
    return pd.DataFrame(righe)


def tabella_dispersione(valori: pd.DataFrame) -> pd.DataFrame:
    righe = []
    for (data, k, conf), g in valori.groupby(["Data", "Caso", "Configurazione"], sort=False):
        v = g["Valore del MIP (€m)"].tolist()
        disp = (max(v) - min(v)) / statistics.median(v)
        righe.append({"Data": data, "Caso": k, "Configurazione": conf, "Modelli": len(v),
                      "Minimo (€m)": min(v), "Massimo (€m)": max(v), "Dispersione": disp})
    for (data, k), g in valori.groupby(["Data", "Caso"], sort=False):
        opm = g[(g.Configurazione == "Exit fissa, senza leaver") & g.Modello.str.startswith("OPM")]["Valore del MIP (€m)"].iloc[0]
        alb = g[g.Configurazione == "Exit incerta, con leaver"]["Valore del MIP (€m)"].tolist()
        v = [opm] + alb
        disp = (max(v) - min(v)) / statistics.median(v)
        righe.append({"Data": data, "Caso": k, "Configurazione": "Prassi: OPM semplice vs alberi completi", "Modelli": len(v),
                      "Minimo (€m)": min(v), "Massimo (€m)": max(v), "Dispersione": disp})
    return pd.DataFrame(righe)


def tabella_convergenza(casi: dict[str, Caso]) -> pd.DataFrame:
    righe = []
    for k, c in casi.items():
        bs = valore_opm(c, mercato(c, "valutazione"))
        for tipo in ("binomiale", "trinomiale"):
            riga = {"Caso": k, "Albero": tipo, "Black-Scholes": bs}
            for N in PASSI_CONVERGENZA:
                riga[f"N={N}"] = valore_albero(c, "valutazione", tipo, N=N)
            if tipo == "binomiale":
                riga["Media N=1000 e 1001"] = (riga["N=1000"] + valore_albero(c, "valutazione", tipo, N=1001)) / 2
            righe.append(riga)
    return pd.DataFrame(righe)


def tabella_art60(valori: pd.DataFrame, casi: dict[str, Caso]) -> pd.DataFrame:
    v = valori[valori.Data == "ingresso"].copy()
    v["Soglia 1% rispettata"] = v["Valore / equity"].map(lambda x: "Sì" if x >= 0.01 else "No")
    v["Fair value − prezzo (€m)"] = v.apply(lambda r: r["Valore del MIP (€m)"] - casi[r.Caso].importo_mip, axis=1)
    return v[["Caso", "Configurazione", "Modello", "Valore del MIP (€m)", "Valore / equity", "Soglia 1% rispettata",
              "Fair value − prezzo (€m)"]]


def tabella_varianti(casi: dict[str, Caso]) -> pd.DataFrame:
    q8 = quota_V8()
    righe = []
    for k, c in casi.items():
        base_f = valore_opm(c, mercato(c))
        base_i = valore_opm_exit_incerta(c)
        for v in VARIANTI[k]:
            q = q8 if v == "V8" else None
            f = valore_opm(c, mercato(c), variante=v, quota_unica=q)
            i = valore_opm_exit_incerta(c, variante=v, quota_unica=q)
            righe.append({"Caso": k, "Variante": v or "Caso base", "Exit fissa (€m)": f, "Exit incerta (€m)": i,
                          "Δ% exit fissa": f / base_f - 1, "Δ% exit incerta": i / base_i - 1})
    return pd.DataFrame(righe)


def esegui(casi: dict[str, Caso]) -> dict[str, pd.DataFrame]:
    valori = pd.concat([tabella_valori(casi, "valutazione"), tabella_valori(casi, "ingresso")], ignore_index=True)
    return {
        "Valori": valori,
        "Dispersione": tabella_dispersione(valori),
        "Art60_ingresso": tabella_art60(valori, casi),
        "Convergenza_alberi": tabella_convergenza(casi),
        "Varianti": tabella_varianti(casi),
    }
