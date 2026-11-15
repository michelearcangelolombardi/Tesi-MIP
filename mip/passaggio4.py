# Passaggio 4: tabelle.
from __future__ import annotations

import statistics

import pandas as pd

from .input import Caso, quota_V8
from .montecarlo import componente_ratchet_mc, valore_mc
from .opm import mercato, valore_opm
from .passaggio3 import _config, tabella_valori
from .pwerm import numero_quantili, pwerm_prassi, pwerm_quantili
from .waterfall import VARIANTI

SIMULAZIONI_CONVERGENZA = (10_000, 50_000, 100_000, 500_000)


def tabella_mc(casi: dict[str, Caso], data: str) -> pd.DataFrame:
    righe = []
    for k, c in casi.items():
        E0 = c.equity_valutazione if data == "valutazione" else c.equity_ingresso
        for nome, incerta, lv in _config():
            r = valore_mc(c, data, exit_incerta=incerta, con_leaver=lv)
            rat = componente_ratchet_mc(c, data=data, exit_incerta=incerta, con_leaver=lv).valore if k == "A" else None
            lo, hi = r.intervallo_95
            righe.append({"Data": data, "Caso": k, "Configurazione": nome, "Modello": "Monte Carlo",
                          "Valore del MIP (€m)": r.valore, "Errore standard (€m)": r.errore_standard,
                          "IC 95% min": lo, "IC 95% max": hi,
                          "Senza variabile di controllo (€m)": r.valore_semplice,
                          "Errore standard senza controllo (€m)": r.errore_standard_semplice,
                          "Valore per azione (€)": r.valore * 1e6 / c.n_mip, "Valore / equity": r.valore / E0,
                          "Componente ratchet (€m)": rat, "Percorsi": r.n_percorsi})
    return pd.DataFrame(righe)


def tabella_mappa(tutti: pd.DataFrame) -> pd.DataFrame:
    righe = []
    for (data, k, conf), g in tutti.groupby(["Data", "Caso", "Configurazione"], sort=False):
        v = g["Valore del MIP (€m)"].tolist()
        disp = (max(v) - min(v)) / statistics.median(v)
        riga = {"Data": data, "Caso": k, "Configurazione": conf}
        riga.update({m: val for m, val in zip(g.Modello, v)})
        riga.update({"Dispersione": disp})
        if k == "A":
            r = g["Componente ratchet (€m)"].dropna().tolist()
            if len(r) > 1:
                riga["Dispersione componente ratchet"] = (max(r) - min(r)) / statistics.median(r)
        righe.append(riga)
    for (data, k), g in tutti.groupby(["Data", "Caso"], sort=False):
        semplice = g[(g.Configurazione == "Exit fissa, senza leaver") & g.Modello.str.startswith("OPM")]["Valore del MIP (€m)"].iloc[0]
        completi = g[g.Configurazione == "Exit incerta, con leaver"]["Valore del MIP (€m)"].tolist()
        v = [semplice] + completi
        disp = (max(v) - min(v)) / statistics.median(v)
        righe.append({"Data": data, "Caso": k, "Configurazione": "Prassi: OPM semplice vs modelli completi",
                      "Dispersione": disp})
    return pd.DataFrame(righe)


def tabella_pwerm_vs_option(casi: dict[str, Caso], tutti: pd.DataFrame) -> pd.DataFrame:
    n = numero_quantili()
    righe = []
    for k, c in casi.items():
        g = tutti[(tutti.Data == "valutazione") & (tutti.Caso == k)]
        base = statistics.median(g[g.Configurazione == "Exit fissa, senza leaver"]["Valore del MIP (€m)"])
        completo = statistics.median(g[g.Configurazione == "Exit incerta, con leaver"]["Valore del MIP (€m)"])
        for nome, v in (("PWERM da prassi", pwerm_prassi(c)[0]), (f"PWERM da quantili (N={n})", pwerm_quantili(c, n=n)[0])):
            righe.append({"Caso": k, "Modello": nome, "Valore (€m)": v,
                          "Mediana option-based, exit fissa senza leaver (€m)": base, "Scarto % (exit fissa)": v / base - 1,
                          "Mediana option-based, exit incerta con leaver (€m)": completo, "Scarto % (completo)": v / completo - 1})
    return pd.DataFrame(righe)


def tabella_convergenza(casi: dict[str, Caso]) -> pd.DataFrame:
    righe = []
    for k, c in casi.items():
        bs = valore_opm(c, mercato(c))
        for n in SIMULAZIONI_CONVERGENZA:
            r = valore_mc(c, n=n)
            righe.append({"Caso": k, "Simulazioni": n, "Black-Scholes": bs,
                          "MC semplice": r.valore_semplice, "Errore standard semplice": r.errore_standard_semplice,
                          "Scarto semplice in errori standard": (r.valore_semplice - bs) / r.errore_standard_semplice,
                          "MC con controllo": r.valore, "Errore standard con controllo": r.errore_standard})
    return pd.DataFrame(righe)


def tabella_varianti(casi: dict[str, Caso]) -> pd.DataFrame:
    q8 = quota_V8()
    righe = []
    for k, c in casi.items():
        base = valore_mc(c, exit_incerta=True, con_leaver=True).valore
        for v in VARIANTI[k]:
            r = valore_mc(c, exit_incerta=True, con_leaver=True, variante=v, quota_unica=q8 if v == "V8" else None)
            righe.append({"Caso": k, "Variante": v or "Caso base", "Valore MC completo (€m)": r.valore,
                          "Errore standard (€m)": r.errore_standard, "Δ% dal caso base": r.valore / base - 1})
    return pd.DataFrame(righe)


def tabella_leaver_fair_value(casi: dict[str, Caso]) -> pd.DataFrame:
    righe = []
    for data in ("valutazione", "ingresso"):
        for k, c in casi.items():
            E0 = c.equity_valutazione if data == "valutazione" else c.equity_ingresso
            senza = valore_mc(c, data, exit_incerta=True).valore
            for regola, fv in (("Valore intrinseco (caso base)", False), ("Fair value (valore dell'opzione)", True)):
                r = valore_mc(c, data, exit_incerta=True, con_leaver=True, leaver_fair_value=fv)
                righe.append({"Data": data, "Caso": k, "Prezzo pagato al leaver": regola,
                              "Valore del MIP (€m)": r.valore, "Errore standard (€m)": r.errore_standard,
                              "Senza leaver (€m)": senza, "Effetto dei leaver": r.valore / senza - 1,
                              "Valore / equity": r.valore / E0})
    return pd.DataFrame(righe)


def tabella_art60(tutti: pd.DataFrame, casi: dict[str, Caso]) -> pd.DataFrame:
    v = tutti[tutti.Data == "ingresso"].copy()
    v["Soglia 1% rispettata"] = v["Valore / equity"].map(lambda x: "Sì" if x >= 0.01 else "No")
    v["Fair value − prezzo (€m)"] = v.apply(lambda r: r["Valore del MIP (€m)"] - casi[r.Caso].importo_mip, axis=1)
    return v[["Caso", "Configurazione", "Modello", "Valore del MIP (€m)", "Valore / equity", "Soglia 1% rispettata",
              "Fair value − prezzo (€m)"]]


def esegui(casi: dict[str, Caso]) -> dict[str, pd.DataFrame]:
    mc = pd.concat([tabella_mc(casi, "valutazione"), tabella_mc(casi, "ingresso")], ignore_index=True)
    alberi = pd.concat([tabella_valori(casi, "valutazione"), tabella_valori(casi, "ingresso")], ignore_index=True)
    tutti = pd.concat([alberi, mc[alberi.columns.intersection(mc.columns)]], ignore_index=True)
    ordine = {nome: i for i, (nome, _, _) in enumerate(_config())}
    tutti = tutti.sort_values(["Data", "Caso", "Configurazione"], key=lambda s: s.map(ordine) if s.name == "Configurazione"
                              else s.map({"valutazione": 0, "ingresso": 1}) if s.name == "Data" else s, kind="stable")
    return {
        "Monte_Carlo": mc,
        "Mappa_convergenza": tabella_mappa(tutti),
        "PWERM_vs_option": tabella_pwerm_vs_option(casi, tutti),
        "Art60_ingresso": tabella_art60(tutti, casi),
        "Convergenza_MC": tabella_convergenza(casi),
        "Varianti_MC_completo": tabella_varianti(casi),
        "Leaver_fair_value": tabella_leaver_fair_value(casi),
        "Tutti_i_valori": tutti,
    }
