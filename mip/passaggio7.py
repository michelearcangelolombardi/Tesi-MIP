# Passaggio 7: confronto tra modelli.
from __future__ import annotations

import statistics

import pandas as pd

from .alberi import valore_albero
from .input import Caso
from .lsm import parametri_lsm, valuta
from .montecarlo import componente_ratchet_mc, valore_mc
from .opm import componente_ratchet, mercato, valore_opm
from .pwerm import numero_quantili, pwerm_prassi, pwerm_quantili
from .strutturale import sigma_attivo_coerente

FAMIGLIA = {
    "OPM semplice (exit fissa, senza leaver)": "option-based semplice",
    "Albero binomiale completo": "option-based completo",
    "Albero trinomiale completo": "option-based completo",
    "Monte Carlo completo": "option-based completo",
    "Strutturale completo (σ attivo dei peer)": "strutturale",
    "Strutturale completo (σ attivo coerente)": "strutturale",
    "PWERM da prassi": "PWERM",
    "PWERM da quantili (diagnostico)": "diagnostico",
    "Exit endogena – regola obiettivi": "exit endogena",
    "Exit endogena – LSM (ottimo del fondo)": "exit endogena",
}


def valori_modelli(caso: Caso, data: str) -> dict[str, float]:
    n = numero_quantili()
    v = {
        "OPM semplice (exit fissa, senza leaver)": valore_opm(caso, mercato(caso, data)),
        "Albero binomiale completo": valore_albero(caso, data, "binomiale", exit_incerta=True, con_leaver=True),
        "Albero trinomiale completo": valore_albero(caso, data, "trinomiale", exit_incerta=True, con_leaver=True),
        "Monte Carlo completo": valore_mc(caso, data, exit_incerta=True, con_leaver=True).valore,
        "Strutturale completo (σ attivo dei peer)": valore_mc(caso, data, exit_incerta=True, con_leaver=True, modello="strutturale").valore,
        "Strutturale completo (σ attivo coerente)": valore_mc(caso, data, exit_incerta=True, con_leaver=True, modello="strutturale",
                                                               sigma_attivo=sigma_attivo_coerente(caso, data)).valore,
        "PWERM da prassi": pwerm_prassi(caso, data)[0],
        "PWERM da quantili (diagnostico)": pwerm_quantili(caso, data, n)[0],
    }
    if parametri_lsm(caso).applicare:
        res = valuta(caso, data)
        v["Exit endogena – regola obiettivi"] = res["Regola obiettivi"].valore_mip
        v["Exit endogena – LSM (ottimo del fondo)"] = res["LSM (ottimo del fondo)"].valore_mip
    return v


def _disp(valori: list[float]) -> float:
    return (max(valori) - min(valori)) / statistics.median(valori)


GRUPPI = [
    ("1. Option-based completi (alberi, Monte Carlo): stesse ipotesi, motori diversi", {"option-based completo"}, True),
    ("2. + OPM semplice (prassi: exit fissa, senza leaver)", {"option-based completo", "option-based semplice"}, True),
    ("3. + modello strutturale", {"option-based completo", "option-based semplice", "strutturale"}, True),
    ("4. + PWERM da prassi", {"option-based completo", "option-based semplice", "strutturale", "PWERM"}, False),
    ("5. + exit endogena", {"option-based completo", "option-based semplice", "strutturale", "PWERM", "exit endogena"}, False),
    ("6. + PWERM da quantili (diagnostico)",
     {"option-based completo", "option-based semplice", "strutturale", "PWERM", "exit endogena", "diagnostico"}, False),
]


def tabella_gruppi(valori: dict[str, dict[str, float]]) -> pd.DataFrame:
    righe = []
    for k, v in valori.items():
        for nome, fam, omogeneo in GRUPPI:
            x = {m: val for m, val in v.items() if FAMIGLIA[m] in fam}
            d = _disp(list(x.values()))
            righe.append({"Caso": k, "Gruppo di modelli": nome, "Ipotesi omogenee": "Sì" if omogeneo else "No",
                          "Modelli": len(x), "Minimo (€m)": min(x.values()),
                          "Massimo (€m)": max(x.values()), "Mediana (€m)": statistics.median(x.values()),
                          "Dispersione": d,
                          "Modello minimo": min(x, key=x.get), "Modello massimo": max(x, key=x.get)})
    return pd.DataFrame(righe)


def tabella_fonti(casi: dict[str, Caso]) -> pd.DataFrame:
    righe = []
    for k, c in casi.items():
        opm = valore_opm(c, mercato(c))
        mc_f = valore_mc(c).valore
        mc_i = valore_mc(c, exit_incerta=True).valore
        mc_l = valore_mc(c, con_leaver=True).valore
        mc_c = valore_mc(c, exit_incerta=True, con_leaver=True).valore
        st_p = valore_mc(c, exit_incerta=True, con_leaver=True, modello="strutturale").valore
        st_c = valore_mc(c, exit_incerta=True, con_leaver=True, modello="strutturale", sigma_attivo=sigma_attivo_coerente(c)).valore
        alb = [valore_albero(c, tipo=t) for t in ("binomiale", "trinomiale")]
        motori = [opm, mc_f] + alb
        effetti = [
            ("Motore numerico (OPM, alberi 200 passi, Monte Carlo) a parità di ipotesi", "intervallo", (max(motori) - min(motori)) / statistics.median(motori)),
            ("Exit incerta invece di exit fissa", "Monte Carlo", mc_i / mc_f - 1),
            ("Leaver (valore economico invece di IFRS 2)", "Monte Carlo", mc_l / mc_f - 1),
            ("Modello completo (exit incerta e leaver) invece di OPM semplice", "Monte Carlo vs OPM", mc_c / opm - 1),
            ("Modello strutturale con σ attivo dei peer invece di GBM sull'equity", "modelli completi", st_p / mc_c - 1),
            ("Sola dinamica di leva (strutturale con σ coerente)", "modelli completi", st_c / mc_c - 1),
            ("PWERM da prassi invece di OPM", "exit fissa", pwerm_prassi(c)[0] / opm - 1),
            ("PWERM da quantili (diagnostico) invece di OPM", "exit fissa", pwerm_quantili(c, n=numero_quantili())[0] / opm - 1),
        ]
        if parametri_lsm(c).applicare:
            res = valuta(c)
            effetti += [("Exit endogena: regola obiettivi invece di Exit_Distribuzione", "senza leaver", res["Regola obiettivi"].valore_mip / mc_i - 1),
                        ("Exit endogena: LSM invece di Exit_Distribuzione", "senza leaver", res["LSM (ottimo del fondo)"].valore_mip / mc_i - 1)]
        effetti.sort(key=lambda e: -abs(e[2]))
        for rango, (nome, confronto, eff) in enumerate(effetti, 1):
            righe.append({"Caso": k, "Rango": rango, "Scelta di modello": nome, "Confronto": confronto, "Effetto sul valore": eff})
    return pd.DataFrame(righe)


def tabella_art60(casi: dict[str, Caso], valori_ingresso: dict[str, dict[str, float]]) -> pd.DataFrame:
    righe = []
    for k, v in valori_ingresso.items():
        c = casi[k]
        for m, val in v.items():
            q = val / c.equity_ingresso
            righe.append({"Caso": k, "Modello": m, "Fair value all'ingresso (€m)": val, "Fair value / equity": q,
                          "Soglia 1% rispettata": "Sì" if q >= 0.01 else "No", "Fair value − prezzo (€m)": val - c.importo_mip})
    return pd.DataFrame(righe)


def tabella_sintesi_art60(art: pd.DataFrame) -> pd.DataFrame:
    righe = []
    for k, g in art.groupby("Caso"):
        righe.append({"Caso": k, "Modelli": len(g), "Modelli sopra l'1%": int((g["Soglia 1% rispettata"] == "Sì").sum()),
                      "Minimo": g["Fair value / equity"].min(), "Massimo": g["Fair value / equity"].max(),
                      "Modello minimo": g.loc[g["Fair value / equity"].idxmin(), "Modello"]})
    return pd.DataFrame(righe)


def tabella_ratchet(caso: Caso) -> pd.DataFrame:
    m = mercato(caso)
    q = {"q_ratchet": caso.parametri["q_base"]}
    righe = []
    for conf, incerta, lv in (("Exit fissa, senza leaver", False, False), ("Exit incerta, con leaver", True, True)):
        val = {}
        if not incerta:
            val["OPM (Black-Scholes)"] = componente_ratchet(caso, m)
        for tipo in ("binomiale", "trinomiale"):
            val[f"Albero {tipo}"] = (valore_albero(caso, tipo=tipo, exit_incerta=incerta, con_leaver=lv)
                                     - valore_albero(caso, tipo=tipo, exit_incerta=incerta, con_leaver=lv, parametri=q))
        val["Monte Carlo"] = componente_ratchet_mc(caso, exit_incerta=incerta, con_leaver=lv).valore
        val["Strutturale (σ peer)"] = componente_ratchet_mc(caso, exit_incerta=incerta, con_leaver=lv, modello="strutturale").valore
        for mod, x in val.items():
            righe.append({"Configurazione": conf, "Modello": mod, "Componente ratchet (€m)": x})
        eq = [x for mod, x in val.items() if not mod.startswith("Strutturale")]
        righe.append({"Configurazione": conf, "Modello": "Dispersione modelli sull'equity", "Componente ratchet (€m)": _disp(eq)})
        righe.append({"Configurazione": conf, "Modello": "Dispersione con lo strutturale", "Componente ratchet (€m)": _disp(list(val.values()))})
    return pd.DataFrame(righe)


def esegui(casi: dict[str, Caso]) -> dict[str, pd.DataFrame]:
    val_v = {k: valori_modelli(c, "valutazione") for k, c in casi.items()}
    val_i = {k: valori_modelli(c, "ingresso") for k, c in casi.items()}
    tutti = pd.DataFrame([{"Caso": k, "Modello": m, "Famiglia": FAMIGLIA[m], "Valore alla valutazione (€m)": x,
                           "Valore per azione (€)": x * 1e6 / casi[k].n_mip,
                           "Valore / equity alla valutazione": x / casi[k].equity_valutazione,
                           "Valore all'ingresso (€m)": val_i[k][m]}
                          for k, v in val_v.items() for m, x in v.items()])
    art = tabella_art60(casi, val_i)
    return {
        "Valori_per_modello": tutti,
        "Dispersione_gruppi": tabella_gruppi(val_v),
        "Fonti_di_divergenza": tabella_fonti(casi),
        "Art60_per_modello": art,
        "Art60_sintesi": tabella_sintesi_art60(art),
        "Ratchet_caso_A": tabella_ratchet(casi["A"]),
    }
