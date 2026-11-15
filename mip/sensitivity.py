# Passaggio 8: prove di sensitivity.
from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass, field, replace

import numpy as np
from dateutil.relativedelta import relativedelta

from .input import Caso, InputMancante, anni, apri
from .montecarlo import valore_mc
from .opm import mercato, valore_opm
from .pwerm import numero_quantili, pwerm_prassi, pwerm_quantili
from .strutturale import piano_bp

COL = {"A": 4, "B": 5, "C": 6}
OPM, MC, PRASSI, QUANT = "OPM semplice", "Monte Carlo completo", "PWERM da prassi", "PWERM da quantili (diagnostico)"
CONV, FINESTRA = "Convenzione di volatilità", "Finestra di stima della volatilità"
STRESS_EXIT = (-2.0, 2.0)
LEAVER_EXTRA = (0.125, 0.15, 0.20)


def griglia(etichetta: str) -> list:
    ws = apri()["Sensitivity"]
    for r in range(1, ws.max_row + 1):
        if str(ws.cell(r, 3).value or "").startswith(etichetta):
            da, a, passo, livelli = (ws.cell(r, c).value for c in (5, 6, 7, 8))
            if da is not None:
                n = int(round((a - da) / passo))
                return [round(da + i * passo, 10) for i in range(n + 1)]
            return [float(x.replace(",", ".")) / 100 for x in re.findall(r"(\d+(?:,\d+)?)%", str(livelli))]
    raise InputMancante(f"'{etichetta}' non trovato nel foglio Sensitivity")


def _pm(etichetta: str, k: str) -> float:
    ws = apri()["Parametri_Mercato"]
    for r in range(1, ws.max_row + 1):
        if str(ws.cell(r, 3).value or "").startswith(etichetta):
            return float(ws.cell(r, COL[k]).value)
    raise InputMancante(f"'{etichetta}' non trovato in Parametri_Mercato")


def _comps(etichetta: str, k: str) -> float:
    ws = apri()["Comps"]
    for r in range(1, ws.max_row + 1):
        if str(ws.cell(r, 3).value or "").startswith(etichetta):
            return float(ws.cell(r, COL[k]).value)
    raise InputMancante(f"'{etichetta}' non trovato in Comps")


def caso_vol_peer(c: Caso) -> Caso:
    et = "Alternativa: volatilità equity dei peer senza relevering – "
    return replace(c, vol_valutazione=_comps(et + "valutazione", c.codice), vol_ingresso=_comps(et + "ingresso", c.codice))


def premi_dimensionali() -> list[tuple[str, float]]:
    ws = apri()["Size_Premium"]
    out = {}
    for r in range(1, ws.max_row + 1):
        et, sp = ws.cell(r, 3).value, ws.cell(r, 6).value
        if isinstance(sp, (int, float)):
            out[str(et).strip()] = float(sp)
    return [("Nessuno", 0.0), ("Decile 9", out["9"]), ("Micro cap (9–10)", out["Micro Cap (9–10)"]),
            ("Decile 10", out["10"]), ("Decile 10B", out["10B"])]


def vol_attivo_finestre(k: str) -> dict[str, float]:
    ws = apri()["Comps"]
    righe = {"A": range(7, 14), "B": range(19, 24), "C": range(29, 34)}[k]
    col = {"T": 6, "1 anno": 7, "2 anni": 8}
    return {nome: float(np.median([ws.cell(r, c).value * (1 - ws.cell(r, 14).value) for r in righe])) for nome, c in col.items()}


COVID = ((dt.date(2020, 2, 24), dt.date(2020, 5, 8)), (dt.date(2020, 11, 9), dt.date(2020, 12, 9)))


def _rendimenti_peer(k: str) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    ws = apri()[f"Vol_Caso {k}"]
    r0 = next(r for r in range(1, 40) for c in range(1, 12) if ws.cell(r, c).value == "Volatilità finestra T (da Comps)")
    out = {}
    for c in range(1, ws.max_column + 1):
        v = ws.cell(r0 + 1, c).value
        if not isinstance(v, (int, float)):
            continue
        tick = ws.cell(r0, c).value
        cd = next(x for x in range(1, ws.max_column + 1) if ws.cell(r0 + 4, x).value == tick)
        d, r = [], []
        for row in range(r0 + 4, ws.max_row + 1):
            dd, rr = ws.cell(row, cd + 1).value, ws.cell(row, cd + 3).value
            if isinstance(dd, dt.datetime) and isinstance(rr, (int, float)):
                d.append(dd.date()); r.append(float(rr))
        out[float(v)] = (np.array(d), np.array(r))
    return out


def _vol_finestra(d, r, inizio, fine, covid=False) -> float:
    m = (d > inizio) & (d <= fine)
    if covid:
        for a, b in COVID:
            m &= ~((d >= a) & (d <= b))
    return float(r[m].std(ddof=1) * math.sqrt(252))


def vol_attivo_finestre_ingresso(c: Caso) -> dict[str, float]:
    ws = apri()["Comps"]
    righe = {"A": range(7, 14), "B": range(19, 24), "C": range(29, 34)}[c.codice]
    serie = _rendimenti_peer(c.codice)
    fine = c.data_ingresso
    out = {"1 anno": [], "2 anni": [], "Ingresso": []}
    for r in righe:
        vT, L = float(ws.cell(r, 6).value), float(ws.cell(r, 14).value)
        d, x = next(v for t, v in serie.items() if abs(t - vT) < 1e-9)
        out["1 anno"].append(_vol_finestra(d, x, fine.replace(year=fine.year - 1), fine) * (1 - L))
        out["2 anni"].append(_vol_finestra(d, x, fine.replace(year=fine.year - 2), fine) * (1 - L))
        out["Ingresso"].append(_vol_finestra(d, x, dt.date(2019, 12, 30), fine, covid=True) * (1 - L))
    return {k: float(np.median(v)) for k, v in out.items()}


@dataclass
class Prova:
    parametro: str
    livello: str
    numero: float
    caso: Caso
    modelli: tuple
    base: bool = False
    dlom: float = 0.0


def _debito_netto(piano, d: dt.date) -> float:
    y = d.year
    if d == dt.date(y, 12, 31):
        return piano.debito[y]
    f = (d - dt.date(y - 1, 12, 31)).days / 365
    return piano.debito[y - 1] + (piano.debito[y] - piano.debito[y - 1]) * f


def caso_leva(c: Caso, delta: float) -> Caso:
    piano = piano_bp(c)
    D0 = piano.debito[c.data_ingresso.year]
    L0 = D0 / (D0 + c.equity_ingresso)
    k = (L0 + delta) / L0
    E_ing = c.equity_ingresso + D0 - k * D0
    Dv = _debito_netto(piano, c.data_valutazione)
    E_val = c.equity_valutazione + Dv - k * Dv
    Lv, Lv_n = Dv / (Dv + c.equity_valutazione), k * Dv / (k * Dv + E_val)
    L0_n = k * D0 / (k * D0 + E_ing)
    inv_old, inv_new = c.investimento_fondo, E_ing - c.importo_mip
    fondo = [replace(s, importo=s.importo * inv_new / inv_old) for s in c.strumenti_fondo]
    return replace(c, equity_ingresso=E_ing, equity_valutazione=E_val, strumenti_fondo=fondo,
                   vol_valutazione=c.vol_valutazione * (1 - Lv) / (1 - Lv_n),
                   vol_ingresso=c.vol_ingresso * (1 - L0) / (1 - L0_n))


def caso_leva_debt_like(c: Caso) -> Caso:
    piano = piano_bp(c)
    Dv = _debito_netto(piano, c.data_valutazione)
    DL = piano.debito_totale(c.data_valutazione) - Dv
    L, L_dl = Dv / (Dv + c.equity_valutazione), (Dv + DL) / (Dv + DL + c.equity_valutazione)
    return replace(c, vol_valutazione=c.vol_valutazione * (1 - L) / (1 - L_dl))


def _scala_soglie(c: Caso, f: float) -> Caso:
    p = dict(c.parametri)
    for k in ("hurdle_mom", "ratchet_mom"):
        if k in p:
            p[k] *= f
    if "hurdle_irr" in p:
        T0 = c.t_exit
        p["hurdle_irr"] = (f * (1 + p["hurdle_irr"]) ** T0) ** (1 / T0) - 1
    return replace(c, parametri=p)


def _scala_quote(c: Caso, f: float) -> Caso:
    p = dict(c.parametri)
    for k in ("q_base", "q_ratchet", "c", "s", "p"):
        if k in p:
            p[k] *= f
    return replace(c, parametri=p)


def _ke(c: Caso, nuovo_val: float, nuovo_ing: float) -> Caso:
    return replace(c, ke_valutazione=nuovo_val, ke_ingresso=nuovo_ing)


def prove(c: Caso) -> list[Prova]:
    k = c.codice
    ob = (OPM, MC)
    out = []

    def add(par, liv, num, caso, modelli, base=False, dlom=0.0):
        out.append(Prova(par, liv, num, caso, modelli, base, dlom))

    for d in griglia("Volatilità equity"):
        add("Volatilità equity", f"{d:+.0%}", d, replace(c, vol_valutazione=c.vol_valutazione + d, vol_ingresso=c.vol_ingresso + d),
            ob + (QUANT,), d == 0)
    for d in sorted(set(griglia("Data di exit")) | set(STRESS_EXIT)):
        mesi = int(round(12 * d))
        nuovo = replace(c, data_exit=c.data_exit + relativedelta(months=mesi, day=31), extra={**c.extra, "sposta_exit_mesi": mesi})
        add("Data di exit", f"{d:+.1f} anni", d, nuovo, ob, d == 0)
    livelli = sorted(set(griglia("Probabilità annua di uscita")) | set(LEAVER_EXTRA))
    pb = _base_leaver(c)
    for p in sorted(set(livelli) | {pb}):
        add("Probabilità annua di uscita (leaver)", f"{p:.1%}", p, replace(c, extra={**c.extra, "prob_leaver": p}), (MC,),
            abs(p - pb) < 1e-12)
    for x in griglia("DLOM"):
        add("DLOM", f"{x:.0%}", x, c, ob + (PRASSI,), x == 0, dlom=x)
    for f in griglia("Soglie di hurdle e ratchet"):
        add("Soglie di hurdle e ratchet", f"× {f:.1f}", f, _scala_soglie(c, f), ob, abs(f - 1) < 1e-12)
    for d in griglia("Leva"):
        add("Leva", f"{d:+.0%}", d, caso_leva(c, d) if d else c, ob, d == 0)
    for f in griglia("Quote del management"):
        add("Quote del management", f"× {f:.1f}", f, _scala_quote(c, f), ob, abs(f - 1) < 1e-12)
    for d in griglia("Risk-free"):
        rf = lambda x: math.log(math.exp(x) + d)
        add("Risk-free", f"{d:+.1%}", d, replace(c, rf_cont_valutazione=rf(c.rf_cont_valutazione), rf_cont_ingresso=rf(c.rf_cont_ingresso)),
            ob, d == 0)
    for d in griglia("Ke del PWERM"):
        add("Ke del PWERM", f"{d:+.0%}", d, _ke(c, c.ke_valutazione + d, c.ke_ingresso + d), (PRASSI, QUANT), d == 0)
    sp_base = _pm("Premio dimensionale", k)
    for nome, sp in premi_dimensionali():
        add("Premio dimensionale", f"{nome} ({sp:.2%})", sp, _ke(c, c.ke_valutazione - sp_base + sp, c.ke_ingresso - sp_base + sp),
            (PRASSI, QUANT), abs(sp - sp_base) < 1e-12)
    erp = _pm("Equity risk premium – valutazione", k)
    b_v, b_i = _pm("Beta relevered – valutazione", k), _pm("Beta relevered – ingresso", k)
    for x in griglia("Equity risk premium"):
        add("Equity risk premium", f"{x:.1%}", x, _ke(c, c.ke_valutazione + b_v * (x - erp), c.ke_ingresso + b_i * (x - erp)),
            (PRASSI, QUANT), abs(x - erp) < 1e-12)
    bu, bu_adj, t = _pm("Beta unlevered (mediana", k), _pm("Beta unlevered adjusted", k), _pm("Aliquota fiscale", k)
    de_v, de_i = _pm("Debito netto / equity – valutazione", k), _pm("Debito netto / equity – ingresso", k)
    add("Beta", "Raw (base)", 0, c, (PRASSI, QUANT), True)
    add("Beta", "Adjusted (Blume)", 1, _ke(c, c.ke_valutazione + (bu_adj - bu) * (1 + (1 - t) * de_v) * erp,
                                           c.ke_ingresso + (bu_adj - bu) * (1 + (1 - t) * de_i) * erp), (PRASSI, QUANT))
    add("Definizione di leva", "Solo debito netto (base)", 0, c, ob, True)
    add("Definizione di leva", "Debito netto + debt-like", 1, caso_leva_debt_like(c), ob)
    if k == "C":
        for q in griglia("Quota del guadagno oltre l'hurdle"):
            add("Quota del guadagno oltre l'hurdle (C)", f"{q:.0%}", q, replace(c, parametri={**c.parametri, "p": q}), ob,
                abs(q - c.parametri["p"]) < 1e-12)
    fin, fin_i = vol_attivo_finestre(k), vol_attivo_finestre_ingresso(c)
    for nome in ("1 anno", "2 anni", "T"):
        vi = c.vol_ingresso * (fin_i[nome] / fin_i["Ingresso"] if nome != "T" else 1.0)
        add("Finestra di stima della volatilità", nome, ["1 anno", "2 anni", "T"].index(nome),
            replace(c, vol_valutazione=c.vol_valutazione * fin[nome] / fin["T"], vol_ingresso=vi), ob, nome == "T")
    add(CONV, "Relevered (base)", 0, c, ob, True)
    add(CONV, "Peer, senza relevering", 1, caso_vol_peer(c), ob)
    return out


def _base_leaver(c: Caso) -> float:
    from .alberi import leaver
    return leaver(replace(c, extra={})).prob_annua


def valori(prova: Prova, data: str, n_quantili: int) -> dict[str, float]:
    c, out = prova.caso, {}
    if OPM in prova.modelli:
        out[OPM] = valore_opm(c, mercato(c, data))
    if MC in prova.modelli:
        out[MC] = valore_mc(c, data, exit_incerta=True, con_leaver=True).valore
    if PRASSI in prova.modelli:
        out[PRASSI] = pwerm_prassi(c, data)[0]
    if QUANT in prova.modelli:
        out[QUANT] = pwerm_quantili(c, data, n_quantili)[0]
    return {m: v * (1 - prova.dlom) for m, v in out.items()}


@dataclass
class Info:
    tipo: str
    plausibile: callable
    motivo: str
    unita: tuple | None = None


INFO = {
    "Volatilità equity": Info("stima", lambda x: abs(x) <= 0.05 + 1e-12,
                              "±5 punti: ordine delle differenze prodotte dalle finestre di stima (da −5,6 a +8,9 punti sulla volatilità relevered) e dal panel",
                              ("per 1 punto di volatilità", 0.01, -0.05, 0.05)),
    "Data di exit": Info("stima", lambda x: abs(x) <= 1.0 + 1e-12, "±1 anno attorno alla data del business plan; ±2 anni è un valore di stress",
                         ("per 1 anno di exit", 1.0, -1.0, 1.0)),
    "Probabilità annua di uscita (leaver)": Info("stima", lambda x: 0.025 - 1e-12 <= x <= 0.15 + 1e-12,
                                                 "2,5–15% annuo: dall'ipotesi prudente al turnover osservato sui CEO; 20% è un valore di stress",
                                                 ("per 1 punto di probabilità annua", 0.01, 0.025, 0.15)),
    "DLOM": Info("stima", lambda x: x <= 0.20 + 1e-12, "0–20%: il 30% è un valore di stress",
                 ("per 10 punti di DLOM", 0.10, 0.0, 0.10)),
    "Risk-free": Info("stima", lambda x: abs(x) <= 0.005 + 1e-12, "±0,5 punti (es. swap contro Bund interpolato)",
                      ("per 1 punto di risk-free", 0.01, -0.005, 0.005)),
    "Ke del PWERM": Info("stima", lambda x: abs(x) <= 0.01 + 1e-12, "±1 punto", ("per 1 punto di Ke", 0.01, -0.01, 0.01)),
    "Premio dimensionale": Info("stima", lambda x: 0.0167 <= x <= 0.0438, "tra decile 9 e decile 10 (Kroll)",
                                ("per 1 punto di premio dimensionale", 0.01, None, None)),
    "Equity risk premium": Info("stima", lambda x: True, "intervallo Kroll 5,0–6,0%", ("per 1 punto di ERP", 0.01, 0.05, 0.06)),
    "Beta": Info("stima", lambda x: True, "raw o adjusted (Blume)"),
    "Finestra di stima della volatilità": Info("stima", lambda x: True, "1 anno, 2 anni o T"),
    "Definizione di leva": Info("stima", lambda x: True, "con o senza debt-like (regola 29)"),
    "Soglie di hurdle e ratchet": Info("contratto", lambda x: True, "termine dello statuto"),
    "Quote del management": Info("contratto", lambda x: True, "termine dello statuto"),
    "Quota del guadagno oltre l'hurdle (C)": Info("contratto", lambda x: True, "termine dello statuto"),
    "Leva": Info("struttura", lambda x: True, "struttura del deal, nota all'ingresso"),
    CONV: Info("convenzione", lambda x: False, "scelta di metodo: mediana delle volatilità equity dei peer senza relevering"),
}
