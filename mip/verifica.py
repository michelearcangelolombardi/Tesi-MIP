# Confronto tra Python ed Excel.
from __future__ import annotations

from pathlib import Path

from openpyxl import load_workbook

from .config import FILE_INPUT
from .input import Caso, anni, apri
from .waterfall import payoff

CONFRONTI = {
    "A": {
        "Incentive shares (MIP)": (None, "mip"),
        "Shareholder loan – fondo": (None, "fondo_shl"),
        "Azioni ordinarie – fondo": (None, "fondo_ordinarie"),
        "Proventi netti del fondo (dopo il MIP)": (None, "proventi_netti_fondo"),
        "V5 – gradino smussato ±δ: incentive shares": ("V5", "mip"),
        "V6 – top slice: incentive shares": ("V6", "mip"),
        "Equity value all'exit – hurdle": (None, "soglia_hurdle"),
        "Equity value all'exit – ratchet": (None, "soglia_ratchet"),
    },
    "B": {
        "Incentive shares (MIP)": (None, "mip"),
        "Shareholder loan – fondo": (None, "fondo_shl"),
        "Preferred – fondo": (None, "fondo_pref"),
        "Azioni ordinarie – fondo": (None, "fondo_ordinarie"),
        "Proventi netti del fondo (dopo il MIP)": (None, "proventi_netti_fondo"),
        "V1 – hurdle solo IRR: incentive shares": ("V1", "mip"),
        "V2 – hurdle solo MoM equivalente: incentive shares": ("V2", "mip"),
        "V7 – senza catch-up (s sull'eccedenza): incentive shares": ("V7", "mip"),
        "V8 – quota unica a parità di fair value: incentive shares": ("V8", "mip"),
        "Equity value all'exit – hurdle": (None, "soglia_hurdle"),
        "Equity value all'exit – fine del catch-up": (None, "soglia_fine_catch_up"),
    },
    "C": {
        "Incentive shares (MIP)": (None, "mip"),
        "Azioni ordinarie – fondo": (None, "fondo_ordinarie"),
        "V3 – solo IRR: incentive shares": ("V3", "mip"),
        "V4 – solo MoM equivalente: incentive shares": ("V4", "mip"),
        "Equity value all'exit – hurdle": (None, "soglia_hurdle"),
    },
}


def _righe(ws) -> dict[str, int]:
    return {str(ws.cell(row=r, column=3).value).strip(): r for r in range(1, ws.max_row + 1) if ws.cell(row=r, column=3).value}


def confronta_waterfall(casi: dict[str, Caso], percorso: Path = FILE_INPUT) -> list[dict]:
    wb = apri(percorso)
    risultati = []
    for k, mappa in CONFRONTI.items():
        ws, caso = wb[f"Waterfall_{k}"], casi[k]
        righe = _righe(ws)
        r_E, r_data, r_test = righe["Equity value all'exit (€m)"], righe["Data di exit"], righe["Caso di test"]
        quota_unica = None
        if k == "B":
            r_q = righe["V8 – quota unica a parità di fair value Black-Scholes all'ingresso"]
            quota_unica = ws.cell(row=r_q, column=5).value
        col = 5
        while ws.cell(row=r_data, column=col).value is not None:
            E = ws.cell(row=r_E, column=col).value
            t = anni(caso.data_ingresso, ws.cell(row=r_data, column=col).value)
            calcoli = {}
            for etichetta, (variante, campo) in mappa.items():
                if variante not in calcoli:
                    calcoli[variante] = payoff(caso, E, t, variante=variante, quota_unica=quota_unica)
                v_py = float(calcoli[variante][campo])
                v_xl = ws.cell(row=righe[etichetta], column=col).value
                risultati.append(dict(caso=k, colonna=ws.cell(row=r_test, column=col).column_letter,
                                      test=ws.cell(row=r_test, column=col).value, grandezza=etichetta,
                                      excel=v_xl, python=v_py, differenza=abs(v_py - v_xl)))
            col += 1
    return risultati


def confronta_test_breakpoint(casi: dict[str, Caso], percorso: Path = FILE_INPUT) -> list[dict]:
    from .waterfall import soglie
    ws = apri(percorso)["Test_Breakpoint"]
    mappa = {"Caso A – hurdle": ("A", "hurdle"), "Caso A – ratchet": ("A", "ratchet"),
             "Caso B – hurdle": ("B", "hurdle"), "Caso B – fine del catch-up": ("B", "fine_catch_up"),
             "Caso C – hurdle": ("C", "hurdle")}
    r_date = _righe(ws)["Soglia"]
    colonne = [c for c in range(4, ws.max_column + 1) if hasattr(ws.cell(row=r_date, column=c).value, "year")]
    risultati = []
    for r in range(r_date + 1, ws.max_row + 1):
        etichetta = ws.cell(row=r, column=3).value
        chiave = next((v for p, v in mappa.items() if str(etichetta or "").startswith(p)), None)
        if chiave is None:
            continue
        k, nome = chiave
        for c in colonne:
            data = ws.cell(row=r_date, column=c).value
            t = anni(casi[k].data_ingresso, data)
            v_py = float(soglie(casi[k], t)[nome])
            v_xl = ws.cell(row=r, column=c).value
            risultati.append(dict(caso=k, soglia=etichetta, data_exit=data.date(), excel=v_xl, python=v_py,
                                  differenza=abs(v_py - v_xl)))
    return risultati
