# Lettura degli input dal file Excel.
from __future__ import annotations

import datetime as dt
import hashlib
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path

from openpyxl import load_workbook

from .config import CASI, FILE_INPUT

COLONNA_ETICHETTE = 3
COLONNE_CASI = {
    "Input_Deal": {"A": 5, "B": 6, "C": 7},
    "Parametri_Mercato": {"A": 4, "B": 5, "C": 6},
    "Termini_MIP": {"A": 4, "B": 5, "C": 6},
}
CT = dict(nome=3, detentore=4, importo=5, numero=6, prezzo=7, rango=8, pik=9, incluso_mip=12)
RAT = (4, 5, 6)
STRUTTURA = {"A": "cliff", "B": "catch-up", "C": "call"}


def anni(d0, d1) -> float:
    if isinstance(d0, dt.datetime):
        d0 = d0.date()
    if isinstance(d1, dt.datetime):
        d1 = d1.date()
    return (d1 - d0).days / 365


def hash_file(percorso: Path) -> str:
    h = hashlib.sha256(Path(percorso).read_bytes()).hexdigest()
    return h[:16]


@lru_cache(maxsize=4)
def apri(percorso: Path = FILE_INPUT):
    return load_workbook(percorso, data_only=True)


class InputMancante(Exception):
    pass


def _riga(ws, etichetta: str, colonna: int = COLONNA_ETICHETTE) -> int:
    for r in range(1, ws.max_row + 1):
        v = ws.cell(row=r, column=colonna).value
        if isinstance(v, str) and v.strip().startswith(etichetta):
            return r
    raise InputMancante(f"Etichetta non trovata nel foglio {ws.title}: '{etichetta}'")


def _valore(ws, etichetta: str, caso: str):
    v = ws.cell(row=_riga(ws, etichetta), column=COLONNE_CASI[ws.title][caso]).value
    if v is None:
        raise InputMancante(
            f"Valore vuoto in {ws.title} / '{etichetta}' / Caso {caso}. "
            "Se la cella contiene una formula, apri il file in Excel e salvalo, così i valori vengono calcolati."
        )
    return v


@dataclass
class Strumento:
    nome: str
    detentore: str
    importo: float
    pik: float
    rango: int | None
    n_strumenti: int | None = None


@dataclass
class Caso:
    codice: str
    nome: str
    struttura: str
    data_ingresso: dt.date
    data_valutazione: dt.date
    data_exit: dt.date
    equity_valutazione: float
    equity_ingresso: float
    strumenti_fondo: list[Strumento]
    importo_mip: float
    n_mip: int
    prezzo_mip: float
    parametri: dict
    vol_valutazione: float
    vol_ingresso: float
    rf_cont_valutazione: float
    rf_cont_ingresso: float
    ke_valutazione: float
    ke_ingresso: float
    extra: dict = field(default_factory=dict)

    @property
    def investimento_fondo(self) -> float:
        return sum(s.importo for s in self.strumenti_fondo)

    @property
    def prezzo_mip_per_azione(self) -> float:
        return self.prezzo_mip

    @property
    def T(self) -> float:
        return anni(self.data_valutazione, self.data_exit)

    @property
    def t_exit(self) -> float:
        return anni(self.data_ingresso, self.data_exit)

    def strumento(self, prefisso: str) -> Strumento | None:
        for s in self.strumenti_fondo:
            if s.nome.startswith(prefisso):
                return s
        return None


def _blocco(ws, caso: str) -> list[int]:
    r = _riga(ws, f"Caso {caso}") + 1
    righe = []
    while True:
        v = ws.cell(row=r, column=COLONNA_ETICHETTE).value
        if v is None or str(v).strip().startswith("Totale"):
            return righe
        righe.append(r)
        r += 1


def carica_casi(percorso: Path = FILE_INPUT) -> dict[str, Caso]:
    wb = apri(percorso)
    idl, cap, rat, pm = wb["Input_Deal"], wb["Cap_Table"], wb["Ratchet"], wb["Parametri_Mercato"]
    casi = {}
    for k in CASI:
        fondo, mip = [], None
        for r in _blocco(cap, k):
            v = lambda col: cap.cell(row=r, column=CT[col]).value
            if v("importo") is None:
                continue
            if v("incluso_mip") == "Sì":
                mip = (float(v("importo")), int(v("numero")), float(v("prezzo")))
            else:
                fondo.append(Strumento(nome=v("nome"), detentore=v("detentore"), importo=float(v("importo")),
                                       pik=float(v("pik") or 0.0), rango=v("rango"), n_strumenti=v("numero")))
        if mip is None:
            raise InputMancante(f"Strumento del MIP non trovato nella Cap_Table, Caso {k}")

        righe = [[rat.cell(row=r, column=c).value for c in RAT] for r in _blocco(rat, k)]
        if k == "A":
            par = dict(hurdle_mom=righe[0][1], q_base=righe[0][2], ratchet_mom=righe[1][1], q_ratchet=righe[1][2])
        elif k == "B":
            par = dict(hurdle_irr=righe[0][0], hurdle_mom=righe[0][1], c=righe[0][2], s=righe[1][2])
        else:
            par = dict(hurdle_irr=righe[0][0], hurdle_mom=righe[0][1], p=righe[0][2])
        if any(v is None for v in par.values()):
            raise InputMancante(f"Parametri del MIP incompleti nel foglio Ratchet, Caso {k}: {par}")
        par = {a: float(b) for a, b in par.items()}

        casi[k] = Caso(
            codice=k,
            nome=_valore(idl, "Nome del caso", k),
            struttura=STRUTTURA[k],
            data_ingresso=_valore(idl, "Data di ingresso", k).date(),
            data_valutazione=_valore(idl, "Data di valutazione", k).date(),
            data_exit=_valore(idl, "Data di exit attesa", k).date(),
            equity_valutazione=float(_valore(idl, "Equity value (input della waterfall)", k)),
            equity_ingresso=float(_valore(idl, "Equity totale investita all'ingresso", k)),
            strumenti_fondo=fondo,
            importo_mip=mip[0],
            n_mip=mip[1],
            prezzo_mip=mip[2],
            parametri=par,
            vol_valutazione=float(_valore(pm, "Volatilità equity scelta – valutazione", k)),
            vol_ingresso=float(_valore(pm, "Volatilità equity scelta – ingresso", k)),
            rf_cont_valutazione=float(_valore(pm, "Risk-free continuo – valutazione", k)),
            rf_cont_ingresso=float(_valore(pm, "Risk-free continuo – ingresso", k)),
            ke_valutazione=float(_valore(pm, "Ke usato nel PWERM – valutazione", k)),
            ke_ingresso=float(_valore(pm, "Ke CAPM – ingresso", k)),
        )
    return casi


def quota_V8(percorso: Path = FILE_INPUT) -> float:
    ws = apri(percorso)["Waterfall_B"]
    for r in range(1, ws.max_row + 1):
        if str(ws.cell(r, COLONNA_ETICHETTE).value or "").startswith("V8 – quota unica a parità"):
            return float(ws.cell(r, 5).value)
    raise InputMancante("Quota V8 non trovata nel foglio Waterfall_B")
