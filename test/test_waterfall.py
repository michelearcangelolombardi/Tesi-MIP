import numpy as np
import pytest

from mip.config import TOLL_SOGLIE, TOLL_WATERFALL
from mip.input import carica_casi
from mip.verifica import confronta_test_breakpoint, confronta_waterfall
from mip.waterfall import payoff, soglie


@pytest.fixture(scope="module")
def casi():
    return carica_casi()


def test_cap_table_quadra(casi):
    for k, c in casi.items():
        assert abs(c.investimento_fondo + c.importo_mip - c.equity_ingresso) < 1e-9, k


def test_waterfall_uguale_a_excel(casi):
    righe = confronta_waterfall(casi)
    assert len({(r["caso"], r["colonna"]) for r in righe}) == 39
    peggiore = max(righe, key=lambda r: r["differenza"])
    assert peggiore["differenza"] < TOLL_WATERFALL, peggiore


def test_soglie_uguali_a_test_breakpoint(casi):
    righe = confronta_test_breakpoint(casi)
    assert len(righe) == 15
    peggiore = max(righe, key=lambda r: r["differenza"])
    assert peggiore["differenza"] < TOLL_SOGLIE, peggiore


def test_quadratura_payoff(casi):
    E = np.linspace(0, 3000, 3001)[:, None]
    t = np.array([3.0, 4.0, 6.0027, 8.0, 11.0])[None, :]
    for k, c in casi.items():
        p = payoff(c, E, t)
        totale = p["mip"] + p["fondo_shl"] + p["fondo_pref"] + p["fondo_ordinarie"]
        assert np.max(np.abs(totale - E)) < 1e-9, k
        assert np.min(p["mip"]) >= 0 and np.min(p["fondo_ordinarie"]) >= -1e-9, k


def test_catch_up_formula_diretta(casi):
    c = casi["B"]
    E = np.linspace(0, 3000, 30001)
    for t in (4.0, 5.5, 6.0027, 8.0):
        p = payoff(c, E, t)
        H = soglie(c, t)["hurdle"]
        diretta = np.where(E >= H, np.minimum(c.parametri["c"] * (E - H), c.parametri["s"] * (E - c.investimento_fondo)), 0.0)
        assert np.max(np.abs(np.minimum(diretta, E) - p["mip"])) < 1e-9


def test_salto_del_cliff(casi):
    c = casi["A"]
    L = float(soglie(c, 6.0)["ratchet"])
    sotto, sopra = payoff(c, L - 1e-6, 6.0)["mip"], payoff(c, L + 1e-6, 6.0)["mip"]
    atteso = (c.parametri["q_ratchet"] - c.parametri["q_base"]) * (L - float(soglie(c, 6.0)["hurdle"]))
    assert abs((sopra - sotto) - atteso) < 1e-5


def test_prezzi_del_mip_sono_il_punto_fisso(casi):
    from mip.config import TOLL_PREZZI
    from mip.prezzi import prezzo_punto_fisso
    for k, c in casi.items():
        assert abs(prezzo_punto_fisso(c) - c.importo_mip) < TOLL_PREZZI, k


def test_quota_V8(casi):
    from openpyxl import load_workbook
    from mip.config import FILE_INPUT
    from mip.prezzi import quota_unica_V8
    ws = load_workbook(FILE_INPUT, data_only=True)["Waterfall_B"]
    riga = next(r for r in range(1, ws.max_row + 1) if str(ws.cell(r, 3).value).startswith("V8 – quota unica a parità"))
    assert abs(quota_unica_V8(casi["B"]) - ws.cell(riga, 5).value) < 0.00005
