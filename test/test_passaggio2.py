import math

import numpy as np
import pytest

from mip.bs import call, digitale
from mip.config import TOLL_PREZZI
from mip.input import apri, carica_casi, quota_V8
from mip.opm import componente_ratchet, mercato, valore_forma_chiusa, valore_integrale, valore_opm
from mip.pwerm import limite_quantili, medie_condizionate, pwerm_prassi, pwerm_quantili
from mip.waterfall import VARIANTI, soglie

DATE = ("valutazione", "ingresso")


@pytest.fixture(scope="module")
def casi():
    return carica_casi()


def test_opm_uguale_alla_formula_chiusa(casi):
    for k, c in casi.items():
        for d in DATE:
            m = mercato(c, d)
            assert abs(valore_opm(c, m) - valore_forma_chiusa(c, m)) < 1e-9, (k, d)


def test_opm_uguale_all_integrazione_numerica(casi):
    q8 = quota_V8()
    for k, c in casi.items():
        m = mercato(c, "valutazione")
        for t in (c.t_exit, 3.0, 11.0, 15.0):
            assert abs(valore_opm(c, m, t) - valore_integrale(c, m, t)) < 1e-8, (k, t)
        for v in VARIANTI[k]:
            if v in (None, "V5"):
                continue
            q = q8 if v == "V8" else None
            assert abs(valore_opm(c, m, variante=v, quota_unica=q) - valore_integrale(c, m, variante=v, quota_unica=q)) < 1e-8, (k, v)


def test_opm_all_ingresso_uguale_al_prezzo(casi):
    for k, c in casi.items():
        assert abs(valore_opm(c, mercato(c, "ingresso")) - c.importo_mip) < TOLL_PREZZI, k


def test_componente_ratchet_formula_chiusa(casi):
    c = casi["A"]
    m = mercato(c, "valutazione")
    sg = soglie(c, c.t_exit)
    H, R = float(sg["hurdle"]), float(sg["ratchet"])
    salto = c.parametri["q_ratchet"] - c.parametri["q_base"]
    atteso = salto * (call(m.E0, R, m.T, m.r, m.sigma) + (R - H) * digitale(m.E0, R, m.T, m.r, m.sigma))
    assert abs(componente_ratchet(c, m) - atteso) < 1e-9


def test_pwerm_prassi_calibrato(casi):
    for k, c in casi.items():
        for d in DATE:
            _, righe = pwerm_prassi(c, d)
            E0 = c.equity_valutazione if d == "valutazione" else c.equity_ingresso
            assert abs(sum(r["probabilita"] for r in righe) - 1) < 1e-12
            assert abs(sum(r["equity_attuale_pesata"] for r in righe) / E0 - 1) < 1e-4, (k, d)
            for r in righe:
                assert abs(r["equity_attuale_pesata"] - r["controllo_excel"]) < 1e-9, (k, d, r["scenario"])


def test_quantili_uguali_al_foglio_excel(casi):
    ws = apri()["PWERM_Quantili"]
    for k, c in casi.items():
        for d in DATE:
            titolo = f"Caso {k} – alla data di {d}"
            r0 = next(r for r in range(1, ws.max_row + 1) if str(ws.cell(r, 3).value or "").startswith(titolo))
            excel = [ws.cell(r0 + 7 + i, 7).value for i in range(5)]
            E0, T, ke, s = ((c.equity_valutazione, c.T, c.ke_valutazione, c.vol_valutazione) if d == "valutazione"
                            else (c.equity_ingresso, c.t_exit, c.ke_ingresso, c.vol_ingresso))
            medie = medie_condizionate(E0, ke, s, T, 5)
            assert np.max(np.abs(medie - excel)) < 1e-9, (k, d)
            assert abs(medie.mean() * (1 + ke) ** (-T) - E0) < 1e-9, (k, d)


def test_quantili_convergono_al_limite(casi):
    for k, c in casi.items():
        assert abs(pwerm_quantili(c, "valutazione", 1000)[0] / limite_quantili(c, "valutazione") - 1) < 1e-3, k
