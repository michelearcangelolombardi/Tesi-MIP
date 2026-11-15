from dataclasses import replace

import numpy as np
import pytest

from mip.input import carica_casi
from mip.lsm import parametri_lsm, regola_obiettivi, simula, valuta


@pytest.fixture(scope="module")
def casi():
    return carica_casi()


@pytest.fixture(scope="module")
def risultati(casi):
    return {k: valuta(c) for k, c in casi.items() if parametri_lsm(c).applicare}


def test_lsm_non_peggiore_delle_exit_fisse(risultati):
    for k, res in risultati.items():
        migliore = max(e.valore_fondo for n, e in res.items() if n.startswith("Exit fissa"))
        assert res["LSM (ottimo del fondo)"].valore_fondo >= migliore * (1 - 0.002), k


def test_lsm_non_banale(casi, risultati):
    for k, res in risultati.items():
        e = res["LSM (ottimo del fondo)"]
        if e.regola == "Regola LSM":
            assert max(e.distribuzione.values()) < 0.95, k
        else:
            ext = valuta(casi[k], finestre_foglio=True)["LSM (ottimo del fondo)"]
            assert ext.regola == "Regola LSM" and max(ext.distribuzione.values()) < 0.95, k


def test_regola_obiettivi_casi_limite(casi):
    c = casi["A"]
    par = parametri_lsm(c)
    p = simula(c, "valutazione", par, 2000, 1)
    impossibile = replace(par, mom_obiettivo=1e6, irr_obiettivo=10.0)
    banale = replace(par, mom_obiettivo=0.0, irr_obiettivo=-1.0)
    assert np.all(regola_obiettivi(c, p, impossibile) == len(p.date) - 1)
    assert np.all(regola_obiettivi(c, p, banale) == 0)


def test_exit_fissa_coerente_con_il_monte_carlo(casi, risultati):
    from mip.montecarlo import valore_mc
    for k, res in risultati.items():
        e = res["Exit fissa 2031"]
        mc = valore_mc(casi[k], controllo=False)
        assert abs(e.valore_mip - mc.valore) < 3 * (e.errore_standard + mc.errore_standard), k
