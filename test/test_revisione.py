import pytest

from mip.input import carica_casi
from mip.montecarlo import fair_value_vettoriale
from mip.opm import mercato, valore_opm
from mip.sensitivity import prove

ESENTI = {"Definizione di leva"}


@pytest.fixture(scope="module")
def casi():
    return carica_casi()


def test_fair_value_vettoriale_uguale_all_opm(casi):
    for k, c in casi.items():
        m = mercato(c)
        v = float(fair_value_vettoriale(c, [m.E0], m.T, m.r, m.sigma, c.t_exit)[0])
        assert abs(v - valore_opm(c, m)) < 1e-6, k


def test_prove_cambiano_entrambe_le_date(casi):
    for k, c in casi.items():
        for p in prove(c):
            if p.base or p.parametro in ESENTI:
                continue
            x = p.caso
            if x.vol_valutazione != c.vol_valutazione:
                assert x.vol_ingresso != c.vol_ingresso, (k, p.parametro, p.livello)
            if x.rf_cont_valutazione != c.rf_cont_valutazione:
                assert x.rf_cont_ingresso != c.rf_cont_ingresso, (k, p.parametro, p.livello)
            if x.ke_valutazione != c.ke_valutazione:
                assert x.ke_ingresso != c.ke_ingresso, (k, p.parametro, p.livello)
