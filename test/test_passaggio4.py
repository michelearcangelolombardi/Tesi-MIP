import pytest

from mip.alberi import valore_albero, valore_opm_exit_incerta
from mip.input import carica_casi
from mip.montecarlo import valore_mc
from mip.opm import mercato, valore_opm


@pytest.fixture(scope="module")
def casi():
    return carica_casi()


def test_mc_semplice_entro_tre_errori_standard_da_black_scholes(casi):
    for k, c in casi.items():
        bs = valore_opm(c, mercato(c))
        r = valore_mc(c, controllo=False)
        assert abs(r.valore - bs) < 3 * r.errore_standard, (k, r.valore, bs, r.errore_standard)


def test_mc_con_controllo_entro_tre_errori_standard(casi):
    for k, c in casi.items():
        for incerta, atteso in ((False, valore_opm(c, mercato(c))), (True, valore_opm_exit_incerta(c))):
            r = valore_mc(c, exit_incerta=incerta)
            assert abs(r.valore - atteso) < 3 * r.errore_standard + 1e-6, (k, incerta, r.valore, atteso)


def test_mc_con_leaver_coerente_con_l_albero(casi):
    for k, c in casi.items():
        r = valore_mc(c, exit_incerta=True, con_leaver=True)
        alb = valore_albero(c, tipo="binomiale", exit_incerta=True, con_leaver=True)
        assert abs(r.valore - alb) < 3 * r.errore_standard + 0.005 * alb, (k, r.valore, alb)


def test_leaver_con_probabilita_zero_non_cambia_il_mc(casi):
    from dataclasses import replace
    import mip.montecarlo as mc
    c = casi["B"]
    originale = mc.leaver
    mc.leaver = lambda caso: replace(originale(caso), prob_annua=0.0)
    try:
        zero = mc.valore_mc(c, con_leaver=True).valore
    finally:
        mc.leaver = originale
    assert abs(zero - mc.valore_mc(c).valore) < 1e-12


def test_mc_riproducibile(casi):
    c = casi["A"]
    assert valore_mc(c, exit_incerta=True, con_leaver=True).valore == valore_mc(c, exit_incerta=True, con_leaver=True).valore
