# Percorsi e tolleranze.
from pathlib import Path

CARTELLA_PROGETTO = Path(__file__).resolve().parents[1]
CARTELLA_DATI = CARTELLA_PROGETTO / "dati"
CARTELLA_RISULTATI = CARTELLA_PROGETTO / "risultati"

FILE_INPUT = CARTELLA_DATI / "Case_Study_MIP.xlsx"

CASI = ("A", "B", "C")

TOLL_WATERFALL = 1e-9
TOLL_SOGLIE = 1e-9
TOLL_PREZZI = 0.005

FONT_GRAFICI = "Times New Roman"
