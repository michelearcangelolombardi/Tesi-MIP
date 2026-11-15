# Tesi MIP

Codice della tesi "Model Risk in the Valuation of Management Incentive Plans".

Il file di input `dati/Case_Study_MIP.xlsx` non è incluso.

## Installazione

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Esecuzione

```
python esegui.py
python stesso_deal.py
python analisi_aggiuntive.py
pytest -q
```

`python esegui.py 4` esegue solo il passaggio 4.

## Cartelle

- `mip/`: passaggi 1-10
- `test/`: test
- `risultati/`: output
