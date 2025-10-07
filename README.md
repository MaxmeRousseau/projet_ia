# Traitement des données — ColBERT_Humor_Detection

Ce dépôt contient un script `traitement.py` qui nettoie le jeu de données `datasets/ColBERT_Humor_Detection.csv` et sauvegarde une version prête à l'emploi pour des expériences ML.

## Commandes essentielles

1) Installer les dépendances (si nécessaire)

```bash
python3 -m pip install -r requirments.txt
```

2) Lancer le traitement et sauvegarder Parquet + Pickle (par défaut) :

```bash
python3 traitement.py --input datasets/ColBERT_Humor_Detection.csv --outdir data/processed
```

3) Produire aussi un CSV nettoyé (UTF-8) :

```bash
python3 traitement.py --input datasets/ColBERT_Humor_Detection.csv --outdir data/processed --save-csv
```

4) Ne pas construire le TF-IDF (pour aller plus vite) :

```bash
python3 traitement.py --input datasets/ColBERT_Humor_Detection.csv --outdir data/processed --no-vectorize
```

5) Variante : changer le préfixe des fichiers de sortie

```bash
python3 traitement.py --input datasets/ColBERT_Humor_Detection.csv --outdir data/processed --name mon_prefix
```

## Exemples pour recharger les données en Python

- Charger le Parquet (rapide, types conservés) :

```python
import pandas as pd
df = pd.read_parquet('data/processed/colbert_humor.parquet')
print(df.head())
```

- Charger le CSV nettoyé (UTF-8) :

```python
import pandas as pd
df = pd.read_csv('data/processed/colbert_humor.csv', encoding='utf-8')
print(df.head())
```

- Recharger le TF-IDF (si construit) :

```python
import joblib
from scipy import sparse
vec = joblib.load('data/processed/colbert_humor_tfidf.joblib')
X = sparse.load_npz('data/processed/colbert_humor_tfidf.npz')
print(X.shape)
```