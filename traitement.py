
"""Traitement et sauvegarde "en dur" du jeu de données ColBERT_Humor_Detection.

Ce script charge le CSV, nettoie les textes, crée une colonne label (0/1),
enregistre la version nettoyée en Parquet et Pickle, et peut aussi
calculer et sauvegarder un TF-IDF + vecteur (sparse) pour entraînement ML.

Usage exemple:
	python traitement.py --input datasets/ColBERT_Humor_Detection.csv --outdir data/processed

Contract (inputs/outputs):
 - input: chemin vers un CSV contenant au moins les colonnes `text` et `humor`
 - outputs: dossier de sortie contenant:
	 - cleaned parquet/pickle du DataFrame
	 - (optionnel) tfidf vectorizer (joblib) et matrice sparse (.npz)

Principaux cas gérés:
 - suppression des lignes vides
 - conversion du label `humor` en int (True->1, False->0)
 - gestion prudente des types et encodages
"""

from pathlib import Path
import re
import json
import argparse
import pandas as pd

def preprocess_text(s: str) -> str:
	"""Nettoyage simple: lowercase, supprime URLs, balises HTML, ponctuation
	excessive et espace duplicate."""
	if s is None:
		return ""
	text = str(s)
	# lowercase
	text = text.lower()
	# remove urls
	text = re.sub(r"https?://\S+|www\.\S+", " ", text)
	# remove html tags
	text = re.sub(r"<[^>]+>", " ", text)
	# replace non-word characters (keep basic punctuation) with space
	text = re.sub(r"[^\w\s'!-]", " ", text)
	# collapse whitespace
	text = re.sub(r"\s+", " ", text).strip()
	return text


def load_and_clean(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)

	# Expect columns `text` and `humor`.
	if "text" not in df.columns:
		raise ValueError("Le fichier doit contenir une colonne 'text'.")

	# Normalize label
	if "humor" in df.columns:
		# try to coerce booleans/strings to int 0/1
		try:
			df["label"] = df["humor"].astype(bool).astype(int)
		except Exception:
			# fallback: map common values
			df["label"] = df["humor"].map({"True": 1, "False": 0, "true": 1, "false": 0})
	else:
		df["label"] = pd.NA

	# Clean text
	df["text_clean"] = df["text"].astype(str).apply(preprocess_text)

	# drop rows with no text
	df = df[df["text_clean"].str.strip().astype(bool)].copy()

	# reset index
	df.reset_index(drop=True, inplace=True)
	return df


def save_processed(df: pd.DataFrame, outdir: Path, name: str = "colbert_humor", save_csv: bool = False) -> dict:
	outdir.mkdir(parents=True, exist_ok=True)
	parquet_path = outdir / f"{name}.parquet"
	pickle_path = outdir / f"{name}.pkl"

	# Save parquet (fast to load later)
	df.to_parquet(parquet_path, index=False)
	# Also save a pickle (complete DataFrame)
	df.to_pickle(pickle_path)

	meta = {
		"n_rows": int(len(df)),
		"parquet": str(parquet_path),
		"pickle": str(pickle_path),
	}
	if save_csv:
		csv_path = outdir / f"{name}.csv"
		# Save as UTF-8, without index
		df.to_csv(csv_path, index=False, encoding="utf-8")
		meta["csv"] = str(csv_path)
	# store metadata
	(outdir / f"{name}_meta.json").write_text(json.dumps(meta, indent=2))
	return meta


def build_and_save_tfidf(df: pd.DataFrame, outdir: Path, name: str = "colbert_humor", max_features: int = 20000):
	# Lazy import to avoid forcing dependency if user doesn't want vectorization
	from sklearn.feature_extraction.text import TfidfVectorizer
	import joblib
	from scipy import sparse

	vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words='english')
	X = vec.fit_transform(df["text_clean"].astype(str))

	outdir.mkdir(parents=True, exist_ok=True)
	vec_path = outdir / f"{name}_tfidf.joblib"
	mat_path = outdir / f"{name}_tfidf.npz"

	joblib.dump(vec, vec_path)
	sparse.save_npz(mat_path, X)

	return {"vectorizer": str(vec_path), "matrix": str(mat_path), "shape": X.shape}


def main():
	parser = argparse.ArgumentParser(description="Nettoyage et sauvegarde du jeu de données humor detection")
	parser.add_argument("--input", "-i", required=True, help="CSV d'entrée")
	parser.add_argument("--outdir", "-o", default="data/processed", help="Dossier de sortie")
	parser.add_argument("--no-vectorize", action="store_true", help="Ne pas construire TF-IDF (plus rapide) ")
	parser.add_argument("--save-csv", action="store_true", help="Enregistrer aussi la version nettoyée en CSV (UTF-8)")
	parser.add_argument("--name", default="colbert_humor", help="Préfixe de nom de fichier de sortie")
	args = parser.parse_args()

	input_path = Path(args.input)
	outdir = Path(args.outdir)

	print(f"Chargement de: {input_path}")
	df = load_and_clean(input_path)

	print(f"Lignes après nettoyage: {len(df)}")
	meta = save_processed(df, outdir, name=args.name, save_csv=args.save_csv)
	print("Fichiers sauvegardés:")
	print(meta)

	if not args.no_vectorize:
		print("Construction du TF-IDF (cela peut prendre du temps)...")
		vec_meta = build_and_save_tfidf(df, outdir, name=args.name)
		print("TF-IDF sauvegardé:", vec_meta)


if __name__ == "__main__":
	main()
