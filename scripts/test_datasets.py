"""
=============================================================================
LEXIA — Test des datasets juridiques disponibles sur Hugging Face
=============================================================================
Ce script teste quels datasets juridiques francais sont accessibles sur le
Hub Hugging Face. Les resultats determinent quelles sources de donnees
seront utilisees pour enrichir le dataset v2.

SORTIE : Liste des datasets accessibles avec leurs colonnes
=============================================================================
"""

from datasets import load_dataset

# Liste des datasets juridiques francais a tester
datasets_a_tester = [
    ("antoinejeannot/jurisprudence", {"data_files": "cour_de_cassation.parquet", "split": "train[:20]"}),
    ("louisbrulenaudet/legalkit", {"split": "train[:20]"}),
    ("harvard-lil/cold-french-law", {"split": "train[:20]"}),
    ("maastrichtlawtech/bsard", {"split": "train[:20]"}),
]

disponibles = []

print("=" * 60)
print("Test des datasets juridiques francais sur HF Hub")
print("=" * 60)

for nom, kwargs in datasets_a_tester:
    try:
        print(f"\nTest : {nom}...")
        ds = load_dataset(nom, **kwargs)
        colonnes = ds.column_names
        print(f"  OK — {nom}")
        print(f"  Colonnes : {colonnes}")
        # Afficher un apercu du premier exemple
        if len(ds) > 0:
            exemple = ds[0]
            for col in colonnes[:3]:
                val = str(exemple.get(col, ""))[:120]
                print(f"  {col} : {val}")
        disponibles.append((nom, colonnes))
    except Exception as e:
        print(f"  ERREUR — {nom} : {e}")

print(f"\n{'=' * 60}")
print(f"Resultat : {len(disponibles)} datasets disponibles sur {len(datasets_a_tester)}")
for nom, cols in disponibles:
    print(f"  - {nom} ({', '.join(cols[:5])})")
