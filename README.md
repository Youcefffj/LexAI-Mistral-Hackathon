# LexIA -- Assistant Juridique Francais

**Mistral Hack-a-ton 2026** · Track Fine-tuning (W&B)

---

## Qu'est-ce que LexIA ?

LexIA est un assistant juridique qui aide les citoyens francophones a comprendre leurs droits.
Il analyse des situations concretes (licenciement, logement, famille, etc.), cite les
articles de loi applicables et propose des pistes d'action.

Le projet a ete construit dans le cadre du **Mistral AI Hackathon 2026** en partant de zero :
collecte de donnees legales ouvertes, fine-tuning d'un LLM, et deploiement d'une interface web.

---

## Tester LexIA

### En ligne (recommande)

L'application est deployee sur Hugging Face Spaces, aucune installation requise :

> **https://huggingface.co/spaces/Youcefffj/lexia**

### En local

```bash
git clone https://github.com/Youcefffj/LexAI-Mistral-Hackathon
cd LexAI-Mistral-Hackathon
pip install -r requirements.txt
```

Creer un fichier `.env` a la racine du repo :

```
MISTRAL_API_KEY=votre_cle_mistral
```

Puis lancer :

```bash
python3 app/app.py
```

L'interface s'ouvre sur `http://localhost:7860`.

---

## Demarche de developpement

Le projet a suivi 4 etapes, chacune materalisee par un script dans `scripts/` :

| Etape | Script | Description |
|-------|--------|-------------|
| 1 | `1_fetch_legifrance.py` | Collecte d'articles de loi depuis le dataset Legifrance (code civil, penal, travail, assurances) |
| 2 | `2_fetch_judilibre.py` | Collecte de decisions de justice anonymisees depuis Judilibre (Cour de Cassation) |
| 3 | `3_prepare_dataset.py` | Nettoyage, mise en forme instruction/reponse, split train/eval, push sur le Hub |
| 4 | `4_train.py` | Fine-tuning LoRA 4-bit sur Ministral 8B avec TRL SFTTrainer, lance via HF Jobs |

L'interface Gradio (`app/app.py`) appelle ensuite l'API Mistral et structure les reponses
en markdown avec citations d'articles, badge de domaine detecte, et historique de conversation.

---

## Donnees

| Source | Type | Volume |
|--------|------|--------|
| Legifrance (`erdal/legifrance`) | Articles de loi officiels | ~1 500 articles |
| Judilibre | Decisions de justice anonymisees | variable |
| Exemples synthetiques | Cas-types rediges manuellement | ~10 exemples |

Dataset publie : [Youcefffj/lexia-french-legal](https://huggingface.co/datasets/Youcefffj/lexia-french-legal)

---

## Resultats du fine-tuning

| Metrique | Valeur |
|----------|--------|
| Modele de base | Ministral 8B Instruct |
| Methode | LoRA r=16, 4-bit (QLoRA) |
| Epochs | 3 |
| Train loss final | 0.5066 |
| Eval loss final | 0.4449 |
| Temps d'entrainement | ~2 h 20 |
| GPU | A10G (Hugging Face Jobs) |

Run W&B : [youcefffj-mistralhack](https://wandb.ai/youcefffj-mistralhack/youcefffj-mistralhack)

---

## Stack technique

- **Modele** : Ministral 8B Instruct, fine-tune via TRL SFTTrainer + LoRA 4-bit
- **Compute** : Hugging Face Jobs (A10G)
- **Tracking** : Weights & Biases
- **Interface** : Gradio (dark theme, rendu markdown, chat avec historique)
- **API** : SDK `mistralai`
- **Deploiement** : Hugging Face Spaces

---

## Liens

| Ressource | URL |
|-----------|-----|
| Demo live | https://huggingface.co/spaces/Youcefffj/lexia |
| Modele | https://huggingface.co/mistral-hackaton-2026/lexia-ministral-8b |
| Dataset | https://huggingface.co/datasets/Youcefffj/lexia-french-legal |
| W&B | https://wandb.ai/youcefffj-mistralhack/youcefffj-mistralhack |

---

## Structure du repo

```
app/app.py               Interface Gradio (point d'entree)
scripts/
  1_fetch_legifrance.py   Collecte articles de loi
  2_fetch_judilibre.py    Collecte jurisprudence
  3_prepare_dataset.py    Preparation du dataset
  4_train.py              Lancement du fine-tuning
data/                     Donnees brutes et traitees
config/                   Configuration d'entrainement
requirements.txt          Dependances Python
```

---

## Disclaimer

LexIA est un outil d'aide a la recherche juridique.
Les informations fournies sont indicatives et ne remplacent en aucun cas les conseils d'un avocat.
