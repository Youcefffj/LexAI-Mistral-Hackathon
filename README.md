# LexIA -- Assistant Juridique Francais

**Mistral Hack-a-ton 2026** · Track Fine-tuning (W&B)

*[English version below](#english)*

---

## Qu'est-ce que LexIA ?

LexIA est un assistant juridique IA qui aide les citoyens francophones a comprendre leurs droits.
Il analyse des situations concretes (licenciement, logement, famille, consommation, penal...),
cite les articles de loi applicables et propose des pistes d'action concretes.

Le projet a ete construit dans le cadre du **Mistral AI Hackathon 2026** en partant de zero :
collecte de donnees legales ouvertes, fine-tuning d'un LLM, et deploiement d'une interface web.

---

## Tester LexIA

### En ligne (recommande)

L'application est deployee sur Hugging Face Spaces :

> **https://huggingface.co/spaces/Youcefffj/lexia**

### En local

```bash
git clone https://github.com/Youcefffj/LexAI-Mistral-Hackathon
cd LexAI-Mistral-Hackathon
pip install -r requirements.txt
```

Creer un fichier `.env` a la racine :

```
MISTRAL_API_KEY=votre_cle_mistral
```

```bash
python3 app/app.py
```

L'interface s'ouvre sur `http://localhost:7860`.

---

## Demarche de developpement

Le projet a suivi 3 etapes principales :

| Etape | Script | Description |
|-------|--------|-------------|
| 1 | `1_fetch_legifrance.py` | Collecte d'articles de loi depuis Legifrance (code civil, penal, travail, assurances) |
| 2 | `2_fetch_judilibre.py` | Collecte de decisions de justice anonymisees depuis Judilibre (Cour de Cassation) |
| 3 | `3_prepare_dataset.py` | Nettoyage, mise en forme ChatML, split train/eval, push sur le Hub |

L'interface Gradio (`app/app.py`) appelle l'API Mistral et structure les reponses
en markdown avec citations d'articles, badge de domaine detecte, et historique de conversation.

---

## Donnees d'entrainement (v1)

### Dataset v1 -- 3 030 exemples

| Source | Type | Volume |
|--------|------|--------|
| **Legifrance** (`erdal/legifrance`) | Articles de loi officiels | ~1 500 articles |
| **Judilibre** | Decisions de justice anonymisees | ~1 500 decisions |
| **Exemples synthetiques** | Cas-types rediges manuellement | ~30 exemples |

Tous les exemples sont convertis au format **ChatML** (system/user/assistant).

Dataset publie : [Youcefffj/lexia-french-legal](https://huggingface.co/datasets/Youcefffj/lexia-french-legal)

---

## Resultats du fine-tuning (v1)

| Metrique | Valeur |
|----------|--------|
| Modele de base | Ministral 8B Instruct |
| Dataset | 3 030 exemples |
| Methode | LoRA r=16, 4-bit (QLoRA) |
| Epochs | 3 |
| Train loss final | 0.5066 |
| Eval loss final | 0.4449 |
| Temps d'entrainement | ~2 h 20 |
| GPU | A10G (Hugging Face Jobs) |

Run W&B : [youcefffj-mistralhack](https://wandb.ai/youcefffj-mistralhack/youcefffj-mistralhack)

---

## Stack technique

- **Modele** : Ministral 8B Instruct, fine-tune via TRL SFTTrainer + QLoRA 4-bit
- **Compute** : Hugging Face Jobs (A10G, 24 GB VRAM)
- **Tracking** : Weights & Biases
- **Interface** : Gradio (dark theme, rendu markdown, chat multi-tour, 8 exemples)
- **API** : SDK `mistralai`
- **Deploiement** : Hugging Face Spaces

---

## Fonctionnalites de l'interface

- **Analyse de cas** : description d'une situation juridique, analyse structuree avec articles de loi
- **Conversation** : chat multi-tour avec historique complet
- **Resume de jugement** : coller un texte de jugement pour obtenir un resume structure
- **8 exemples cliquables** : licenciement, degats des eaux, caution, pension alimentaire, arnaque en ligne, garde d'enfants, harcelement, vice cache
- **Detection de domaine** : badge automatique (travail, logement, famille, penal, consommation, assurances)
- **Rendu Markdown** : sections structurees, articles de loi en gras, listes d'actions

---

## Liens

| Ressource | URL |
|-----------|-----|
| Demo live | https://huggingface.co/spaces/Youcefffj/lexia |
| Modele v1 | https://huggingface.co/mistral-hackaton-2026/lexia-ministral-8b |
| Dataset v1 | https://huggingface.co/datasets/Youcefffj/lexia-french-legal |
| W&B | https://wandb.ai/youcefffj-mistralhack/youcefffj-mistralhack |

---

## Structure du repo

```
app/app.py                  Interface Gradio (point d'entree)
scripts/
  1_fetch_legifrance.py     Collecte articles de loi (Legifrance)
  2_fetch_judilibre.py      Collecte jurisprudence (Judilibre)
  3_prepare_dataset.py      Preparation du dataset v1
  4_enrich_dataset.py       Enrichissement dataset v2
  5_launch_job_v2.py        Lancement fine-tuning v2
  monitor_job.py            Monitoring du job d'entrainement
  test_datasets.py          Test des datasets HF disponibles
data/                       Donnees brutes et traitees
config/                     Configuration d'entrainement
requirements.txt            Dependances Python
```

---

## V2 -- Enrichissement massif (en cours)

Le fine-tuning v2 est en cours de training. Il s'appuie sur un dataset enrichi de **162 444 exemples**
construit a partir de multiples sources juridiques ouvertes.

### Dataset v2 -- 162 444 exemples

| Source | Type | Volume |
|--------|------|--------|
| **Judilibre** (`antoinejeannot/jurisprudence`) | Decisions de la Cour de Cassation | ~100 000 |
| **LegalKit** (`louisbrulenaudet/legalkit`) | Corpus legislatif francais | ~20 000 |
| **Cold French Law** (`harvard-lil/cold-french-law`) | Textes legislatifs (Harvard) | ~20 000 |
| **BSARD** (`maastrichtlawtech/bsard`) | Q&A juridiques | ~20 000 |
| **LexIA v1** (`Youcefffj/lexia-french-legal`) | Dataset initial + synthetiques | ~3 000 |

Dataset v2 publie : [Youcefffj/lexia-french-legal-v2](https://huggingface.co/datasets/Youcefffj/lexia-french-legal-v2)

### Fine-tuning v2 (en cours)

| Metrique | Valeur |
|----------|--------|
| Modele de base | lexia-ministral-8b (continued training sur v1) |
| Dataset | 162 444 exemples (v2) |
| Methode | LoRA r=16, alpha=32, 4-bit (QLoRA) |
| Epochs | 2 |
| Learning rate | 1e-4, cosine scheduler |
| GPU | A10G Large (Hugging Face Jobs) |

Scripts : `4_enrich_dataset.py` (enrichissement) + `5_launch_job_v2.py` (lancement)

---

## Disclaimer

LexIA est un outil d'aide a la recherche juridique.
Les informations fournies sont indicatives et ne remplacent en aucun cas les conseils d'un avocat.

---
---

# <a name="english"></a> LexIA -- French Legal Assistant (English)

**Mistral Hack-a-ton 2026** · Fine-tuning Track (W&B)

---

## What is LexIA?

LexIA is an AI-powered legal assistant that helps French-speaking citizens understand their rights.
It analyzes real-life situations (employment dismissal, housing, family, consumer, criminal law...),
cites applicable legal articles and suggests concrete courses of action.

The project was built from scratch during the **Mistral AI Hackathon 2026**:
open legal data collection, LLM fine-tuning, and web interface deployment.

---

## Try LexIA

### Online (recommended)

The app is deployed on Hugging Face Spaces:

> **https://huggingface.co/spaces/Youcefffj/lexia**

### Locally

```bash
git clone https://github.com/Youcefffj/LexAI-Mistral-Hackathon
cd LexAI-Mistral-Hackathon
pip install -r requirements.txt
```

Create a `.env` file:

```
MISTRAL_API_KEY=your_mistral_key
```

```bash
python3 app/app.py
```

Opens at `http://localhost:7860`.

---

## Development approach

The project followed 3 main steps:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_fetch_legifrance.py` | Collect legal articles from Legifrance (civil, criminal, labor, insurance codes) |
| 2 | `2_fetch_judilibre.py` | Collect anonymized court decisions from Judilibre (Cour de Cassation) |
| 3 | `3_prepare_dataset.py` | Cleaning, ChatML formatting, train/eval split, push to Hub |

The Gradio interface (`app/app.py`) calls the Mistral API and formats responses
as structured markdown with legal article citations, domain badges, and conversation history.

---

## Training data (v1)

### Dataset v1 -- 3,030 examples

| Source | Type | Volume |
|--------|------|--------|
| **Legifrance** (`erdal/legifrance`) | Official legal articles | ~1,500 articles |
| **Judilibre** | Anonymized court decisions | ~1,500 decisions |
| **Synthetic examples** | Manually written case studies | ~30 examples |

All examples converted to **ChatML** format (system/user/assistant).

Published dataset: [Youcefffj/lexia-french-legal](https://huggingface.co/datasets/Youcefffj/lexia-french-legal)

---

## Fine-tuning results (v1)

| Metric | Value |
|--------|-------|
| Base model | Ministral 8B Instruct |
| Dataset | 3,030 examples |
| Method | LoRA r=16, 4-bit (QLoRA) |
| Epochs | 3 |
| Final train loss | 0.5066 |
| Final eval loss | 0.4449 |
| Training time | ~2h 20min |
| GPU | A10G (Hugging Face Jobs) |

W&B run: [youcefffj-mistralhack](https://wandb.ai/youcefffj-mistralhack/youcefffj-mistralhack)

---

## Tech stack

- **Model**: Ministral 8B Instruct, fine-tuned with TRL SFTTrainer + QLoRA 4-bit
- **Compute**: Hugging Face Jobs (A10G, 24 GB VRAM)
- **Tracking**: Weights & Biases
- **Interface**: Gradio (dark theme, markdown rendering, multi-turn chat, 8 examples)
- **API**: `mistralai` SDK
- **Deployment**: Hugging Face Spaces

---

## Interface features

- **Case analysis**: describe a legal situation, get structured analysis with legal articles
- **Conversation**: multi-turn chat with full history
- **Judgment summary**: paste court judgment text for structured summary
- **8 clickable examples**: wrongful dismissal, water damage, security deposit, alimony, online fraud, child custody, workplace harassment, hidden defects
- **Domain detection**: automatic badge (labor, housing, family, criminal, consumer, insurance law)
- **Markdown rendering**: structured sections, bold legal articles, action lists

---

## Links

| Resource | URL |
|----------|-----|
| Live demo | https://huggingface.co/spaces/Youcefffj/lexia |
| Model v1 | https://huggingface.co/mistral-hackaton-2026/lexia-ministral-8b |
| Dataset v1 | https://huggingface.co/datasets/Youcefffj/lexia-french-legal |
| W&B | https://wandb.ai/youcefffj-mistralhack/youcefffj-mistralhack |

---

## V2 -- Massive data enrichment (in progress)

V2 fine-tuning is currently training. It uses an enriched dataset of **162,444 examples**
built from multiple open legal data sources.

### Dataset v2 -- 162,444 examples

| Source | Type | Volume |
|--------|------|--------|
| **Judilibre** (`antoinejeannot/jurisprudence`) | Cour de Cassation decisions | ~100,000 |
| **LegalKit** (`louisbrulenaudet/legalkit`) | French legislative corpus | ~20,000 |
| **Cold French Law** (`harvard-lil/cold-french-law`) | Legislative texts (Harvard) | ~20,000 |
| **BSARD** (`maastrichtlawtech/bsard`) | Legal Q&A | ~20,000 |
| **LexIA v1** (`Youcefffj/lexia-french-legal`) | Initial dataset + synthetics | ~3,000 |

Published dataset: [Youcefffj/lexia-french-legal-v2](https://huggingface.co/datasets/Youcefffj/lexia-french-legal-v2)

### V2 fine-tuning (in progress)

| Metric | Value |
|--------|-------|
| Base model | lexia-ministral-8b (continued training on v1) |
| Dataset | 162,444 examples (v2) |
| Method | LoRA r=16, alpha=32, 4-bit (QLoRA) |
| Epochs | 2 |
| Learning rate | 1e-4, cosine scheduler |
| GPU | A10G Large (Hugging Face Jobs) |

Scripts: `4_enrich_dataset.py` (enrichment) + `5_launch_job_v2.py` (launch)

---

## Disclaimer

LexIA is a legal research assistance tool.
The information provided is for guidance only and does not replace the advice of a qualified lawyer.
