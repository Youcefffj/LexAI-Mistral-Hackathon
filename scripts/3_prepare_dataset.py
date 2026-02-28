"""
=============================================================================
LEXIA — Script 3 : Préparation et formatage du dataset pour TRL SFT
=============================================================================
Ce script transforme les données brutes (articles Légifrance + décisions
Judilibre) en paires instruction/réponse au format ChatML.

Le format ChatML est le standard utilisé par TRL pour le fine-tuning :
  - system : personnalité et instructions permanentes du modèle
  - user   : la question / le cas juridique
  - assistant : la réponse attendue du modèle fine-tuné

ENTRÉE  : Fichiers JSON dans data/legifrance/ et data/judilibre/
SORTIE  : Dataset HF poussé sur le Hub + copie locale dans data/processed/
=============================================================================
"""

import os
import json
import random
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

HF_TOKEN     = os.getenv("HF_TOKEN")
HF_USERNAME  = os.getenv("HF_USERNAME", "Youcefffj")
DATASET_NAME = os.getenv("HF_DATASET_NAME", "lexia-french-legal")

DATA_DIR   = Path("data")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Graine aléatoire pour la reproductibilité
random.seed(42)

# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────

# Ce prompt sera injecté au début de chaque exemple d'entraînement.
# Il définit la personnalité et les capacités de LexIA.
SYSTEM_PROMPT = """Tu es LexIA, un assistant juridique expert en droit français.
Tu aides les avocats et juristes à :
- Analyser des cas juridiques et identifier les articles de loi applicables
- Générer des arguments de défense et d'accusation solides
- Résumer des jugements complexes en termes accessibles
- Répondre à des questions sur le Code civil, pénal, du travail, etc.

Tes réponses respectent ces règles :
1. Toujours citer les articles de loi avec leur numéro exact et leur code source
2. Rédiger en français juridique rigoureux et précis
3. Structurer la réponse avec des sections claires
4. Ne jamais inventer d'articles inexistants
5. Indiquer si un cas nécessite une expertise complémentaire"""

# Variations de formulations pour diversifier le dataset
# Cela aide le modèle à généraliser à différentes façons de poser la même question
TEMPLATES_ANALYSE = [
    "Analyse ce cas juridique et identifie les lois françaises applicables : {cas}",
    "En tant qu'avocat, quels articles de loi s'appliquent à cette situation : {cas}",
    "Quels sont les fondements juridiques pertinents pour ce cas ? {cas}",
    "Réalise une analyse juridique de la situation suivante : {cas}",
]

TEMPLATES_DEFENSE = [
    "Génère les arguments de défense pour ce cas : {cas}",
    "Quels arguments juridiques peut invoquer la défense dans ce cas ? {cas}",
    "Liste les moyens de défense disponibles pour : {cas}",
]

TEMPLATES_ACCUSATION = [
    "Quels arguments peut utiliser le demandeur dans ce cas ? {cas}",
    "Développe les arguments pour la partie demanderesse : {cas}",
    "Construis le dossier de la partie civile pour : {cas}",
]

TEMPLATES_RESUME = [
    "Résume ce jugement en termes simples : {texte}",
    "Explique cette décision de justice à un client non-juriste : {texte}",
    "Quels sont les points clés de cette décision ? {texte}",
]

# ─── CONSTRUCTEUR D'EXEMPLES ──────────────────────────────────────────────────

def construire_exemple(instruction, reponse):
    """
    Construire un exemple d'entraînement au format ChatML.
    Ce format est directement compatible avec TRL SFTTrainer.

    Args:
        instruction: La question ou le cas juridique (rôle "user")
        reponse: La réponse juridique attendue (rôle "assistant")

    Returns:
        Dictionnaire avec la structure de messages ChatML
    """
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": instruction},
            {"role": "assistant", "content": reponse},
        ]
    }

# ─── FORMATAGE DES ARTICLES LÉGIFRANCE ───────────────────────────────────────

def formater_articles_legifrance(articles):
    """
    Transformer les articles de loi bruts en exemples d'entraînement.
    Crée deux types d'exemples par article :
    1. Question sur un article spécifique → explication de l'article
    2. Recherche par domaine → citation de l'article pertinent
    """
    exemples = []

    for art in articles:
        texte  = art.get("texte", "").strip()
        numero = art.get("numero", "")
        titre  = art.get("titre", "")
        query  = art.get("query_origine", "")
        code   = art.get("code_source", "")

        # Ignorer les articles trop courts
        if not texte or len(texte) < 50:
            continue

        # Nom lisible du code source
        nom_code = code.replace("-", " ").replace("code ", "Code ").title() if code else "droit français"

        # Type 1 : Question sur un article spécifique avec son contexte d'usage
        if numero:
            instruction = f"Explique l'article {numero} du {nom_code} et son application pratique."
            reponse = (
                f"## Article {numero} — {nom_code}\n\n"
                f"{titre}\n\n"
                f"**Texte de l'article :**\n{texte}\n\n"
                f"**Application pratique :**\n"
                f"Cet article définit le cadre légal applicable et les droits/obligations des parties."
            )
            exemples.append(construire_exemple(instruction, reponse))

        # Type 2 : Recherche par domaine juridique
        if numero and titre:
            instruction = f"Quels textes de loi s'appliquent en matière de '{titre}' ?"
            reponse = (
                f"En matière de **{titre}**, l'article {numero} du {nom_code} est fondamental :\n\n"
                f"{texte}\n\n"
                f"Cet article doit être invoqué lors de toute procédure relative à ce domaine."
            )
            exemples.append(construire_exemple(instruction, reponse))

    return exemples


def formater_decisions_judilibre(decisions):
    """
    Transformer les décisions de justice en exemples d'entraînement.
    Crée deux types d'exemples par décision :
    1. Résumé de la décision dans un langage accessible
    2. Analyse des articles de loi appliqués dans la décision
    """
    exemples = []

    for dec in decisions:
        texte           = dec.get("texte_complet", "").strip()
        sommaire        = dec.get("sommaire", "").strip()
        solution        = dec.get("solution", "")
        chambre         = dec.get("chambre", "")
        textes_appliques = dec.get("textes_appliques", [])
        query           = dec.get("query_origine", "situation juridique")

        if not texte or len(texte) < 100:
            continue

        # Type 1 : Résumé de jugement
        if sommaire:
            template  = random.choice(TEMPLATES_RESUME)
            instruction = template.format(texte=texte[:500])
            reponse = (
                f"## Résumé de la décision\n\n"
                f"{sommaire}\n\n"
                f"**Solution retenue :** {solution}\n\n"
                f"**Chambre :** {chambre}"
            )
            exemples.append(construire_exemple(instruction, reponse))

        # Type 2 : Analyse avec articles appliqués
        if textes_appliques:
            template  = random.choice(TEMPLATES_ANALYSE)
            instruction = template.format(cas=query)
            lois_str  = "\n".join([f"- {t}" for t in textes_appliques[:5]])
            reponse = (
                f"## Articles de loi applicables\n\n"
                f"{lois_str}\n\n"
                f"## Jurisprudence pertinente\n\n"
                f"La Cour de Cassation a jugé dans une affaire similaire :\n\n"
                f"{sommaire or texte[:400]}\n\n"
                f"**Solution :** {solution}"
            )
            exemples.append(construire_exemple(instruction, reponse))

    return exemples


def formater_exemples_synthetiques(exemples):
    """
    Transformer les exemples synthétiques en exemples d'entraînement.
    Ces exemples couvrent les 4 modes de LexIA : analyse, défense, accusation, résumé.
    """
    samples = []

    for ex in exemples:
        cas              = ex.get("cas", "")
        articles         = ex.get("articles_applicables", [])
        analyse          = ex.get("analyse", "")
        args_defense     = ex.get("arguments_defense", "")
        solution         = ex.get("solution", "")

        if not cas or not articles:
            continue

        articles_str = "\n".join([f"- {a}" for a in articles])

        # Exemple 1 : Analyse complète (mode principal)
        instruction = random.choice(TEMPLATES_ANALYSE).format(cas=cas)
        reponse = (
            f"## Articles applicables\n\n{articles_str}\n\n"
            f"## Analyse juridique\n\n{analyse}\n\n"
            f"## Solution recommandée\n\n{solution}"
        )
        samples.append(construire_exemple(instruction, reponse))

        # Exemple 2 : Arguments de défense
        if args_defense:
            instruction = random.choice(TEMPLATES_DEFENSE).format(cas=cas)
            reponse = (
                f"## Fondements légaux\n\n{articles_str}\n\n"
                f"## Arguments de défense\n\n{args_defense}"
            )
            samples.append(construire_exemple(instruction, reponse))

        # Exemple 3 : Arguments de la partie adverse
        instruction = random.choice(TEMPLATES_ACCUSATION).format(cas=cas)
        reponse = (
            f"## Base légale de l'action\n\n{articles_str}\n\n"
            f"## Arguments du demandeur\n\n{analyse}"
        )
        samples.append(construire_exemple(instruction, reponse))

    return samples

# ─── CHARGEMENT DES DONNÉES BRUTES ───────────────────────────────────────────

def charger_donnees_brutes():
    """
    Charger toutes les données brutes disponibles dans le dossier data/.
    Cherche les fichiers dans l'ordre : API > fallback HF > synthétique.
    Affiche un résumé de ce qui a été trouvé.
    """
    articles   = []
    decisions  = []
    synthetiques = []

    # Chercher les articles Légifrance (API ou fallback)
    for fichier in ["data/legifrance/articles_legifrance.json", "data/legifrance/articles_hf_fallback.json"]:
        if Path(fichier).exists():
            with open(fichier) as f:
                articles = json.load(f)
            print(f"✅ Articles chargés depuis : {fichier} ({len(articles)} articles)")
            break

    # Chercher les décisions Judilibre
    if Path("data/judilibre/decisions_judilibre.json").exists():
        with open("data/judilibre/decisions_judilibre.json") as f:
            decisions = json.load(f)
        print(f"✅ Décisions chargées : {len(decisions)}")

    # Chercher les exemples synthétiques
    if Path("data/judilibre/decisions_synthetic.json").exists():
        with open("data/judilibre/decisions_synthetic.json") as f:
            synthetiques = json.load(f)
        print(f"✅ Exemples synthétiques chargés : {len(synthetiques)}")

    return articles, decisions, synthetiques

# ─── PIPELINE PRINCIPAL ───────────────────────────────────────────────────────

def preparer_dataset():
    """
    Pipeline complet de préparation du dataset :
    1. Charger les données brutes
    2. Formater en exemples ChatML
    3. Mélanger et diviser en train/eval
    4. Sauvegarder localement
    5. Pousser sur HF Hub
    """
    print("=" * 60)
    print("⚖️  LEXIA — Préparation du dataset pour TRL SFT")
    print("=" * 60)

    print("\n📂 Chargement des données brutes...")
    articles, decisions, synthetiques = charger_donnees_brutes()

    print("\n🔄 Formatage en exemples d'entraînement...")
    exemples_legi   = formater_articles_legifrance(articles)
    exemples_judi   = formater_decisions_judilibre(decisions)
    exemples_synth  = formater_exemples_synthetiques(synthetiques)

    tous_les_exemples = exemples_legi + exemples_judi + exemples_synth

    print(f"\n📊 Résumé du dataset :")
    print(f"   Légifrance   : {len(exemples_legi)} exemples")
    print(f"   Judilibre    : {len(exemples_judi)} exemples")
    print(f"   Synthétiques : {len(exemples_synth)} exemples")
    print(f"   TOTAL        : {len(tous_les_exemples)} exemples")

    if len(tous_les_exemples) == 0:
        print("\n❌ Aucune donnée ! Lance d'abord les scripts 1 et 2.")
        return None

    # Mélanger pour éviter les biais d'ordre dans l'entraînement
    random.shuffle(tous_les_exemples)

    # Diviser en 90% train / 10% évaluation
    idx_split  = int(len(tous_les_exemples) * 0.9)
    train_data = tous_les_exemples[:idx_split]
    eval_data  = tous_les_exemples[idx_split:]
    print(f"\n✂️  Split : {len(train_data)} train / {len(eval_data)} eval")

    # Créer le dataset au format Hugging Face
    from datasets import Dataset, DatasetDict

    dataset = DatasetDict({
        "train": Dataset.from_dict({
            "messages": [ex["messages"] for ex in train_data]
        }),
        "eval": Dataset.from_dict({
            "messages": [ex["messages"] for ex in eval_data]
        }),
    })

    # Sauvegarder localement (pour usage offline si besoin)
    chemin_local = OUTPUT_DIR / "lexia_dataset"
    dataset.save_to_disk(str(chemin_local))
    print(f"✅ Dataset sauvegardé localement → {chemin_local}")

    # Sauvegarder aussi en JSONL pour inspection manuelle
    with open(OUTPUT_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"✅ Fichier JSONL créé → {OUTPUT_DIR}/train.jsonl")

    # Pousser sur HF Hub
    if HF_TOKEN and HF_TOKEN != "xxxx":
        from huggingface_hub import login
        login(token=HF_TOKEN)

        hub_name = f"{HF_USERNAME}/{DATASET_NAME}"
        print(f"\n🚀 Push sur HF Hub : {hub_name}...")
        dataset.push_to_hub(hub_name, private=False)
        print(f"✅ Dataset disponible : https://huggingface.co/datasets/{hub_name}")
    else:
        print("\n⚠️  HF_TOKEN manquant — dataset uniquement en local")

    return dataset


if __name__ == "__main__":
    preparer_dataset()
