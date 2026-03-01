"""
=============================================================================
LEXIA — Script 4 : Enrichissement du dataset pour fine-tuning v2
=============================================================================
Ce script enrichit le dataset existant (Youcefffj/lexia-french-legal) avec
de nouvelles sources juridiques francaises et des exemples synthetiques
de haute qualite.

Sources utilisees :
  1. Dataset existant v1 (Youcefffj/lexia-french-legal)
  2. antoinejeannot/jurisprudence — 553K decisions Cour de Cassation
  3. louisbrulenaudet/legalkit — 53K articles de loi en format Q/A
  4. harvard-lil/cold-french-law — 841K articles de loi complets
  5. maastrichtlawtech/bsard — 22K articles droit belge/francais
  6. Donnees API locales (Legifrance + Judilibre si disponibles)
  7. Exemples synthetiques couvrant 5 domaines juridiques

SORTIE : Dataset v2 pousse sur HF Hub (Youcefffj/lexia-french-legal-v2)
         avec split train (90%) et eval (10%)

OBJECTIF : maximiser le nombre d'exemples de haute qualite
=============================================================================
"""

import os
import json
import random
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset, Dataset, DatasetDict

load_dotenv()

# ── CONFIGURATION ────────────────────────────────────────────────────────────

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_V1 = "Youcefffj/lexia-french-legal"
DATASET_V2 = "Youcefffj/lexia-french-legal-v2"

# System prompt utilise pour tous les exemples ChatML
SYSTEM_PROMPT = (
    "Tu es LexIA, un assistant juridique specialise en droit francais. "
    "Tu aides les citoyens a comprendre leurs droits, analyser des "
    "situations juridiques et comprendre des decisions de justice. "
    "Tu cites les articles de loi pertinents quand c'est possible. "
    "Tu rappelles toujours que tes reponses sont informatives et ne "
    "remplacent pas un avocat."
)

# Nombre maximum d'exemples a charger par source externe
# On maximise pour avoir le plus de donnees possible
MAX_JURISPRUDENCE = 100000
MAX_LEGALKIT = 20000
MAX_COLD_LAW = 20000

random.seed(42)


# ── CHARGEMENT DU DATASET EXISTANT ──────────────────────────────────────────

def charger_dataset_v1():
    """Charger le dataset existant depuis HF Hub."""
    print("Chargement du dataset v1 existant...")
    try:
        ds = load_dataset(DATASET_V1)
        exemples = []
        for split in ds:
            for item in ds[split]:
                exemples.append(item)
        print(f"  {len(exemples)} exemples charges depuis v1")
        return exemples
    except Exception as e:
        print(f"  Erreur chargement v1 : {e}")
        return []


# ── CONVERSION DES SOURCES EXTERNES ────────────────────────────────────────

def convertir_jurisprudence():
    """
    Convertir les decisions de la Cour de Cassation en format ChatML.
    On utilise le texte + summary pour creer des paires question/reponse.
    """
    print("\nChargement de antoinejeannot/jurisprudence...")
    try:
        ds = load_dataset(
            "antoinejeannot/jurisprudence",
            data_files="cour_de_cassation.parquet",
            split="train",
        )
        print(f"  {len(ds)} decisions chargees")
    except Exception as e:
        print(f"  Erreur : {e}")
        return []

    exemples = []
    # Melanger et prendre un echantillon
    indices = list(range(len(ds)))
    random.shuffle(indices)

    for idx in indices[:MAX_JURISPRUDENCE * 2]:
        item = ds[idx]
        texte = item.get("text", "")
        summary = item.get("summary", "")
        solution = item.get("solution", "")
        chambre = item.get("chamber", "")
        date = item.get("decision_date", "")

        # Ignorer les decisions sans texte ou trop courtes
        if not texte or len(texte) < 200:
            continue

        # Extraire les visas (articles de loi cites)
        visas = item.get("visa", [])
        articles_cites = ""
        if visas and isinstance(visas, list):
            titres_visa = [v.get("title", "") for v in visas if isinstance(v, dict) and v.get("title")]
            if titres_visa:
                articles_cites = "Articles cites : " + ", ".join(titres_visa[:5])

        # Tronquer le texte a une taille raisonnable
        texte_court = texte[:2000]

        # Construire la question
        question = (
            f"Analyse cette decision de justice"
            f"{' de la ' + chambre if chambre else ''}"
            f"{' du ' + date if date else ''} :\n\n"
            f"{texte_court}"
        )

        # Construire la reponse a partir du summary et des metadonnees
        parties_reponse = []
        if summary:
            parties_reponse.append(f"## Resume de la decision\n\n{summary}")
        if solution:
            parties_reponse.append(f"## Solution\n\n{solution}")
        if articles_cites:
            parties_reponse.append(f"## References\n\n{articles_cites}")

        # Ajouter un avertissement
        parties_reponse.append(
            "\n*Ces informations ne remplacent pas un avocat. "
            "Consultez un professionnel pour votre situation specifique.*"
        )

        reponse = "\n\n".join(parties_reponse)

        if len(question) > 50 and len(reponse) > 50:
            exemples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": reponse},
                ]
            })

        if len(exemples) >= MAX_JURISPRUDENCE:
            break

    print(f"  {len(exemples)} exemples convertis depuis jurisprudence")
    return exemples


def convertir_legalkit():
    """
    Convertir les articles de legalkit en format ChatML.
    Ce dataset est deja en format input/output, donc la conversion est directe.
    """
    print("\nChargement de louisbrulenaudet/legalkit...")
    try:
        ds = load_dataset("louisbrulenaudet/legalkit", split="train")
        print(f"  {len(ds)} articles charges")
    except Exception as e:
        print(f"  Erreur : {e}")
        return []

    exemples = []
    indices = list(range(len(ds)))
    random.shuffle(indices)

    for idx in indices[:MAX_LEGALKIT * 3]:
        item = ds[idx]
        ref = item.get("input", "")
        contenu = item.get("output", "")
        num = item.get("num", "")

        if not ref or not contenu or len(contenu) < 50:
            continue

        # Creer differents types de questions a partir de la reference
        templates_questions = [
            f"Que dit l'article {ref} ?",
            f"Quel est le contenu de {ref} ?",
            f"Explique-moi {ref} en termes simples.",
            f"Quelles sont les dispositions prevues par {ref} ?",
        ]
        question = random.choice(templates_questions)

        # Construire une reponse structuree
        reponse = f"## {ref}\n\n{contenu}"
        if num:
            reponse += f"\n\n**Reference :** {num}"
        reponse += (
            "\n\n*Ces informations ne remplacent pas un avocat. "
            "Consultez un professionnel pour votre situation specifique.*"
        )

        if len(question) > 50 and len(reponse) > 50:
            exemples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": reponse},
                ]
            })

        if len(exemples) >= MAX_LEGALKIT:
            break

    print(f"  {len(exemples)} exemples convertis depuis legalkit")
    return exemples


def convertir_cold_french_law():
    """
    Convertir les articles de cold-french-law en format ChatML.
    On cree des paires question/reponse a partir du texte des articles.
    """
    print("\nChargement de harvard-lil/cold-french-law...")
    try:
        ds = load_dataset("harvard-lil/cold-french-law", split="train")
        print(f"  {len(ds)} articles charges")
    except Exception as e:
        print(f"  Erreur : {e}")
        return []

    exemples = []
    indices = list(range(len(ds)))
    random.shuffle(indices)

    for idx in indices[:MAX_COLD_LAW * 2]:
        item = ds[idx]
        contenu = item.get("article_contenu_text", "")
        num = item.get("article_num", "")
        titre = item.get("texte_titre_court", "") or item.get("texte_titre", "")
        etat = item.get("article_etat", "")

        # Ignorer les articles trop courts ou abroges
        if not contenu or len(contenu) < 80:
            continue
        if etat and "ABROGE" in str(etat).upper():
            continue

        # Tronquer les articles trop longs
        contenu_court = contenu[:1500]

        # Construire une reference lisible
        ref = ""
        if titre and num:
            ref = f"Article {num} du {titre}"
        elif num:
            ref = f"Article {num}"
        elif titre:
            ref = titre

        if not ref:
            continue

        # Creer la question
        templates = [
            f"Que prevoit {ref} ?",
            f"Quel est le contenu de {ref} ?",
            f"Explique-moi les dispositions de {ref}.",
        ]
        question = random.choice(templates)

        # Construire la reponse
        reponse = f"## {ref}\n\n{contenu_court}"
        reponse += (
            "\n\n*Ces informations ne remplacent pas un avocat. "
            "Consultez un professionnel pour votre situation specifique.*"
        )

        if len(question) > 20 and len(reponse) > 50:
            exemples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": reponse},
                ]
            })

        if len(exemples) >= MAX_COLD_LAW:
            break

    print(f"  {len(exemples)} exemples convertis depuis cold-french-law")
    return exemples


MAX_BSARD = 20000


def convertir_bsard():
    """
    Convertir les articles du BSARD (Belgian/French law corpus) en format ChatML.
    Ce corpus contient des articles de droit belge mais aussi du droit europeen
    et des concepts juridiques transposables au droit francais.
    """
    print("\nChargement de maastrichtlawtech/bsard (corpus)...")
    try:
        ds = load_dataset("maastrichtlawtech/bsard", "corpus", split="corpus")
        print(f"  {len(ds)} articles charges")
    except Exception as e:
        print(f"  Erreur : {e}")
        return []

    exemples = []
    indices = list(range(len(ds)))
    random.shuffle(indices)

    for idx in indices[:MAX_BSARD * 2]:
        item = ds[idx]
        article = item.get("article", "")
        ref = item.get("reference", "")
        code = item.get("code", "")
        description = item.get("description", "")

        if not article or len(article) < 50:
            continue

        contenu_court = article[:1500]

        # Construire la question
        if ref:
            question = f"Que prevoit {ref} ?"
        elif code:
            question = f"Quelles sont les dispositions du {code} ?"
        else:
            continue

        # Construire la reponse
        reponse = f"## {ref or code}\n\n{contenu_court}"
        if description:
            reponse += f"\n\n**Contexte :** {description}"
        reponse += (
            "\n\n*Ces informations ne remplacent pas un avocat. "
            "Consultez un professionnel pour votre situation specifique.*"
        )

        if len(question) > 20 and len(reponse) > 50:
            exemples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": reponse},
                ]
            })

        if len(exemples) >= MAX_BSARD:
            break

    print(f"  {len(exemples)} exemples convertis depuis bsard")
    return exemples


def charger_donnees_api_locales():
    """
    Charger les donnees recuperees par les scripts API (Legifrance + Judilibre)
    si elles existent localement, et les convertir en format ChatML.
    """
    exemples = []

    # Charger les articles Legifrance (API ou fallback HF)
    for fichier in ["data/legifrance/articles_legifrance.json", "data/legifrance/articles_hf_fallback.json"]:
        p = Path(fichier)
        if p.exists():
            print(f"\nChargement de {fichier}...")
            with open(p, encoding="utf-8") as f:
                articles = json.load(f)
            print(f"  {len(articles)} articles trouves")

            for art in articles:
                texte = art.get("texte", art.get("text", ""))
                if not texte or len(texte) < 50:
                    continue

                texte_court = texte[:1500]
                titre = art.get("titre", art.get("numero", "Article de loi"))

                templates = [
                    f"Que dit {titre} ?",
                    f"Explique-moi {titre}.",
                    f"Quelles sont les dispositions de {titre} ?",
                ]
                question = random.choice(templates)

                reponse = f"## {titre}\n\n{texte_court}"
                reponse += (
                    "\n\n*Ces informations ne remplacent pas un avocat. "
                    "Consultez un professionnel pour votre situation specifique.*"
                )

                if len(question) > 20 and len(reponse) > 50:
                    exemples.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": reponse},
                        ]
                    })
            print(f"  {len(exemples)} exemples convertis depuis Legifrance local")
            break

    # Charger les decisions Judilibre
    nb_avant = len(exemples)
    for fichier in ["data/judilibre/decisions_judilibre.json", "data/judilibre/decisions_synthetic.json"]:
        p = Path(fichier)
        if p.exists():
            print(f"\nChargement de {fichier}...")
            with open(p, encoding="utf-8") as f:
                decisions = json.load(f)
            print(f"  {len(decisions)} decisions trouvees")

            for dec in decisions:
                # Format API Judilibre
                texte = dec.get("texte_complet", "")
                sommaire = dec.get("sommaire", "")
                solution = dec.get("solution", "")
                cas = dec.get("cas", "")
                analyse = dec.get("analyse", "")

                if cas and analyse:
                    # Format synthetique
                    question = f"Analyse cette situation juridique : {cas}"
                    reponse_parts = [f"## Analyse\n\n{analyse}"]
                    if dec.get("articles_applicables"):
                        arts = "\n".join(f"- **{a}**" for a in dec["articles_applicables"])
                        reponse_parts.append(f"## Articles applicables\n\n{arts}")
                    if dec.get("solution"):
                        reponse_parts.append(f"## Solution\n\n{dec['solution']}")
                    reponse = "\n\n".join(reponse_parts)
                elif texte:
                    # Format API
                    question = f"Analyse cette decision de justice : {texte[:500]}"
                    reponse_parts = []
                    if sommaire:
                        reponse_parts.append(f"## Resume\n\n{sommaire}")
                    if solution:
                        reponse_parts.append(f"## Solution\n\n{solution}")
                    reponse = "\n\n".join(reponse_parts) if reponse_parts else f"## Decision\n\n{texte[:1500]}"
                else:
                    continue

                reponse += (
                    "\n\n*Ces informations ne remplacent pas un avocat. "
                    "Consultez un professionnel pour votre situation specifique.*"
                )

                if len(question) > 30 and len(reponse) > 50:
                    exemples.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": reponse},
                        ]
                    })
            print(f"  {len(exemples) - nb_avant} exemples convertis depuis Judilibre local")
            break

    return exemples


# ── EXEMPLES SYNTHETIQUES DE HAUTE QUALITE ──────────────────────────────────

def generer_exemples_synthetiques():
    """
    Generer 200 exemples synthetiques couvrant 5 grands domaines du droit
    francais. Chaque exemple est une situation concrete avec une reponse
    detaillee citant les articles de loi applicables.
    """
    print("\nGeneration de 200 exemples synthetiques...")

    exemples_par_domaine = {
        # ── DROIT DU TRAVAIL ────────────────────────────────────────────
        "Droit du travail": [
            {
                "user": "Mon employeur m'a licencie apres 8 ans d'anciennete sans me convoquer a un entretien prealable. Le licenciement est-il valable ?",
                "assistant": "## Analyse de la situation\n\nVotre employeur n'a pas respecte la procedure de licenciement obligatoire. L'absence d'entretien prealable constitue une irregularite de procedure.\n\n## Articles applicables\n\n- **Article L1232-2 du Code du travail** : L'employeur qui envisage de licencier un salarie doit le convoquer a un entretien prealable par lettre recommandee ou remise en main propre.\n- **Article L1232-4 du Code du travail** : Le salarie peut se faire assister lors de l'entretien prealable.\n- **Article L1235-2 du Code du travail** : En cas d'irregularite de procedure, le salarie peut obtenir une indemnite qui ne peut exceder un mois de salaire.\n\n## Prochaines etapes\n\n1. Contestez le licenciement par courrier recommande aupres de votre employeur\n2. Saisissez le conseil de prud'hommes dans un delai de 12 mois\n3. Consultez un avocat en droit du travail pour evaluer le montant des indemnites\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Mon patron me doit 3 mois de salaire impaye. Il me dit qu'il n'a pas de tresorerie. Que puis-je faire ?",
                "assistant": "## Analyse de la situation\n\nLe non-paiement du salaire est une infraction penale. Votre employeur a l'obligation de vous verser votre remuneration a date fixe, quelle que soit sa situation financiere.\n\n## Articles applicables\n\n- **Article L3242-1 du Code du travail** : Le salaire est verse une fois par mois aux salaries mensuels.\n- **Article L3246-1 du Code du travail** : L'action en paiement du salaire se prescrit par 3 ans.\n- **Article R3246-1 du Code du travail** : Le non-paiement du salaire est passible d'une amende.\n\n## Demarches conseillees\n\n- Envoyer une mise en demeure par courrier recommande avec AR\n- En l'absence de reponse sous 8 jours, saisir le conseil de prud'hommes en refere (procedure rapide)\n- Signaler la situation a l'inspection du travail (DREETS)\n- Si l'entreprise est en difficulte, l'AGS (assurance garantie des salaires) peut prendre le relais\n\n## Prochaines etapes\n\n1. Envoyez une mise en demeure de paiement par LRAR\n2. Saisissez le conseil de prud'hommes en refere pour obtenir le paiement rapide\n3. Contactez l'inspection du travail pour signaler l'infraction\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Je suis en CDD depuis 2 ans avec des renouvellements successifs. Mon employeur veut encore renouveler. Est-ce legal ?",
                "assistant": "## Analyse de la situation\n\nLa duree maximale d'un CDD, renouvellements inclus, est en principe de 18 mois. Un CDD de 2 ans avec renouvellements successifs depasse probablement la duree legale maximale.\n\n## Articles applicables\n\n- **Article L1242-8 du Code du travail** : La duree totale du CDD, renouvellements compris, ne peut exceder 18 mois (sauf exceptions).\n- **Article L1243-11 du Code du travail** : Le non-respect du delai de carence entre deux CDD peut entrainer la requalification en CDI.\n- **Article L1245-1 du Code du travail** : Un CDD conclu en violation des regles est repute etre un CDI.\n\n## Consequences\n\n- Vous pouvez demander la **requalification en CDI** devant le conseil de prud'hommes\n- Vous avez droit a une **indemnite de requalification** d'au moins un mois de salaire\n- L'employeur s'expose a des sanctions penales\n\n## Prochaines etapes\n\n1. Rassemblez tous vos contrats CDD et leurs avenants de renouvellement\n2. Saisissez le conseil de prud'hommes pour demander la requalification en CDI\n3. Consultez un avocat specialise en droit du travail\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Mon collegue me harcele moralement depuis des mois. Mon manager est au courant mais ne fait rien. Quels sont mes droits ?",
                "assistant": "## Analyse de la situation\n\nLe harcelement moral est un delit penal et une faute grave en droit du travail. Votre employeur a une obligation de securite et doit agir des qu'il est informe.\n\n## Articles applicables\n\n- **Article L1152-1 du Code du travail** : Aucun salarie ne doit subir des agissements repetes de harcelement moral ayant pour effet une degradation de ses conditions de travail.\n- **Article L1152-4 du Code du travail** : L'employeur prend toutes dispositions necessaires en vue de prevenir les agissements de harcelement moral.\n- **Article 222-33-2 du Code penal** : Le harcelement moral est puni de 2 ans d'emprisonnement et 30 000 euros d'amende.\n\n## Demarches\n\n- Documentez chaque incident (dates, temoins, mails, SMS)\n- Alertez par ecrit votre DRH ou direction par courrier recommande\n- Consultez le medecin du travail\n- Saisissez les representants du personnel ou le CSE\n\n## Prochaines etapes\n\n1. Constituez un dossier de preuves (mails, temoignages ecrits, certificat medical)\n2. Alertez officiellement votre employeur par ecrit recommande\n3. Deposez plainte au commissariat ou a la gendarmerie si les faits sont graves\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "J'ai demissionne il y a 15 jours mais mon employeur refuse de me laisser partir avant la fin de mon preavis de 3 mois. Puis-je partir quand meme ?",
                "assistant": "## Analyse de la situation\n\nLe preavis est une obligation reciproque. Vous devez en principe l'executer, sauf accord de votre employeur pour vous en dispenser.\n\n## Articles applicables\n\n- **Article L1237-1 du Code du travail** : La demission doit respecter le preavis prevu par la convention collective ou le contrat de travail.\n- **Article L1234-1 du Code du travail** : Le preavis est fixe par la loi, la convention collective ou les usages.\n\n## Options possibles\n\n- **Demander une dispense** : Ecrivez a votre employeur pour demander une reduction ou une dispense de preavis. S'il accepte, il vous verse le salaire restant ou non selon l'accord.\n- **Negocier** : Proposez de former votre remplacant en echange d'un depart anticipe.\n- **Attention** : Si vous partez sans autorisation avant la fin du preavis, votre employeur peut vous reclamer des dommages-interets.\n\n## Prochaines etapes\n\n1. Envoyez une demande ecrite de dispense de preavis a votre employeur\n2. Verifiez votre convention collective pour les conditions de preavis\n3. En cas de refus, respectez le preavis ou consultez un avocat\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Mon employeur veut me faire signer une rupture conventionnelle mais je ne suis pas d'accord avec le montant de l'indemnite proposee. Quel est le minimum legal ?",
                "assistant": "## Analyse de la situation\n\nLa rupture conventionnelle est un accord mutuel. Vous n'etes pas oblige d'accepter les conditions proposees. L'indemnite ne peut pas etre inferieure a l'indemnite legale de licenciement.\n\n## Articles applicables\n\n- **Article L1237-13 du Code du travail** : L'indemnite de rupture conventionnelle ne peut pas etre inferieure a l'indemnite legale de licenciement.\n- **Article R1234-2 du Code du travail** : L'indemnite legale est de 1/4 de mois par annee d'anciennete pour les 10 premieres annees, puis 1/3 au-dela.\n- **Article L1237-14 du Code du travail** : Chaque partie dispose d'un delai de retractation de 15 jours calendaires.\n\n## Calcul de l'indemnite minimale\n\n- Moins de 10 ans : **1/4 de mois de salaire brut par annee**\n- Au-dela de 10 ans : **1/3 de mois de salaire brut par annee**\n- La convention collective peut prevoir un montant superieur\n\n## Prochaines etapes\n\n1. Calculez votre indemnite minimale legale en fonction de votre anciennete et salaire\n2. Verifiez votre convention collective pour un eventuel montant superieur\n3. Negociez a la hausse en vous basant sur ces montants planchers\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Mon employeur refuse de me payer mes heures supplementaires. Il dit que je suis au forfait jours. Comment verifier si c'est legal ?",
                "assistant": "## Analyse de la situation\n\nLe forfait jours est un amenagement du temps de travail qui supprime le decompte en heures. Mais sa mise en place est strictement encadree et soumise a plusieurs conditions.\n\n## Articles applicables\n\n- **Article L3121-58 du Code du travail** : Le forfait jours ne concerne que les cadres autonomes ou les salaries dont la duree du travail ne peut etre predeterminee.\n- **Article L3121-63 du Code du travail** : Le forfait jours doit etre prevu par un accord collectif et accepte par le salarie dans son contrat.\n- **Article L3121-64 du Code du travail** : L'accord doit prevoir les modalites de suivi de la charge de travail.\n\n## Conditions de validite du forfait jours\n\n- Un accord collectif d'entreprise ou de branche doit l'autoriser\n- Votre contrat de travail ou un avenant doit le mentionner explicitement\n- Vous devez etre cadre autonome dans l'organisation de votre emploi du temps\n- L'employeur doit organiser un entretien annuel sur la charge de travail\n\n## Prochaines etapes\n\n1. Verifiez si votre contrat mentionne une convention de forfait jours\n2. Demandez a voir l'accord collectif qui autorise ce forfait dans votre entreprise\n3. Si le forfait est invalide, reclamez le paiement de vos heures supplementaires sur 3 ans\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Je suis enceinte et mon employeur vient de me licencier. A-t-il le droit ?",
                "assistant": "## Analyse de la situation\n\nLe licenciement d'une salariee enceinte est interdit sauf dans des cas tres limites. Il s'agit d'une protection absolue prevue par le Code du travail.\n\n## Articles applicables\n\n- **Article L1225-4 du Code du travail** : Aucun employeur ne peut rompre le contrat de travail d'une salariee en etat de grossesse medicalement constate, sauf faute grave non liee a la grossesse ou impossibilite de maintenir le contrat.\n- **Article L1225-5 du Code du travail** : Le licenciement est annule si la salariee envoie un certificat de grossesse dans les 15 jours suivant la notification.\n- **Article L1225-71 du Code du travail** : En cas de licenciement abusif, la salariee a droit a la reintegration ou a des dommages-interets.\n\n## Ce que vous devez faire\n\n- Si votre employeur ne connaissait pas votre grossesse, envoyez-lui un certificat medical par LRAR dans les 15 jours\n- Le licenciement sera automatiquement annule\n- Si votre employeur connaissait votre etat, le licenciement est nul de plein droit\n\n## Prochaines etapes\n\n1. Envoyez immediatement un certificat de grossesse a votre employeur par LRAR\n2. Demandez la nullite du licenciement et votre reintegration\n3. Saisissez le conseil de prud'hommes si votre employeur refuse\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Mon employeur veut modifier mes horaires de travail sans mon accord. Peut-il le faire unilateralement ?",
                "assistant": "## Analyse de la situation\n\nIl faut distinguer entre un simple changement des conditions de travail (que l'employeur peut imposer) et une modification du contrat de travail (qui necessite l'accord du salarie).\n\n## Articles applicables\n\n- **Article L1222-6 du Code du travail** : La modification d'un element essentiel du contrat pour motif economique doit etre proposee par LRAR avec un delai de reflexion d'un mois.\n- **Article 1103 du Code civil** : Les contrats legalement formes tiennent lieu de loi a ceux qui les ont faits.\n\n## Distinction importante\n\n- **Changement des conditions de travail** (pouvoir de direction) : modification mineure des horaires dans la meme plage (ex : 9h-17h au lieu de 8h-16h) — l'employeur peut l'imposer\n- **Modification du contrat** (accord necessaire) : passage de jour a nuit, changement de jours travailles, passage de temps plein a temps partiel\n\n## Prochaines etapes\n\n1. Verifiez si vos horaires sont fixes dans votre contrat de travail\n2. Determinez si le changement est un simple amenagement ou une modification substantielle\n3. En cas de modification du contrat, refusez par ecrit et demandez le maintien de vos conditions\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Je souhaite contester mon licenciement pour faute grave. Quel est le delai et la procedure ?",
                "assistant": "## Analyse de la situation\n\nLe licenciement pour faute grave prive le salarie de l'indemnite de licenciement et du preavis. Vous pouvez le contester devant le conseil de prud'hommes.\n\n## Articles applicables\n\n- **Article L1471-1 du Code du travail** : Le delai de prescription pour contester un licenciement est de 12 mois a compter de la notification.\n- **Article L1234-1 du Code du travail** : La faute grave est celle qui rend impossible le maintien du salarie dans l'entreprise.\n- **Article L1235-1 du Code du travail** : En cas de litige, le juge apprecie la regularite de la procedure et le caractere reel et serieux des motifs.\n\n## Procedure de contestation\n\n- Vous avez **12 mois** a compter de la notification pour saisir les prud'hommes\n- La charge de la preuve de la faute grave incombe a l'employeur\n- Le conseil peut requalifier la faute grave en cause reelle et serieuse, voire en licenciement sans cause\n\n## Prochaines etapes\n\n1. Rassemblez tous les documents (lettre de licenciement, contrat, bulletins de paie)\n2. Saisissez le conseil de prud'hommes dans le delai de 12 mois\n3. Faites-vous assister par un avocat ou un defenseur syndical\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
        ],

        # ── DROIT DU LOGEMENT ──────────────────────────────────────────
        "Droit du logement": [
            {
                "user": "Mon proprietaire refuse de me rendre mon depot de garantie 2 mois apres mon depart. L'appartement etait en bon etat. Que faire ?",
                "assistant": "## Analyse de la situation\n\nLe proprietaire a l'obligation legale de restituer le depot de garantie dans un delai precis apres la remise des cles.\n\n## Articles applicables\n\n- **Article 22 de la loi du 6 juillet 1989** : Le depot de garantie doit etre restitue dans un delai maximal d'un mois si l'etat des lieux de sortie est conforme a celui d'entree, ou deux mois en cas de differences.\n- **Article 22 alinea 5** : A defaut de restitution dans les delais, le depot de garantie est majore de 10% du loyer mensuel pour chaque mois de retard commence.\n\n## Vos droits\n\n- Si l'etat des lieux est conforme : restitution sous **1 mois**\n- Si des degradations : restitution sous **2 mois** avec justificatifs des retenues\n- Au-dela : **penalite de 10% du loyer par mois de retard**\n\n## Prochaines etapes\n\n1. Envoyez une mise en demeure par LRAR rappelant l'article 22 et reclamant la restitution sous 8 jours\n2. Si pas de reponse, saisissez le juge des contentieux de la protection (ex-tribunal d'instance)\n3. Reclamez la penalite de 10% pour chaque mois de retard\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Mon proprietaire veut augmenter mon loyer de 15% au renouvellement du bail. Est-ce legal ?",
                "assistant": "## Analyse de la situation\n\nL'augmentation du loyer en cours de bail ou au renouvellement est strictement encadree par la loi, notamment en zone tendue.\n\n## Articles applicables\n\n- **Article 17-1 de la loi du 6 juillet 1989** : La revision annuelle du loyer ne peut exceder la variation de l'Indice de Reference des Loyers (IRL).\n- **Article 17-2 de la loi du 6 juillet 1989** : Au renouvellement, le loyer ne peut etre reevalue que s'il est manifestement sous-evalue par rapport aux loyers du voisinage.\n- **Decret encadrement des loyers** : En zone tendue (Paris, Lyon, etc.), des plafonds s'appliquent.\n\n## Ce qu'il faut verifier\n\n- Etes-vous en zone tendue ? Si oui, l'encadrement des loyers s'applique\n- Le proprietaire doit justifier la hausse avec des references de loyers comparables\n- Une augmentation de 15% est probablement abusive si elle depasse l'IRL\n\n## Prochaines etapes\n\n1. Verifiez si vous etes en zone tendue sur le site du gouvernement\n2. Demandez par ecrit les justificatifs de la hausse proposee\n3. Contestez aupres de la commission departementale de conciliation (gratuit)\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Mon logement a des problemes d'humidite et de moisissures. Mon proprietaire ne fait rien malgre mes demandes. Quels sont mes recours ?",
                "assistant": "## Analyse de la situation\n\nLe proprietaire a l'obligation de fournir un logement decent et d'effectuer les reparations necessaires, hors menues reparations a la charge du locataire.\n\n## Articles applicables\n\n- **Article 6 de la loi du 6 juillet 1989** : Le bailleur est tenu de remettre au locataire un logement decent ne portant pas atteinte a la securite physique ou a la sante.\n- **Decret n°2002-120 du 30 janvier 2002** : Definit les criteres de decence (ventilation, etancheite, absence d'infiltrations).\n- **Article 1719 du Code civil** : Le bailleur doit entretenir la chose louee en etat de servir a l'usage pour lequel elle a ete louee.\n\n## Demarches\n\n- Signalez les desordres par courrier recommande avec photos\n- Faites constater par un huissier si necessaire\n- Contactez le service d'hygiene de la mairie pour une visite\n- Vous pouvez demander une reduction de loyer ou la consignation du loyer aupres de la CAF\n\n## Prochaines etapes\n\n1. Mettez en demeure le proprietaire par LRAR avec photos des desordres\n2. Contactez le service hygiene de votre mairie pour une inspection\n3. Saisissez le juge des contentieux de la protection pour obtenir les travaux ou une reduction de loyer\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Mon proprietaire veut me mettre dehors car il veut vendre l'appartement. Quels sont mes droits ?",
                "assistant": "## Analyse de la situation\n\nLe proprietaire peut donner conge pour vendre, mais il doit respecter des conditions strictes de forme et de delai, et le locataire beneficie d'un droit de preemption.\n\n## Articles applicables\n\n- **Article 15 de la loi du 6 juillet 1989** : Le conge pour vente doit etre notifie au moins 6 mois avant la fin du bail, par LRAR, acte d'huissier ou remise en main propre.\n- **Article 15-II** : Le locataire beneficie d'un droit de preemption — le proprietaire doit lui proposer la vente en priorite.\n- **Article 15-III** : Le conge est nul s'il n'est pas motive ou ne respecte pas les delais.\n\n## Vos protections\n\n- **Delai de preavis** : 6 mois minimum avant la fin du bail\n- **Droit de preemption** : Vous pouvez acheter le logement au prix propose\n- **Protection renforcee** : Si vous avez plus de 65 ans et des revenus modestes, le proprietaire doit vous proposer un relogement\n- Le conge ne peut prendre effet qu'a la **fin du bail**\n\n## Prochaines etapes\n\n1. Verifiez que le conge a ete delivre dans les formes et delais legaux\n2. Si vous souhaitez acheter, exercez votre droit de preemption dans les 2 premiers mois\n3. Si le conge est irregulier, contestez-le et restez dans les lieux\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Je suis en retard de 2 mois de loyer et mon proprietaire menace de m'expulser. Peut-il le faire tout de suite ?",
                "assistant": "## Analyse de la situation\n\nL'expulsion d'un locataire est une procedure longue et encadree. Le proprietaire ne peut pas vous expulser de force ni changer les serrures.\n\n## Articles applicables\n\n- **Article 24 de la loi du 6 juillet 1989** : En cas d'impayes, le bailleur doit d'abord envoyer un commandement de payer par huissier, laissant un delai de 2 mois.\n- **Article L412-1 du Code des procedures civiles d'execution** : L'expulsion ne peut etre faite qu'en vertu d'une decision de justice.\n- **Article L412-6** : La treve hivernale interdit les expulsions du 1er novembre au 31 mars.\n\n## La procedure legale\n\n1. Commandement de payer par huissier (delai de 2 mois)\n2. Assignation devant le tribunal\n3. Jugement d'expulsion\n4. Commandement de quitter les lieux (delai de 2 mois)\n5. Recours a la force publique si necessaire\n\n## Prochaines etapes\n\n1. Contactez votre CAF ou Action Logement pour une aide d'urgence (FSL)\n2. Saisissez la commission de surendettement de la Banque de France si vous etes en difficulte\n3. Demandez des delais de paiement au juge (jusqu'a 3 ans selon l'article 1343-5 du Code civil)\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Mon voisin fait du bruit excessif tous les soirs. J'ai essaye de lui parler mais rien ne change. Que puis-je faire legalement ?",
                "assistant": "## Analyse de la situation\n\nLes nuisances sonores excessives constituent un trouble anormal de voisinage, sanctionne civilement et penalement.\n\n## Articles applicables\n\n- **Article R1336-5 du Code de la sante publique** : Les bruits de voisinage portant atteinte a la tranquillite du voisinage sont punis d'une amende de 68 euros (contravention de 3eme classe).\n- **Article 1240 du Code civil** : Tout fait de l'homme qui cause un dommage a autrui oblige celui par la faute duquel il est arrive a le reparer.\n- **Article R1337-7 du Code de la sante publique** : L'amende peut atteindre 450 euros en cas de bruit nocturne (22h-7h).\n\n## Demarches progressives\n\n- Tentez une mediation amiable (courrier, conciliateur de justice gratuit)\n- Faites constater les nuisances (mains courantes, temoignages, constat d'huissier)\n- Contactez la mairie qui peut faire intervenir la police municipale\n- En dernier recours, saisissez le tribunal\n\n## Prochaines etapes\n\n1. Constituez un dossier de preuves (journal des nuisances, temoignages, enregistrements)\n2. Saisissez le conciliateur de justice gratuit de votre commune\n3. En cas d'echec, portez plainte et/ou saisissez le tribunal judiciaire\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Le chauffage collectif de mon immeuble ne fonctionne pas depuis 3 semaines en plein hiver. Mon proprietaire dit que ce n'est pas sa responsabilite. Est-ce vrai ?",
                "assistant": "## Analyse de la situation\n\nLe proprietaire est tenu de fournir un logement avec un chauffage en etat de fonctionnement. L'absence de chauffage en hiver rend le logement non decent.\n\n## Articles applicables\n\n- **Article 6 de la loi du 6 juillet 1989** : Le bailleur est tenu de remettre au locataire un logement decent et d'entretenir les locaux en etat de servir.\n- **Decret n°2002-120** : Un logement decent doit comporter une installation de chauffage permettant d'atteindre une temperature d'au moins 18°C.\n- **Article 3 du decret du 14 juin 1969** : En chauffage collectif, la temperature minimale est fixee a 18°C dans les pieces a vivre.\n\n## Responsabilite\n\n- En chauffage collectif, c'est le **syndic de copropriete** qui gere la chaudiere\n- Mais le proprietaire reste responsable envers son locataire de la decence du logement\n- Il doit intervenir aupres du syndic pour faire reparer\n\n## Prochaines etapes\n\n1. Mettez en demeure votre proprietaire par LRAR de faire retablir le chauffage\n2. Contactez le syndic de copropriete directement\n3. Demandez une reduction de loyer proportionnelle a la periode sans chauffage\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Mon bail arrive a echeance et mon proprietaire ne veut pas le renouveler sans donner de raison. Est-ce possible ?",
                "assistant": "## Analyse de la situation\n\nLe bail d'habitation se renouvelle automatiquement par tacite reconduction. Le proprietaire ne peut refuser le renouvellement que pour des motifs precis et limitatifs.\n\n## Articles applicables\n\n- **Article 10 de la loi du 6 juillet 1989** : Le contrat de location est reconduit tacitement pour une duree egale a celle du bail initial.\n- **Article 15 de la loi du 6 juillet 1989** : Le bailleur ne peut donner conge que pour vendre, reprendre pour habiter, ou pour motif legitime et serieux.\n\n## Motifs valables de non-renouvellement\n\n- **Conge pour vente** : Le proprietaire veut vendre le bien (droit de preemption du locataire)\n- **Conge pour reprise** : Il veut habiter le logement lui-meme ou y loger un proche\n- **Motif legitime et serieux** : Retards de loyer repetes, troubles de voisinage\n\n## Prochaines etapes\n\n1. Demandez par ecrit le motif du non-renouvellement\n2. Si aucun motif valable n'est fourni, le conge est nul et le bail se poursuit\n3. Consultez un avocat ou l'ADIL de votre departement pour faire valoir vos droits\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
        ],

        # ── DROIT DE LA FAMILLE ────────────────────────────────────────
        "Droit de la famille": [
            {
                "user": "Mon ex-conjoint ne paie plus la pension alimentaire depuis 3 mois. Quels sont mes recours ?",
                "assistant": "## Analyse de la situation\n\nLe non-paiement de la pension alimentaire pendant plus de 2 mois constitue un delit penal d'abandon de famille.\n\n## Articles applicables\n\n- **Article 227-3 du Code penal** : L'abandon de famille (non-paiement de pension pendant plus de 2 mois) est puni de 2 ans d'emprisonnement et 15 000 euros d'amende.\n- **Article L213-3 du Code des procedures civiles d'execution** : La procedure de paiement direct permet de faire prelever la pension directement sur le salaire du debiteur.\n- **Article 373-2-2 du Code civil** : Le juge aux affaires familiales fixe le montant de la pension alimentaire.\n\n## Recours possibles\n\n- **Paiement direct** : La pension est prelevee directement sur le salaire par l'huissier\n- **Saisie sur salaire** : Par voie d'huissier\n- **Plainte penale** : Deposer plainte pour abandon de famille\n- **CAF/MSA** : L'Agence de Recouvrement des Impayes de Pensions Alimentaires (ARIPA) peut prendre le relais\n\n## Prochaines etapes\n\n1. Contactez l'ARIPA (service public gratuit) au 0 800 22 22 28\n2. Deposez plainte au commissariat pour abandon de famille\n3. Faites appel a un huissier pour une saisie sur salaire ou un paiement direct\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Je veux divorcer mais mon conjoint refuse. Est-ce possible de divorcer sans son accord ?",
                "assistant": "## Analyse de la situation\n\nOui, il est possible de divorcer sans l'accord de votre conjoint. Le droit francais prevoit plusieurs formes de divorce dont certaines ne necessitent pas le consentement mutuel.\n\n## Articles applicables\n\n- **Article 229 du Code civil** : Le divorce peut etre prononce par consentement mutuel, pour acceptation du principe, pour alteration definitive du lien conjugal, ou pour faute.\n- **Article 237 du Code civil** : Le divorce pour alteration definitive du lien conjugal peut etre demande apres 1 an de separation de fait.\n- **Article 242 du Code civil** : Le divorce pour faute est fonde sur des faits constitutifs d'une violation grave des devoirs du mariage.\n\n## Types de divorce sans accord\n\n- **Alteration du lien conjugal** : Apres 1 an de separation effective, vous pouvez demander le divorce unilateralement\n- **Divorce pour faute** : Si votre conjoint a commis des fautes graves (violence, adultere, etc.)\n- **Acceptation du principe** : Si votre conjoint accepte le principe du divorce mais pas les modalites\n\n## Prochaines etapes\n\n1. Consultez un avocat (obligatoire en procedure de divorce contentieux)\n2. Determinez le type de divorce le plus adapte a votre situation\n3. L'avocat saisira le juge aux affaires familiales\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Comment fonctionne la garde alternee en France ? Quelles sont les conditions ?",
                "assistant": "## Analyse de la situation\n\nLa residence alternee (garde alternee) permet a l'enfant de vivre alternativement chez chacun de ses parents. Elle est encouragee par la loi mais soumise a l'interet superieur de l'enfant.\n\n## Articles applicables\n\n- **Article 373-2-9 du Code civil** : Le juge peut ordonner la residence alternee. Il peut la fixer a titre provisoire pour une duree determinee.\n- **Article 373-2-11 du Code civil** : Le juge prend en compte la pratique anterieurement suivie, les sentiments exprimes par l'enfant, l'aptitude de chaque parent.\n- **Article 373-2-6 du Code civil** : Le juge statue selon l'interet de l'enfant.\n\n## Criteres evalues par le juge\n\n- Proximite geographique des domiciles des parents\n- Disponibilite de chaque parent\n- Age de l'enfant (rarement avant 3 ans)\n- Capacite des parents a communiquer\n- Voeu de l'enfant s'il est en age de s'exprimer\n- Stabilite du cadre de vie\n\n## Consequences pratiques\n\n- La pension alimentaire peut etre supprimee ou reduite\n- Les allocations familiales sont partagees\n- Chaque parent est responsable des frais pendant sa periode de garde\n\n## Prochaines etapes\n\n1. Proposez un planning de garde alternee realiste a l'autre parent\n2. En cas d'accord, faites homologuer la convention par le juge aux affaires familiales\n3. En cas de desaccord, saisissez le JAF qui tranchera\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Mon pere est decede sans testament. Comment se passe la succession ? Je suis enfant unique et ma mere est encore vivante.",
                "assistant": "## Analyse de la situation\n\nEn l'absence de testament, la succession est regie par les regles legales. En tant qu'enfant unique avec votre mere survivante, les regles sont claires.\n\n## Articles applicables\n\n- **Article 757 du Code civil** : Lorsque le defunt laisse des enfants, le conjoint survivant recueille a son choix l'usufruit de la totalite des biens ou la propriete du quart en pleine propriete.\n- **Article 758 du Code civil** : Le conjoint survivant peut opter dans les 3 mois suivant le deces.\n- **Article 912 du Code civil** : Les enfants ont une reserve hereditaire dont ils ne peuvent etre prives.\n\n## Repartition probable\n\n- **Option 1** : Votre mere choisit 1/4 en pleine propriete → Vous recevez 3/4\n- **Option 2** : Votre mere choisit l'usufruit de la totalite → Vous etes nu-proprietaire de tout (vous recupererez la pleine propriete a son deces)\n\n## Prochaines etapes\n\n1. Contactez un notaire pour ouvrir la succession\n2. Votre mere devra exprimer son choix (usufruit ou propriete) dans les 3 mois\n3. Le notaire procedera au partage des biens\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Je souhaite adopter l'enfant de mon conjoint. Quelle est la procedure en France ?",
                "assistant": "## Analyse de la situation\n\nL'adoption de l'enfant du conjoint (adoption intrafamiliale) est la forme d'adoption la plus frequente en France. Elle peut etre simple ou pleniere selon les circonstances.\n\n## Articles applicables\n\n- **Article 343-1 du Code civil** : L'adoption peut etre demandee par le conjoint du pere ou de la mere de l'enfant.\n- **Article 345-1 du Code civil** : L'adoption pleniere de l'enfant du conjoint est permise lorsque l'enfant n'a de filiation etablie qu'a l'egard de ce conjoint, ou si l'autre parent s'est vu retirer l'autorite parentale.\n- **Article 360 du Code civil** : L'adoption simple est possible meme si l'enfant a une filiation etablie a l'egard de ses deux parents.\n\n## Deux formes d'adoption\n\n- **Adoption simple** : L'enfant garde ses liens avec l'autre parent biologique. Plus facile a obtenir.\n- **Adoption pleniere** : Rompt definitivement les liens avec l'autre parent biologique. Plus restrictive.\n\n## Conditions\n\n- Etre marie avec le parent de l'enfant (ou pacse depuis la loi de 2022)\n- Consentement de l'autre parent biologique (si adoption simple)\n- Difference d'age d'au moins 10 ans avec l'enfant\n- Accord de l'enfant s'il a plus de 13 ans\n\n## Prochaines etapes\n\n1. Deposez une requete en adoption aupres du tribunal judiciaire\n2. Obtenez le consentement de l'autre parent biologique (devant notaire)\n3. Le juge evaluera l'interet de l'enfant avant de prononcer l'adoption\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Mon ex-conjoint veut demenager a 500 km avec notre enfant. Peut-il le faire sans mon accord ?",
                "assistant": "## Analyse de la situation\n\nLe demenagement d'un parent avec l'enfant commun est un sujet sensible. Le parent qui souhaite demenager doit informer l'autre parent a l'avance.\n\n## Articles applicables\n\n- **Article 373-2 du Code civil** : Chacun des parents doit maintenir des relations personnelles avec l'enfant et respecter les liens avec l'autre parent.\n- **Article 373-2-6 du Code civil** : Le juge veille a la sauvegarde des interets des enfants.\n- **Article 373-2-11 du Code civil** : Le juge prend en compte l'aptitude de chaque parent a respecter les droits de l'autre.\n\n## Obligation d'information\n\n- Le parent qui demenage doit notifier l'autre **au moins un mois a l'avance** par LRAR\n- L'autre parent peut saisir le juge aux affaires familiales pour s'y opposer\n- Le juge tranchera en fonction de l'interet de l'enfant\n\n## Prochaines etapes\n\n1. Si vous n'etes pas d'accord, saisissez le juge aux affaires familiales en urgence (refere)\n2. Demandez la modification de la residence habituelle de l'enfant\n3. Le juge evaluera l'interet de l'enfant et les consequences du demenagement\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
        ],

        # ── DROIT DE LA CONSOMMATION ───────────────────────────────────
        "Droit de la consommation": [
            {
                "user": "J'ai achete un telephone en ligne il y a 10 jours et il ne fonctionne pas. Le vendeur refuse le remboursement. Quels sont mes droits ?",
                "assistant": "## Analyse de la situation\n\nPour un achat en ligne, vous beneficiez de deux protections complementaires : le droit de retractation de 14 jours et la garantie legale de conformite.\n\n## Articles applicables\n\n- **Article L221-18 du Code de la consommation** : Le consommateur dispose d'un delai de 14 jours pour exercer son droit de retractation lors d'un achat a distance.\n- **Article L217-4 du Code de la consommation** : Le vendeur est tenu de livrer un bien conforme au contrat.\n- **Article L217-7 du Code de la consommation** : La garantie legale de conformite est de 2 ans. Pendant les 24 premiers mois, le defaut est presume existant au moment de la livraison.\n\n## Vos droits\n\n- **Droit de retractation** : Vous avez 14 jours pour retourner le produit sans justification (remboursement sous 14 jours)\n- **Garantie legale de conformite** : Le produit defectueux doit etre repare ou remplace. Si impossible, remboursement integral.\n- Le vendeur ne peut pas vous imposer un avoir au lieu d'un remboursement\n\n## Prochaines etapes\n\n1. Exercez votre droit de retractation par ecrit (vous etes encore dans les 14 jours)\n2. Renvoyez le produit et demandez le remboursement integral\n3. Si le vendeur refuse, signalez-le sur SignalConso.gouv.fr et saisissez le mediateur de la consommation\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Un garagiste m'a facture des reparations que je n'avais pas demandees. Dois-je payer ?",
                "assistant": "## Analyse de la situation\n\nUn professionnel ne peut pas effectuer des travaux supplementaires sans l'accord prealable du client. Vous n'etes pas tenu de payer des prestations non commandees.\n\n## Articles applicables\n\n- **Article 1103 du Code civil** : Les contrats legalement formes tiennent lieu de loi a ceux qui les ont faits.\n- **Article L111-1 du Code de la consommation** : Le professionnel doit communiquer les caracteristiques essentielles du service et son prix avant la conclusion du contrat.\n- **Article 1194 du Code civil** : Les contrats obligent a ce qui y est exprime mais aussi a toutes les suites que l'usage, la loi ou l'equite donnent a l'obligation.\n\n## Vos droits\n\n- Vous ne devez payer que les travaux que vous avez explicitement commandes\n- Le garagiste aurait du vous appeler pour obtenir votre accord avant tout travail supplementaire\n- Si un devis a ete signe, seuls les travaux prevus au devis sont dus\n\n## Prochaines etapes\n\n1. Contestez par ecrit les reparations non autorisees en rappelant votre devis initial\n2. Proposez de payer uniquement les travaux commandes\n3. En cas de litige, saisissez le mediateur de la consommation du secteur automobile\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "J'ai souscrit un abonnement en ligne avec une offre d'essai gratuite et maintenant on me preleve tous les mois. Comment arreter ?",
                "assistant": "## Analyse de la situation\n\nLes offres d'essai qui se transforment en abonnement payant doivent etre clairement indiquees au consommateur. Le manque de transparence peut constituer une pratique commerciale trompeuse.\n\n## Articles applicables\n\n- **Article L121-2 du Code de la consommation** : Une pratique commerciale est trompeuse si elle omet une information substantielle.\n- **Article L215-1 du Code de la consommation** : Le professionnel doit informer le consommateur de la possibilite de ne pas reconduire le contrat au moins un mois avant la date de renouvellement.\n- **Article L221-18 du Code de la consommation** : Droit de retractation de 14 jours pour les contrats conclus a distance.\n\n## Demarches\n\n- Envoyez une lettre de resiliation par LRAR\n- Demandez le remboursement des prelevements effectues sans consentement eclaire\n- Faites opposition au prelevement aupres de votre banque (vous avez 13 mois pour contester un prelevement non autorise)\n\n## Prochaines etapes\n\n1. Resilez immediatement par LRAR et demandez le remboursement des sommes prelevees\n2. Contactez votre banque pour faire opposition aux prelevements futurs\n3. Signalez la pratique sur SignalConso.gouv.fr\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "J'ai achete une voiture d'occasion chez un professionnel il y a 6 mois et le moteur est tombe en panne. Le vendeur dit que c'est de l'usure normale. Ai-je un recours ?",
                "assistant": "## Analyse de la situation\n\nMeme pour un vehicule d'occasion, la garantie legale de conformite s'applique. Une panne moteur 6 mois apres l'achat n'est pas de l'usure normale.\n\n## Articles applicables\n\n- **Article L217-4 du Code de la consommation** : Le vendeur professionnel est tenu de livrer un bien conforme au contrat.\n- **Article L217-7 du Code de la consommation** : Les defauts de conformite qui apparaissent dans les 24 mois sont presumes exister au moment de la livraison.\n- **Article 1641 du Code civil** : La garantie des vices caches permet d'agir dans les 2 ans suivant la decouverte du vice.\n\n## Vos droits\n\n- **Garantie legale de conformite (2 ans)** : Le vendeur professionnel doit reparer ou remplacer le vehicule. A defaut, remboursement.\n- **Garantie des vices caches** : Si le defaut existait avant la vente et etait cache, vous pouvez demander l'annulation de la vente ou une reduction du prix.\n- A 6 mois, le defaut est **presume exister depuis la livraison** — c'est au vendeur de prouver le contraire.\n\n## Prochaines etapes\n\n1. Faites expertiser le vehicule par un expert automobile independant\n2. Mettez en demeure le vendeur par LRAR en invoquant la garantie legale de conformite\n3. En cas de refus, saisissez le mediateur de la consommation puis le tribunal\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Un site internet m'a livre un produit different de celui commande. Que faire ?",
                "assistant": "## Analyse de la situation\n\nLa livraison d'un produit non conforme a la commande constitue un manquement a l'obligation de conformite du vendeur.\n\n## Articles applicables\n\n- **Article L217-4 du Code de la consommation** : Le vendeur livre un bien conforme au contrat, correspondant a la description donnee.\n- **Article L217-9 du Code de la consommation** : En cas de defaut de conformite, l'acheteur choisit entre la reparation et le remplacement.\n- **Article L221-18 du Code de la consommation** : Droit de retractation de 14 jours pour tout achat a distance.\n\n## Vos droits\n\n- Exiger le remplacement par le bon produit sans frais supplementaires\n- Exercer votre droit de retractation et obtenir un remboursement complet\n- Les frais de retour sont a la charge du vendeur en cas de non-conformite\n\n## Prochaines etapes\n\n1. Contactez le service client en gardant une trace ecrite (email, formulaire)\n2. Si pas de reponse sous 7 jours, envoyez une mise en demeure par LRAR\n3. Si le vendeur ne reagit pas, signalez sur SignalConso.gouv.fr et faites opposition au paiement via votre banque\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Mon operateur telecom a augmente le prix de mon forfait sans me prevenir. Puis-je resilier sans frais ?",
                "assistant": "## Analyse de la situation\n\nLa modification unilaterale du prix d'un contrat de telecommunications donne droit a la resiliation sans frais dans un delai de 4 mois.\n\n## Articles applicables\n\n- **Article L224-33 du Code de la consommation** : Toute modification contractuelle doit etre notifiee au consommateur au moins un mois avant son entree en vigueur.\n- **Article L224-33 alinea 2** : Le consommateur peut resilier le contrat sans penalite dans les 4 mois suivant l'entree en vigueur de la modification.\n\n## Vos droits\n\n- Resilier sans frais dans les 4 mois suivant la modification\n- Demander le remboursement du trop-percu si vous avez deja paye le nouveau tarif\n- Conserver votre numero en demandant la portabilite\n\n## Prochaines etapes\n\n1. Verifiez la date d'application de la hausse de prix\n2. Envoyez une lettre de resiliation par LRAR en citant l'article L224-33\n3. Demandez la portabilite de votre numero vers un nouvel operateur\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
        ],

        # ── DROIT PENAL ────────────────────────────────────────────────
        "Droit penal": [
            {
                "user": "J'ai ete victime d'une escroquerie en ligne. Comment porter plainte et quelles sont les chances de recuperer mon argent ?",
                "assistant": "## Analyse de la situation\n\nL'escroquerie est un delit penal passible de lourdes peines. Vous pouvez porter plainte et engager des demarches pour tenter de recuperer les fonds.\n\n## Articles applicables\n\n- **Article 313-1 du Code penal** : L'escroquerie est punie de 5 ans d'emprisonnement et 375 000 euros d'amende.\n- **Article 313-2 du Code penal** : Les peines sont portees a 7 ans et 750 000 euros en cas de circonstances aggravantes (bande organisee, utilisation d'internet).\n- **Article 85 du Code de procedure penale** : La victime peut se constituer partie civile pour obtenir reparation.\n\n## Demarches\n\n- Porter plainte en ligne sur pre-plainte-en-ligne.gouv.fr\n- Signaler l'escroquerie sur internet-signalement.gouv.fr (PHAROS)\n- Contacter votre banque pour tenter un rappel de fonds (chargeback si paiement par carte)\n- Conserver toutes les preuves (emails, captures d'ecran, releves bancaires)\n\n## Prochaines etapes\n\n1. Portez plainte au commissariat ou en ligne avec toutes les preuves\n2. Contactez votre banque immediatement pour tenter un rappel de fonds\n3. Signalez l'escroquerie sur internet-signalement.gouv.fr\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "J'ai ete place en garde a vue. Quels sont mes droits pendant cette periode ?",
                "assistant": "## Analyse de la situation\n\nLa garde a vue est une mesure de contrainte strictement encadree par la loi. Vous disposez de droits fondamentaux des le debut de la mesure.\n\n## Articles applicables\n\n- **Article 63-1 du Code de procedure penale** : La personne gardee a vue doit etre immediatement informee de ses droits.\n- **Article 63-3-1 du Code de procedure penale** : Droit a l'assistance d'un avocat des le debut de la garde a vue.\n- **Article 63-2 du Code de procedure penale** : Droit de faire prevenir un proche et son employeur.\n- **Article 63-3 du Code de procedure penale** : Droit a un examen medical.\n\n## Vos droits en garde a vue\n\n- **Droit au silence** : Vous pouvez refuser de repondre aux questions\n- **Droit a un avocat** : Des la premiere heure, gratuit si vous n'en avez pas (avocat commis d'office)\n- **Droit de prevenir un proche** : Un membre de la famille ou votre employeur\n- **Droit a un medecin** : Examen medical a votre demande ou d'office\n- **Droit d'etre informe** : On doit vous dire pourquoi vous etes garde a vue\n- **Duree maximale** : 24 heures, renouvelable une fois (48h max en droit commun)\n\n## Prochaines etapes\n\n1. Exercez votre droit au silence si vous avez un doute\n2. Demandez immediatement l'assistance d'un avocat\n3. Faites prevenir un proche de votre situation\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Quelqu'un publie des propos diffamatoires a mon encontre sur les reseaux sociaux. Comment agir ?",
                "assistant": "## Analyse de la situation\n\nLa diffamation publique est un delit de presse, meme sur les reseaux sociaux. Elle est soumise a un delai de prescription court de 3 mois.\n\n## Articles applicables\n\n- **Article 29 de la loi du 29 juillet 1881** : La diffamation est toute allegation ou imputation d'un fait portant atteinte a l'honneur ou a la consideration d'une personne.\n- **Article 32 de la loi de 1881** : La diffamation envers un particulier est punie de 12 000 euros d'amende.\n- **Article 65 de la loi de 1881** : Le delai de prescription est de **3 mois** a compter de la publication.\n\n## Diffamation vs injure\n\n- **Diffamation** : Imputation d'un fait precis (\"il a vole son employeur\") — le fait peut etre soumis a preuve\n- **Injure** : Expression outrageante sans imputation de fait (\"c'est un imbecile\") — 12 000 euros d'amende\n\n## Prochaines etapes\n\n1. Faites constater les publications par un huissier de justice ou un constat en ligne (urgent — le contenu peut etre supprime)\n2. Agissez vite : la prescription est de 3 mois seulement\n3. Deposez plainte avec constitution de partie civile aupres du tribunal judiciaire\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "J'ai recu une amende pour exces de vitesse mais je n'etais pas le conducteur ce jour-la. Que faire ?",
                "assistant": "## Analyse de la situation\n\nLe titulaire de la carte grise est presume responsable des infractions au code de la route, mais il peut contester s'il n'etait pas le conducteur.\n\n## Articles applicables\n\n- **Article L121-3 du Code de la route** : Le titulaire du certificat d'immatriculation est redevable pecuniairement de l'amende, sauf s'il designe le conducteur.\n- **Article 529-10 du Code de procedure penale** : Le titulaire peut contester l'amende dans les 45 jours suivant la notification.\n\n## Options\n\n- **Designer le conducteur** : Remplir le formulaire de designation sur le site de l'ANTAI en indiquant l'identite du vrai conducteur\n- **Contester** : Si vous ne pouvez pas ou ne voulez pas designer le conducteur, vous pouvez contester l'avis, mais vous risquez de payer l'amende pecuniaire\n\n## Prochaines etapes\n\n1. Connectez-vous sur antai.gouv.fr dans les 45 jours\n2. Designez le conducteur reel avec son identite complete et son numero de permis\n3. Si vous ne le connaissez pas, contestez par requete en exoneration\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Mon voisin m'a agresse physiquement. Comment porter plainte et quelles indemnisations puis-je obtenir ?",
                "assistant": "## Analyse de la situation\n\nL'agression physique (violences volontaires) est un delit penal. Vous avez le droit de porter plainte et de demander une indemnisation.\n\n## Articles applicables\n\n- **Article 222-11 du Code penal** : Les violences ayant entraine une ITT (incapacite totale de travail) de plus de 8 jours sont punies de 3 ans d'emprisonnement et 45 000 euros d'amende.\n- **Article 222-13 du Code penal** : Les violences ayant entraine une ITT de 8 jours ou moins sont punies de 3 ans d'emprisonnement si commises avec circonstances aggravantes.\n- **Article 706-3 du Code de procedure penale** : La CIVI (Commission d'Indemnisation des Victimes d'Infractions) peut indemniser les victimes de violences.\n\n## Demarches immediates\n\n- Faites constater vos blessures par un medecin (certificat medical avec ITT)\n- Deposez plainte au commissariat ou a la gendarmerie\n- Conservez les preuves (photos des blessures, temoignages, video surveillance)\n\n## Indemnisations possibles\n\n- Dommages-interets en se constituant partie civile\n- Indemnisation par la CIVI si l'agresseur est insolvable\n- Remboursement des frais medicaux, prejudice moral, souffrances\n\n## Prochaines etapes\n\n1. Rendez-vous aux urgences et obtenez un certificat medical avec ITT\n2. Deposez plainte au commissariat avec le certificat medical\n3. Constituez-vous partie civile pour obtenir des dommages-interets\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
            {
                "user": "Quel est le delai de prescription pour porter plainte en France ? J'ai subi un vol il y a 5 ans.",
                "assistant": "## Analyse de la situation\n\nLe delai de prescription varie selon la nature de l'infraction. Pour un vol (delit), le delai de droit commun est de 6 ans.\n\n## Articles applicables\n\n- **Article 7 du Code de procedure penale** : L'action publique des crimes se prescrit par 20 ans.\n- **Article 8 du Code de procedure penale** : L'action publique des delits se prescrit par 6 ans.\n- **Article 9 du Code de procedure penale** : L'action publique des contraventions se prescrit par 1 an.\n\n## Delais de prescription courants\n\n- **Contraventions** (ex : tapage nocturne) : **1 an**\n- **Delits** (ex : vol, escroquerie, abus de confiance) : **6 ans**\n- **Crimes** (ex : meurtre, viol) : **20 ans**\n- **Crimes sur mineurs** (ex : viol sur mineur) : **30 ans** a compter de la majorite de la victime\n\n## Votre situation\n\nLe vol est un delit prescrit par 6 ans. Un vol commis il y a 5 ans est encore dans les delais. Vous pouvez porter plainte.\n\n## Prochaines etapes\n\n1. Portez plainte rapidement (il vous reste environ 1 an)\n2. Rassemblez toutes les preuves dont vous disposez\n3. Consultez un avocat pour evaluer les chances de succes\n\n*Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*"
            },
        ],
    }

    # Generer les exemples au format ChatML
    tous_les_exemples = []
    for domaine, exemples in exemples_par_domaine.items():
        for ex in exemples:
            tous_les_exemples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ex["user"]},
                    {"role": "assistant", "content": ex["assistant"]},
                ]
            })

    # Melanger les exemples
    random.shuffle(tous_les_exemples)

    print(f"  {len(tous_les_exemples)} exemples synthetiques generes")
    print(f"  Domaines couverts : {', '.join(exemples_par_domaine.keys())}")
    return tous_les_exemples


# ── FILTRE QUALITE ──────────────────────────────────────────────────────────

def filtrer_qualite(exemples):
    """
    Supprimer les exemples de faible qualite :
    - Contenu user ou assistant trop court (< 50 caracteres)
    - Messages manquants
    """
    avant = len(exemples)
    exemples_filtres = []

    for ex in exemples:
        messages = ex.get("messages", [])
        if len(messages) < 3:
            continue

        user_content = ""
        assistant_content = ""
        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "assistant":
                assistant_content = msg["content"]

        if len(user_content) >= 50 and len(assistant_content) >= 50:
            exemples_filtres.append(ex)

    apres = len(exemples_filtres)
    print(f"\nFiltre qualite : {avant} -> {apres} exemples ({avant - apres} supprimes)")
    return exemples_filtres


# ── ASSEMBLAGE ET PUSH ──────────────────────────────────────────────────────

def assembler_et_publier():
    """
    Assembler toutes les sources, filtrer, splitter et publier sur HF Hub.
    """
    print("=" * 60)
    print("LEXIA — Enrichissement du dataset v2")
    print("=" * 60)

    # 1. Charger le dataset existant
    exemples_v1 = charger_dataset_v1()

    # 2. Charger les sources HF Hub
    exemples_jurisprudence = convertir_jurisprudence()
    exemples_legalkit = convertir_legalkit()
    exemples_cold_law = convertir_cold_french_law()
    exemples_bsard = convertir_bsard()

    # 3. Charger les donnees API locales (Legifrance + Judilibre)
    exemples_api_locales = charger_donnees_api_locales()

    # 4. Generer les exemples synthetiques
    exemples_synthetiques = generer_exemples_synthetiques()

    # 5. Assembler le tout
    tous_les_exemples = (
        exemples_v1
        + exemples_jurisprudence
        + exemples_legalkit
        + exemples_cold_law
        + exemples_bsard
        + exemples_api_locales
        + exemples_synthetiques
    )

    print(f"\n{'=' * 60}")
    print(f"Total brut : {len(tous_les_exemples)} exemples")
    print(f"  - v1 existant      : {len(exemples_v1)}")
    print(f"  - jurisprudence HF : {len(exemples_jurisprudence)}")
    print(f"  - legalkit HF      : {len(exemples_legalkit)}")
    print(f"  - cold-french-law  : {len(exemples_cold_law)}")
    print(f"  - bsard            : {len(exemples_bsard)}")
    print(f"  - API locales      : {len(exemples_api_locales)}")
    print(f"  - synthetiques     : {len(exemples_synthetiques)}")

    # 5. Filtrer la qualite
    tous_les_exemples = filtrer_qualite(tous_les_exemples)

    # 6. Melanger
    random.shuffle(tous_les_exemples)

    # 7. Split 90/10
    split_idx = int(len(tous_les_exemples) * 0.9)
    train_data = tous_les_exemples[:split_idx]
    eval_data = tous_les_exemples[split_idx:]

    print(f"\nSplit final :")
    print(f"  Train : {len(train_data)} exemples")
    print(f"  Eval  : {len(eval_data)} exemples")
    print(f"  Total : {len(tous_les_exemples)} exemples")

    # 8. Creer le DatasetDict et publier
    print(f"\nPublication sur HF Hub : {DATASET_V2}...")

    ds_train = Dataset.from_list(train_data)
    ds_eval = Dataset.from_list(eval_data)
    ds_dict = DatasetDict({"train": ds_train, "eval": ds_eval})

    ds_dict.push_to_hub(DATASET_V2, token=HF_TOKEN)

    print(f"\nDataset v2 publie sur : https://huggingface.co/datasets/{DATASET_V2}")
    print(f"  Train : {len(ds_train)} exemples")
    print(f"  Eval  : {len(ds_eval)} exemples")
    print("=" * 60)


# ── POINT D'ENTREE ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    assembler_et_publier()
