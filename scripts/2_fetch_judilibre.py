"""
=============================================================================
LEXIA — Script 2 : Récupération des décisions de justice depuis Judilibre
=============================================================================
Ce script se connecte à l'API Judilibre, la base officielle de décisions
anonymisées de la Cour de Cassation française. Ces décisions permettront
au modèle d'apprendre à analyser de vrais cas juridiques.

ENTRÉE  : Mêmes clés PISTE que Légifrance (même portail d'authentification)
SORTIE  : Fichier JSON dans data/judilibre/decisions_judilibre.json
PLAN B  : Génère des exemples synthétiques si l'API n'est pas disponible
=============================================================================
"""

import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Les clés Judilibre sont les mêmes que les clés Légifrance (même portail PISTE)
PISTE_CLIENT_ID     = os.getenv("JUDILIBRE_CLIENT_ID", os.getenv("PISTE_CLIENT_ID"))
PISTE_CLIENT_SECRET = os.getenv("JUDILIBRE_CLIENT_SECRET", os.getenv("PISTE_CLIENT_SECRET"))

# URL de base de l'API Judilibre
BASE_URL  = "https://api.piste.gouv.fr/cassation/judilibre/v1.0"
TOKEN_URL = "https://oauth.piste.gouv.fr/api/oauth/token"

# Dossier de sortie
OUTPUT_DIR = Path("data/judilibre")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sujets à rechercher par chambre de la Cour de Cassation
# Chaque chambre spécialisée dans un domaine juridique
SUJETS_RECHERCHE = [
    {"q": "licenciement",      "chambre": "soc"},   # Chambre sociale → droit du travail
    {"q": "responsabilité",    "chambre": "civ1"},  # 1ère civile → contrats, famille
    {"q": "contrat vente",     "chambre": "civ3"},  # 3ème civile → immobilier
    {"q": "divorce",           "chambre": "civ1"},  # 1ère civile → famille
    {"q": "accident corporel", "chambre": "civ2"},  # 2ème civile → accidents
    {"q": "escroquerie",       "chambre": "crim"},  # Chambre criminelle → pénal
    {"q": "harcèlement",       "chambre": "soc"},   # Sociale → travail
    {"q": "bail locatif",      "chambre": "civ3"},  # 3ème civile → locations
]

# ─── AUTHENTIFICATION ─────────────────────────────────────────────────────────

def obtenir_token():
    """Obtenir le token OAuth2 depuis PISTE (même mécanisme que Légifrance)."""
    reponse = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": PISTE_CLIENT_ID,
            "client_secret": PISTE_CLIENT_SECRET,
            "scope": "openid",
        },
    )
    reponse.raise_for_status()
    print("✅ Token Judilibre obtenu")
    return reponse.json()["access_token"]


def construire_headers(token):
    """Construire les headers avec le token d'authentification."""
    return {"Authorization": f"Bearer {token}", "accept": "application/json"}

# ─── RÉCUPÉRATION DES DÉCISIONS ───────────────────────────────────────────────

def rechercher_decisions(token, query, chambre=None, nombre=25):
    """
    Rechercher des décisions de justice par mots-clés et chambre.

    Args:
        token: Token d'accès OAuth2
        query: Mots-clés de recherche
        chambre: Code de la chambre (soc, civ1, civ2, civ3, crim)
        nombre: Nombre de résultats souhaités

    Returns:
        Liste des décisions trouvées avec leurs métadonnées
    """
    parametres = {
        "query": query,
        "resolve_references": "true",
        "batch": nombre,
        "batch_size": nombre,
    }
    # Filtrer par chambre si spécifiée
    if chambre:
        parametres["chamber"] = chambre

    reponse = requests.get(
        f"{BASE_URL}/search",
        headers=construire_headers(token),
        params=parametres
    )

    if reponse.status_code != 200:
        print(f"  ⚠️ Erreur {reponse.status_code} pour '{query}'")
        return []

    return reponse.json().get("results", [])


def recuperer_detail_decision(token, decision_id):
    """
    Récupérer le texte complet et les métadonnées d'une décision.

    Args:
        token: Token d'accès OAuth2
        decision_id: Identifiant unique de la décision

    Returns:
        Dictionnaire complet de la décision, ou None si erreur
    """
    reponse = requests.get(
        f"{BASE_URL}/decision",
        headers=construire_headers(token),
        params={"id": decision_id, "resolve_references": "true"},
    )

    if reponse.status_code != 200:
        return None

    return reponse.json()


def extraire_informations_utiles(decision):
    """
    Extraire uniquement les informations utiles d'une décision pour le fine-tuning.
    On garde : le texte, le résumé, la solution, les articles appliqués.
    On ignore : les informations personnelles, les références internes.

    Args:
        decision: Dictionnaire complet de la décision Judilibre

    Returns:
        Dictionnaire simplifié avec les champs utiles, ou None si données insuffisantes
    """
    if not decision:
        return None

    texte = decision.get("text", "")
    # Ignorer les décisions trop courtes (pas assez de contenu)
    if not texte or len(texte) < 100:
        return None

    # Extraire les textes de loi appliqués dans la décision
    textes_appliques = []
    zones = decision.get("zones", {})
    for ref in zones.get("visa", []):
        textes_appliques.append(ref)

    return {
        "source": "judilibre",
        "id": decision.get("id"),
        "date": decision.get("decision_date"),
        "juridiction": decision.get("jurisdiction"),
        "chambre": decision.get("chamber"),
        "solution": decision.get("solution"),           # Ex: "Cassation", "Rejet"
        "sommaire": decision.get("summary", ""),         # Résumé de la décision
        "textes_appliques": textes_appliques,            # Articles de loi cités
        "texte_complet": texte[:3000],                   # Limité à 3000 chars
    }

# ─── BOUCLE PRINCIPALE ────────────────────────────────────────────────────────

def recuperer_toutes_les_decisions():
    """
    Boucle principale : récupère les décisions pour chaque domaine juridique.
    Gère les doublons et respecte les limites de l'API.
    """
    print("=" * 60)
    print("⚖️  LEXIA — Récupération des décisions Judilibre")
    print("=" * 60)

    print("\n🔑 Authentification Judilibre...")
    token = obtenir_token()

    toutes_les_decisions = []
    ids_vus = set()  # Pour éviter les doublons

    for sujet in SUJETS_RECHERCHE:
        query   = sujet["q"]
        chambre = sujet.get("chambre")
        print(f"\n⚖️  Recherche : '{query}' (chambre : {chambre or 'toutes'})")

        resultats = rechercher_decisions(token, query, chambre, nombre=25)
        print(f"   → {len(resultats)} décisions trouvées")

        for resultat in resultats:
            dec_id = resultat.get("id")
            if not dec_id or dec_id in ids_vus:
                continue

            # Récupérer le détail complet de la décision
            detail = recuperer_detail_decision(token, dec_id)
            info   = extraire_informations_utiles(detail)

            if info:
                info["query_origine"] = query
                toutes_les_decisions.append(info)
                ids_vus.add(dec_id)

            # Pause pour respecter les limites de l'API
            time.sleep(0.3)

        nb = len([d for d in toutes_les_decisions if d.get("query_origine") == query])
        print(f"   ✅ {nb} décisions sauvegardées")

    # Sauvegarder dans un fichier JSON
    fichier_sortie = OUTPUT_DIR / "decisions_judilibre.json"
    with open(fichier_sortie, "w", encoding="utf-8") as f:
        json.dump(toutes_les_decisions, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ TOTAL : {len(toutes_les_decisions)} décisions récupérées")
    print(f"📁 Sauvegardé dans : {fichier_sortie}")
    return toutes_les_decisions

# ─── PLAN B : EXEMPLES SYNTHÉTIQUES ──────────────────────────────────────────

def creer_exemples_synthetiques():
    """
    Plan B : Créer des exemples juridiques synthétiques si l'API Judilibre
    n'est pas accessible. Ces exemples couvrent les cas les plus fréquents
    en droit français et sont rédigés par un expert juridique.
    """
    print("📝 Création d'exemples juridiques synthétiques...")

    exemples = [
        {
            "source": "synthetic",
            "query_origine": "licenciement",
            "cas": "Un salarié a été licencié après 12 ans d'ancienneté sans convocation préalable à entretien et sans lettre de licenciement motivée.",
            "articles_applicables": [
                "Article L1232-2 Code du travail — Convocation à entretien préalable",
                "Article L1232-6 Code du travail — Lettre de licenciement motivée",
                "Article L1235-3 Code du travail — Indemnités selon ancienneté"
            ],
            "analyse": "Le licenciement est irrégulier sur la forme (absence de convocation et de lettre motivée) et potentiellement sans cause réelle et sérieuse. L'ancienneté de 12 ans donne droit à des indemnités significatives.",
            "solution": "Le salarié peut obtenir des dommages-intérêts pour licenciement sans cause réelle et sérieuse, plus les indemnités légales de licenciement.",
            "arguments_defense": "1. Absence de convocation préalable (Art. L1232-2). 2. Absence de lettre motivée (Art. L1232-6). 3. 12 ans d'ancienneté → indemnités élevées (Art. L1235-3).",
        },
        {
            "source": "synthetic",
            "query_origine": "vice caché",
            "cas": "Un acheteur découvre 8 mois après l'acquisition de son appartement que la toiture fuit. Le vendeur, un professionnel de l'immobilier, avait connaissance du problème.",
            "articles_applicables": [
                "Article 1641 Code civil — Garantie des vices cachés",
                "Article 1642-1 Code civil — Vice apparent vs vice caché",
                "Article 1648 Code civil — Délai de 2 ans pour agir"
            ],
            "analyse": "Vice caché caractérisé : existait avant la vente, était inconnu de l'acheteur, rend le bien impropre à l'usage. Le vendeur professionnel est présumé de mauvaise foi.",
            "solution": "Action en garantie des vices cachés dans les 2 ans de la découverte. Possibilité de résolution de la vente ou de réduction du prix.",
            "arguments_defense": "1. Vice antérieur à la vente (Art. 1641). 2. Inconnu de l'acheteur. 3. Vendeur professionnel présumé connaître les vices. 4. Délai respecté (Art. 1648).",
        },
        {
            "source": "synthetic",
            "query_origine": "loyer impayé",
            "cas": "Un locataire n'a pas payé son loyer depuis 4 mois malgré deux lettres de relance. Il invoque des troubles de jouissance dus à des infiltrations non réparées.",
            "articles_applicables": [
                "Article 7 Loi du 6 juillet 1989 — Obligations du locataire",
                "Article 6 Loi du 6 juillet 1989 — Obligations du bailleur",
                "Article L412-1 Code des procédures civiles d'exécution — Expulsion"
            ],
            "analyse": "Situation complexe : le locataire a l'obligation de payer le loyer (Art. 7) mais le bailleur a l'obligation d'assurer la jouissance paisible (Art. 6). L'exception d'inexécution peut être invoquée.",
            "solution": "Mise en demeure préalable obligatoire. Si les infiltrations sont prouvées, le juge peut accorder une réduction de loyer. L'expulsion nécessite une décision judiciaire.",
            "arguments_defense": "1. Exception d'inexécution si troubles prouvés. 2. Mise en demeure de réparer préalablement. 3. Consignation du loyer possible. 4. Procédure d'expulsion longue (Art. L412-1).",
        },
        {
            "source": "synthetic",
            "query_origine": "harcèlement moral",
            "cas": "Une salariée subit depuis 18 mois des reproches répétés injustifiés, une mise à l'écart des réunions, et une surcharge de travail documentée. Elle a consulté le médecin du travail.",
            "articles_applicables": [
                "Article L1152-1 Code du travail — Définition du harcèlement moral",
                "Article L1152-2 Code du travail — Protection du salarié harcelé",
                "Article L1154-1 Code du travail — Charge de la preuve partagée"
            ],
            "analyse": "Les éléments caractérisent le harcèlement moral : répétition des actes, dégradation des conditions de travail, atteinte à la dignité. La consultation du médecin du travail constitue un élément de preuve.",
            "solution": "Action aux prud'hommes pour harcèlement moral. Possibilité de prise d'acte de la rupture du contrat aux torts de l'employeur.",
            "arguments_defense": "1. Établir la répétition des agissements (Art. L1152-1). 2. Présenter les preuves : emails, témoignages, avis médecin. 3. Charge de la preuve partagée (Art. L1154-1).",
        },
        {
            "source": "synthetic",
            "query_origine": "accident circulation",
            "cas": "Un piéton a été renversé par un véhicule qui a grillé un feu rouge. Il souffre d'une fracture du genou avec 3 mois d'arrêt de travail.",
            "articles_applicables": [
                "Article 3 Loi Badinter du 5 juillet 1985 — Indemnisation sans faute",
                "Article 4 Loi Badinter — Faute du piéton",
                "Article L211-1 Code des assurances — Obligation d'assurance"
            ],
            "analyse": "La loi Badinter de 1985 offre une protection maximale aux piétons victimes d'accidents de la circulation. L'indemnisation est automatique sauf faute inexcusable du piéton.",
            "solution": "Indemnisation par l'assurance du conducteur fautif : préjudice corporel, perte de revenus (3 mois d'arrêt), préjudice moral, frais médicaux.",
            "arguments_defense": "1. Loi Badinter : indemnisation automatique (Art. 3). 2. Feu rouge grillé = faute exclusive du conducteur. 3. Évaluation du préjudice : ITT, AIPP, pertes de revenus.",
        },
        {
            "source": "synthetic",
            "query_origine": "divorce",
            "cas": "Un couple marié depuis 15 ans souhaite divorcer. Le mari refuse le divorce par consentement mutuel et conteste la garde des deux enfants de 8 et 12 ans.",
            "articles_applicables": [
                "Article 229 Code civil — Les cas de divorce",
                "Article 373-2 Code civil — Exercice de l'autorité parentale",
                "Article 270 Code civil — Prestation compensatoire"
            ],
            "analyse": "En l'absence de consentement mutuel, le divorce peut être demandé pour altération définitive du lien conjugal (2 ans de séparation) ou pour faute. La garde est fixée selon l'intérêt supérieur de l'enfant.",
            "solution": "Saisir le juge aux affaires familiales. Demander la résidence alternée ou principale selon l'intérêt des enfants. Évaluer la prestation compensatoire selon la durée du mariage.",
            "arguments_defense": "1. Divorce pour altération du lien conjugal (Art. 237). 2. Intérêt supérieur de l'enfant (Art. 373-2). 3. Prestation compensatoire selon disparité (Art. 270).",
        },
        {
            "source": "synthetic",
            "query_origine": "escroquerie",
            "cas": "Un individu a vendu en ligne un véhicule qu'il ne possédait pas, encaissant 12 000 euros par virement bancaire. L'acheteur n'a jamais reçu le véhicule.",
            "articles_applicables": [
                "Article 313-1 Code pénal — Escroquerie",
                "Article 313-7 Code pénal — Peines complémentaires",
                "Article 1240 Code civil — Responsabilité civile délictuelle"
            ],
            "analyse": "L'escroquerie est caractérisée : usage de manœuvres frauduleuses (annonce mensongère), tromperie sur l'existence du bien, remise de fonds par la victime. Peine encourue : 5 ans d'emprisonnement et 375 000€ d'amende.",
            "solution": "Porter plainte pour escroquerie (Art. 313-1 Code pénal). Constitution de partie civile pour obtenir la restitution des 12 000€ et des dommages-intérêts.",
            "arguments_defense": "1. Manœuvres frauduleuses caractérisées (Art. 313-1). 2. Préjudice financier direct de 12 000€. 3. Action civile en parallèle du pénal (Art. 1240).",
        },
        {
            "source": "synthetic",
            "query_origine": "discrimination embauche",
            "cas": "Une candidate qualifiée a été refusée à un poste après avoir mentionné sa grossesse lors de l'entretien d'embauche. Le poste a été attribué à un candidat moins expérimenté.",
            "articles_applicables": [
                "Article L1132-1 Code du travail — Principe de non-discrimination",
                "Article L1225-1 Code du travail — Protection de la femme enceinte",
                "Article 225-1 Code pénal — Discrimination"
            ],
            "analyse": "La discrimination à l'embauche en raison de la grossesse est interdite par le Code du travail et le Code pénal. L'employeur ne peut pas prendre en compte l'état de grossesse dans sa décision.",
            "solution": "Saisir le Défenseur des droits et/ou le conseil de prud'hommes. Possibilité de plainte pénale pour discrimination (Art. 225-1 CP). Dommages-intérêts pour préjudice moral.",
            "arguments_defense": "1. Discrimination prohibée (Art. L1132-1). 2. Protection spécifique grossesse (Art. L1225-1). 3. Comparaison des profils candidats. 4. Sanction pénale possible (Art. 225-1 CP).",
        },
        {
            "source": "synthetic",
            "query_origine": "rupture contrat",
            "cas": "Un artisan a réalisé des travaux de rénovation pour un particulier. Le client refuse de payer le solde de 8 000€ en invoquant des malfaçons non prouvées.",
            "articles_applicables": [
                "Article 1217 Code civil — Inexécution du contrat",
                "Article 1231-1 Code civil — Dommages-intérêts pour inexécution",
                "Article 1353 Code civil — Charge de la preuve"
            ],
            "analyse": "L'artisan a exécuté sa prestation. Le client qui invoque des malfaçons doit les prouver (Art. 1353). Sans preuve de malfaçon, le refus de payer constitue une inexécution contractuelle.",
            "solution": "Mise en demeure de payer sous 8 jours. En cas de refus, saisir le tribunal judiciaire pour obtenir le paiement forcé et des dommages-intérêts pour résistance abusive.",
            "arguments_defense": "1. Obligation de payer le prix convenu (Art. 1217). 2. Charge de la preuve des malfaçons sur le client (Art. 1353). 3. Dommages-intérêts pour retard (Art. 1231-1).",
        },
        {
            "source": "synthetic",
            "query_origine": "garde enfant",
            "cas": "Après un divorce, la mère souhaite déménager à 500 km avec l'enfant de 6 ans. Le père, qui exerce la garde alternée, s'y oppose.",
            "articles_applicables": [
                "Article 373-2 Code civil — Exercice de l'autorité parentale",
                "Article 373-2-6 Code civil — Modification de la résidence",
                "Article 373-2-11 Code civil — Critères de décision du juge"
            ],
            "analyse": "Le déménagement à 500 km rendrait la garde alternée impossible. Le juge devra trancher selon l'intérêt supérieur de l'enfant en tenant compte de l'âge, des liens affectifs, et de la stabilité.",
            "solution": "Saisir le juge aux affaires familiales avant le déménagement. Le juge évaluera si le déménagement est justifié et proposera un aménagement du droit de visite.",
            "arguments_defense": "1. Obligation d'informer l'autre parent (Art. 373-2). 2. Intérêt de l'enfant prime (Art. 373-2-11). 3. Maintien des liens avec les deux parents.",
        },
    ]

    # Sauvegarder les exemples synthétiques
    fichier_sortie = OUTPUT_DIR / "decisions_synthetic.json"
    with open(fichier_sortie, "w", encoding="utf-8") as f:
        json.dump(exemples, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(exemples)} exemples synthétiques créés → {fichier_sortie}")
    return exemples

# ─── POINT D'ENTRÉE ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Vérifier si les clés PISTE sont disponibles
    if not PISTE_CLIENT_ID or PISTE_CLIENT_ID == "xxxx":
        print("⚠️  Clés PISTE non configurées → plan B (exemples synthétiques)")
        creer_exemples_synthetiques()
    else:
        try:
            recuperer_toutes_les_decisions()
        except Exception as e:
            print(f"❌ Erreur API Judilibre : {e}")
            print("🔄 Basculement sur le plan B...")
            creer_exemples_synthetiques()
