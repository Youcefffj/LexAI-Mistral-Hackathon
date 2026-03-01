"""
=============================================================================
LEXIA — Script 1 : Récupération des articles de loi depuis Légifrance
=============================================================================
Ce script se connecte à l'API officielle Légifrance via le portail PISTE
du gouvernement français. Il récupère les articles de loi pertinents pour
les domaines juridiques couverts par LexIA (civil, pénal, travail, etc.)

ENTRÉE  : Clés API PISTE dans le fichier .env
SORTIE  : Fichier JSON dans data/legifrance/articles_legifrance.json
PLAN B  : Si les clés PISTE ne sont pas disponibles, utilise les datasets
          publics de Hugging Face comme source de données alternative
=============================================================================
"""

import os
import json
import time
import re
import requests
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Clés API récupérées depuis le portail PISTE (piste.gouv.fr)
PISTE_CLIENT_ID     = os.getenv("PISTE_CLIENT_ID")
PISTE_CLIENT_SECRET = os.getenv("PISTE_CLIENT_SECRET")

# URLs de l'API Légifrance
BASE_URL  = "https://api.piste.gouv.fr/dila/legifrance/lf-engine-app"
TOKEN_URL = "https://oauth.piste.gouv.fr/api/oauth/token"

# Dossier de sortie pour les données brutes
OUTPUT_DIR = Path("data/legifrance")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Domaines juridiques à récupérer — choisis pour couvrir les cas courants
# que LexIA devra analyser (licenciement, divorce, loyer, accident, etc.)
SEARCH_QUERIES = [
    # Droit du travail
    "licenciement abusif",
    "licenciement économique",
    "contrat de travail rupture",
    "harcèlement moral travail",
    "discrimination embauche",
    "heures supplémentaires",
    "rupture conventionnelle",
    "période essai",
    "congé maternité",
    "accident travail",
    "faute grave licenciement",
    "clause non concurrence",
    # Droit civil / famille
    "responsabilité civile",
    "divorce séparation biens",
    "garde enfant autorité parentale",
    "pension alimentaire",
    "succession héritage",
    "prestation compensatoire",
    "adoption",
    "obligation alimentaire",
    # Droit du logement
    "vice caché achat immobilier",
    "loyer impayé expulsion",
    "bail habitation",
    "dépôt garantie caution",
    "trouble voisinage",
    "copropriété charges",
    "expulsion locataire",
    # Droit pénal
    "abus de confiance escroquerie",
    "vol agression",
    "diffamation injure",
    "harcèlement pénal",
    "violence conjugale",
    "conduite état alcoolique",
    # Droit de la consommation
    "garantie conformité",
    "rétractation achat distance",
    "pratique commerciale trompeuse",
    "clause abusive contrat",
    # Droit des assurances
    "indemnisation accident",
    "assurance habitation sinistre",
    "rupture promesse vente",
    # Droit administratif
    "permis construire",
    "fonction publique",
]

# ─── AUTHENTIFICATION ─────────────────────────────────────────────────────────

def obtenir_token():
    """
    Obtenir un token d'accès OAuth2 depuis le portail PISTE.
    Ce token est valable 1 heure et doit être inclus dans toutes les requêtes.
    """
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
    token = reponse.json()["access_token"]
    print("✅ Token PISTE obtenu avec succès")
    return token


def construire_headers(token):
    """Construire les headers HTTP avec le token d'authentification."""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "accept": "application/json",
    }

# ─── RECHERCHE D'ARTICLES ─────────────────────────────────────────────────────

def rechercher_articles(token, query, taille_page=25):
    """
    Rechercher des articles de loi par mot-clé dans la base Légifrance.
    Filtre uniquement les articles en vigueur (pas les articles abrogés).

    Args:
        token: Token d'accès OAuth2
        query: Mots-clés de recherche (ex: "licenciement abusif")
        taille_page: Nombre de résultats à récupérer

    Returns:
        Liste des métadonnées des articles trouvés
    """
    headers = construire_headers(token)

    # Corps de la requête de recherche
    corps = {
        "recherche": {
            "champs": [{
                "typeChamp": "ALL",
                "criteres": [{
                    "typeRecherche": "EXACTE",
                    "valeur": query,
                    "operateur": "ET"
                }]
            }],
            # Filtrer uniquement les articles en vigueur
            "filtres": [{"facette": "ETAT_JURIDIQUE", "valeur": "VIGUEUR"}],
            "pageNumber": 1,
            "pageSize": taille_page,
            "operateur": "ET",
            "sort": "PERTINENCE",
            "typePagination": "DEFAUT",
        },
        "fond": "LEGI",  # Base de données Légifrance
    }

    reponse = requests.post(f"{BASE_URL}/search", headers=headers, json=corps)

    if reponse.status_code != 200:
        print(f"  ⚠️ Erreur {reponse.status_code} pour '{query}'")
        return []

    resultats = reponse.json().get("results", [])
    articles = []

    for resultat in resultats:
        for titre in resultat.get("titles", []):
            articles.append({
                "id": titre.get("id"),
                "titre": titre.get("title"),
                "cid": titre.get("cid"),
            })

    return articles


def recuperer_contenu_article(token, article_id):
    """
    Récupérer le texte complet d'un article de loi par son identifiant.

    Args:
        token: Token d'accès OAuth2
        article_id: Identifiant unique de l'article dans Légifrance

    Returns:
        Dictionnaire avec le contenu complet de l'article, ou None si erreur
    """
    headers = construire_headers(token)

    reponse = requests.post(
        f"{BASE_URL}/consult/getArticle",
        headers=headers,
        json={"id": article_id}
    )

    if reponse.status_code != 200:
        return None

    donnees = reponse.json()
    article = donnees.get("article", {})

    return {
        "id": article_id,
        "numero": article.get("num", ""),
        "texte": article.get("texteHtml", article.get("texte", "")),
        "etat": article.get("etat", ""),
        "date_debut": article.get("dateDebut", ""),
    }


def nettoyer_html(texte):
    """
    Supprimer les balises HTML du texte pour obtenir du texte brut.
    Légifrance retourne parfois du HTML dans les textes d'articles.
    """
    # Remplacer les balises de retour à la ligne par des vrais retours
    texte = texte.replace("<br>", "\n").replace("<br/>", "\n")
    # Supprimer toutes les autres balises HTML
    texte = re.sub(r"<[^>]+>", "", texte)
    return texte.strip()

# ─── BOUCLE PRINCIPALE ────────────────────────────────────────────────────────

def recuperer_tous_les_articles():
    """
    Boucle principale : récupère les articles pour chaque domaine juridique.
    Évite les doublons en gardant un ensemble des IDs déjà traités.
    Respecte les quotas de l'API avec un délai entre les requêtes.
    """
    print("=" * 60)
    print("⚖️  LEXIA — Récupération des articles Légifrance")
    print("=" * 60)

    print("\n🔑 Authentification au portail PISTE...")
    token = obtenir_token()

    tous_les_articles = []
    ids_vus = set()  # Pour éviter les doublons

    for query in SEARCH_QUERIES:
        print(f"\n🔍 Recherche : '{query}'")

        # Récupérer les métadonnées des articles
        articles_meta = rechercher_articles(token, query, taille_page=50)
        print(f"   → {len(articles_meta)} articles trouvés")

        for meta in articles_meta:
            article_id = meta.get("id")

            # Ignorer si pas d'ID ou déjà traité
            if not article_id or article_id in ids_vus:
                continue

            # Récupérer le contenu complet de l'article
            contenu = recuperer_contenu_article(token, article_id)

            if contenu and contenu.get("texte"):
                # Nettoyer le texte HTML
                texte_propre = nettoyer_html(contenu["texte"])

                # Ignorer les articles trop courts (probablement vides)
                if len(texte_propre) > 50:
                    tous_les_articles.append({
                        "source": "legifrance",
                        "query_origine": query,
                        "article_id": article_id,
                        "numero": contenu["numero"],
                        "titre": meta.get("titre", ""),
                        "texte": texte_propre,
                        "etat": contenu["etat"],
                    })
                    ids_vus.add(article_id)

            # Pause de 200ms pour respecter les quotas de l'API
            time.sleep(0.2)

        nb_query = len([a for a in tous_les_articles if a["query_origine"] == query])
        print(f"   ✅ {nb_query} articles sauvegardés pour '{query}'")

    # Sauvegarder tous les articles dans un fichier JSON
    fichier_sortie = OUTPUT_DIR / "articles_legifrance.json"
    with open(fichier_sortie, "w", encoding="utf-8") as f:
        json.dump(tous_les_articles, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ TOTAL : {len(tous_les_articles)} articles récupérés")
    print(f"📁 Sauvegardé dans : {fichier_sortie}")
    return tous_les_articles

# ─── PLAN B : FALLBACK DATASETS HUGGING FACE ──────────────────────────────────

def fallback_huggingface():
    """
    Plan B : Si les clés PISTE ne sont pas disponibles ou si l'API échoue,
    utiliser le dataset erdal/legifrance sur Hugging Face.
    Ce dataset contient les articles de loi officiels français avec leurs
    numéros, textes, codes sources et états juridiques.
    """
    print("📦 PLAN B : Chargement depuis erdal/legifrance sur HF Hub...")
    from datasets import load_dataset

    tous_les_articles = []

    # Codes juridiques pertinents pour les domaines couverts par LexIA
    codes_cibles = [
        "code-civil",
        "code-penal",
        "code-du-travail",
        "code-de-la-consommation",
        "code-de-procedure-civile",
        "code-de-procedure-penale",
        "code-de-commerce",
        "code-des-assurances",
    ]

    try:
        print("  → Chargement du dataset erdal/legifrance (streaming)...")
        ds = load_dataset("erdal/legifrance", split="train", streaming=True)

        compteur = 0
        for item in ds:
            code_name = item.get("code_name", "")

            # Filtrer uniquement les codes pertinents
            if code_name not in codes_cibles:
                continue

            texte = item.get("texte", "").strip()
            etat = item.get("etat", "")

            # Garder uniquement les articles en vigueur et suffisamment longs
            if etat != "VIGUEUR" or len(texte) < 50:
                continue

            tous_les_articles.append({
                "source": "hf_legifrance",
                "article_id": item.get("id", ""),
                "numero": item.get("num", ""),
                "texte": texte,
                "etat": etat,
                "titre": item.get("section_titre", ""),
                "code_source": code_name,
                "query_origine": code_name,
            })
            compteur += 1

            # Limiter à 800 articles pour garder un dataset gérable
            if compteur >= 800:
                break

            # Afficher la progression
            if compteur % 100 == 0:
                print(f"  → {compteur} articles collectés...")

        print(f"  ✅ {len(tous_les_articles)} articles de loi chargés depuis HF")
    except Exception as e:
        print(f"  ⚠️ erdal/legifrance indisponible : {e}")

    # Sauvegarder le fallback
    fichier_sortie = OUTPUT_DIR / "articles_hf_fallback.json"
    with open(fichier_sortie, "w", encoding="utf-8") as f:
        json.dump(tous_les_articles, f, ensure_ascii=False, indent=2)

    print(f"✅ Plan B : {len(tous_les_articles)} articles sauvegardés → {fichier_sortie}")
    return tous_les_articles

# ─── POINT D'ENTRÉE ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Vérifier si les clés PISTE sont configurées
    if not PISTE_CLIENT_ID or PISTE_CLIENT_ID == "xxxx":
        print("⚠️  Clés PISTE non configurées → utilisation du plan B (HF datasets)")
        fallback_huggingface()
    else:
        try:
            recuperer_tous_les_articles()
        except Exception as e:
            print(f"❌ Erreur API Légifrance : {e}")
            print("🔄 Basculement sur le plan B...")
            fallback_huggingface()
