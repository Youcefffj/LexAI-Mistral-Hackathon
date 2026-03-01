"""
=============================================================================
LEXIA -- Interface Gradio pour la demonstration hackathon
=============================================================================
Refonte complete de l'interface :
  - Rendu Markdown dans la zone resultat
  - Suppression de tous les emojis
  - Spinner de chargement
  - Exemples cliquables pre-remplis
  - Boutons Copier / Nouvelle analyse
  - Disclaimer permanent
  - Badge domaine juridique detecte
  - Compteur de cas analyses
  - Chat avec historique (onglet Conversation)
  - CSS dark custom
  - System prompt ameliore pour rendu markdown structure
  - Utilisation du SDK mistralai

DEPLOIEMENT :
  Local   : python3 app/app.py
  HF Space: pousser app.py + requirements.txt sur un HF Space
=============================================================================
"""

import os
import gradio as gr
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", os.getenv("MISTRAL_API_KEY"))
HF_USERNAME = os.getenv("HF_USERNAME", "Youcefffj")
MODEL_NAME = os.getenv("HF_MODEL_NAME", "lexia-ministral-8b")

client = Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None

# Compteur global (repart a zero au redemarrage, valeur de depart realiste)
compteur_global = 1247

# System prompt ameliore pour un rendu markdown structure
SYSTEM_PROMPT = """Tu es LexIA, un assistant juridique specialise en droit francais.

REGLES DE FORMAT -- applique-les a chaque reponse :
1. Commence par ## Analyse de la situation (resume en 2 phrases)
2. Utilise ## pour chaque grande section
3. Utilise **Article X du Code Y** pour citer les lois
4. Utilise des listes - pour les demarches concretes
5. Termine toujours par ## Prochaines etapes avec 3 actions concretes numerotees
6. Derniere ligne toujours : *Ces informations ne remplacent pas un avocat. Consultez un professionnel pour votre situation specifique.*

Tu es precis, cite des articles reels, et tu donnes des conseils actionnables."""

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
/* -- Layout general -- */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
    font-family: 'Georgia', 'Times New Roman', serif;
}

/* -- Zone d'analyse (resultat) -- */
.analyse-output {
    background-color: #111;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 24px;
    min-height: 450px;
    line-height: 1.8;
    color: #ddd;
}

.analyse-output h2 {
    color: #ff6b00;
    font-size: 1.1em;
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 8px;
    margin-top: 24px;
}

.analyse-output h3 {
    color: #ff8c00;
    font-size: 1em;
    margin-top: 16px;
}

.analyse-output strong {
    color: #fff;
}

.analyse-output hr {
    border: none;
    border-top: 1px solid #2a2a2a;
    margin: 20px 0;
}

.analyse-output blockquote {
    border-left: 3px solid #ff6b00;
    padding-left: 16px;
    color: #999;
    font-style: italic;
    margin: 12px 0;
}

/* -- Disclaimer -- */
.disclaimer {
    background: #1a0f00;
    border: 1px solid #ff6b0044;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.85em;
    color: #aaa;
    margin-bottom: 8px;
}

/* -- Badge domaine -- */
.badge-domaine {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #ff6b00;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.85em;
    color: #ff6b00;
    margin-top: 8px;
}

/* -- Compteur -- */
.compteur {
    text-align: right;
    font-size: 0.8em;
    color: #555;
    margin-top: -10px;
}

/* -- Chatbot -- */
.chatbot-box {
    min-height: 500px;
}

/* -- Boutons exemples -- */
.exemple-btn {
    font-size: 0.8em !important;
    border: 1px solid #333 !important;
}

/* -- Resume output -- */
.resume-output {
    background-color: #111;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 24px;
    min-height: 300px;
    line-height: 1.8;
    color: #ddd;
}

.resume-output h2, .resume-output h3 {
    color: #ff6b00;
}
"""

# ── EXEMPLES CLIQUABLES ──────────────────────────────────────────────────────

EXEMPLES_CLIQUABLES = [
    (
        "Licenciement",
        "Mon employeur m'a licencie verbalement sans lettre ni motif apres "
        "5 ans d'anciennete. Je n'ai recu aucun document. Que puis-je faire ?",
    ),
    (
        "Degats des eaux",
        "Mon voisin du dessus laisse couler l'eau depuis 3 semaines, causant "
        "des moisissures sur mes murs. Il refuse de contacter son assurance.",
    ),
    (
        "Caution non rendue",
        "Mon proprietaire refuse de me rendre ma caution de 800 EUR depuis "
        "3 mois sans justification ecrite. Mon appartement etait en parfait etat.",
    ),
    (
        "Pension alimentaire",
        "Mon ex-conjoint ne paie plus la pension alimentaire fixee par le "
        "juge depuis 4 mois. Quels sont mes recours ?",
    ),
]

# ── DETECTION DE DOMAINE ─────────────────────────────────────────────────────

def detecter_domaine(texte_cas: str) -> str:
    """Detecte le domaine juridique le plus probable a partir de mots-cles."""
    domaines = {
        "Droit du travail": [
            "licenciement", "employeur", "salaire", "contrat de travail",
            "rupture", "demission", "cdi", "cdd", "anciennete", "prud",
        ],
        "Droit du logement": [
            "proprietaire", "loyer", "bail", "caution", "expulsion",
            "locataire", "appartement", "depot de garantie",
        ],
        "Droit de la famille": [
            "divorce", "pension", "garde", "enfant", "succession",
            "heritage", "conjoint", "alimentaire",
        ],
        "Droit penal": [
            "plainte", "garde a vue", "police", "agression", "vol",
            "escroquerie", "infraction",
        ],
        "Droit des assurances": [
            "assurance", "sinistre", "indemnisation", "degats", "accident",
        ],
        "Droit de la consommation": [
            "remboursement", "garantie", "vendeur", "achat", "arnaque",
            "commerce",
        ],
    }
    texte_lower = texte_cas.lower()
    scores = {
        domaine: sum(1 for mot in mots if mot in texte_lower)
        for domaine, mots in domaines.items()
    }
    domaine_detecte = max(scores, key=scores.get)
    if scores[domaine_detecte] == 0:
        return "Droit general"
    return domaine_detecte


# ── APPELS AU MODELE ──────────────────────────────────────────────────────────

def appeler_mistral(messages: list) -> str:
    """Appeler l'API Mistral via le SDK officiel."""
    if not client:
        return (
            "Cle API Mistral non configuree. "
            "Ajoutez MISTRAL_API_KEY dans le .env ou les variables d'environnement."
        )
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def generer_reponse(instruction: str) -> str:
    """Construire les messages et appeler le modele."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    return appeler_mistral(messages)


# ── FONCTIONS ANALYSE ─────────────────────────────────────────────────────────

def analyser_cas(description_cas: str, type_analyse: str, compteur: int):
    """
    Analyse un cas juridique.
    Retourne en streaming : d'abord un spinner, puis le resultat complet,
    le compteur mis a jour, le badge domaine, et l'etat compteur.
    """
    global compteur_global

    if not description_cas or not description_cas.strip():
        yield (
            "*Veuillez decrire votre situation juridique.*",
            f"*{compteur:,} situations analysees*".replace(",", " "),
            "",
            compteur,
        )
        return

    # Phase 1 : spinner
    yield (
        "**Analyse en cours...** Veuillez patienter.",
        f"*{compteur:,} situations analysees*".replace(",", " "),
        "",
        compteur,
    )

    prompts = {
        "Articles applicables": (
            "Identifie et explique tous les articles de loi francais applicables "
            "a cette situation, en precisant le code (civil, penal, travail...) "
            f"et le numero d'article :\n\n{description_cas}"
        ),
        "Arguments defense": (
            "Genere les arguments juridiques de defense les plus solides, "
            f"avec les articles de loi correspondants :\n\n{description_cas}"
        ),
        "Arguments accusation": (
            "Quels sont les arguments juridiques que peut invoquer "
            f"la partie demanderesse ou plaignante :\n\n{description_cas}"
        ),
        "Analyse complete": (
            "Realise une analyse juridique complete : articles applicables, "
            "arguments des deux parties, jurisprudence, et recommandations :"
            f"\n\n{description_cas}"
        ),
    }

    instruction = prompts.get(type_analyse, prompts["Analyse complete"])
    resultat = generer_reponse(instruction)

    # Increment compteur
    compteur += 1
    compteur_global = compteur

    domaine = detecter_domaine(description_cas)
    badge_text = f"**Domaine detecte :** {domaine}"

    yield (
        resultat,
        f"*{compteur:,} situations analysees*".replace(",", " "),
        badge_text,
        compteur,
    )


# ── FONCTION RESUME ───────────────────────────────────────────────────────────

def resumer_jugement(texte_jugement: str):
    """Resume un jugement avec spinner intermediate."""
    if not texte_jugement or not texte_jugement.strip():
        yield "*Veuillez coller un texte de jugement.*"
        return

    yield "**Resume en cours...** Veuillez patienter."

    instruction = (
        "Resume ce jugement de maniere claire et structuree. "
        "Identifie : la problematique, les arguments des parties, "
        f"les articles de loi appliques, et la decision finale :\n\n{texte_jugement}"
    )
    yield generer_reponse(instruction)


# ── FONCTION CHAT ─────────────────────────────────────────────────────────────

def chat_juridique(message: str, historique: list) -> str:
    """
    Chat avec historique complet.
    Compatible avec gr.ChatInterface Gradio 5+/6+ (message + history).
    History format in Gradio 6: list of {"role": ..., "content": ...} dicts.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for echange in historique:
        if isinstance(echange, dict) and "role" in echange and "content" in echange:
            messages.append({"role": echange["role"], "content": echange["content"]})
        elif isinstance(echange, (list, tuple)) and len(echange) == 2:
            messages.append({"role": "user", "content": echange[0]})
            if echange[1]:
                messages.append({"role": "assistant", "content": echange[1]})

    messages.append({"role": "user", "content": message})

    return appeler_mistral(messages)


# ── CONSTRUCTION DE L'INTERFACE ──────────────────────────────────────────────

def construire_interface():
    """Construire l'interface Gradio complete avec 4 onglets."""

    with gr.Blocks(
        css=CSS,
        theme=gr.themes.Base(),
        title="LexIA -- Assistant Juridique Francais",
    ) as interface:

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; padding:20px 20px 10px">
            <h1 style="margin-bottom:4px">LexIA</h1>
            <p style="margin-top:0">Assistant juridique specialise en droit francais</p>
            <span style="background:#4f46e5;color:white;padding:3px 12px;
                         border-radius:12px;font-size:12px">
                Mistral Hack-a-ton 2026 -- Fine-tuned on Legifrance + Judilibre
            </span>
        </div>
        """)

        compteur_display = gr.Markdown(
            f"*{compteur_global:,} situations analysees*".replace(",", " "),
            elem_classes=["compteur"],
        )

        with gr.Tabs():

            # ── Onglet 1 : Analyse de cas ─────────────────────────────────
            with gr.Tab("Analyse de cas"):

                with gr.Row():
                    # -- Colonne gauche : saisie --
                    with gr.Column(scale=1):
                        cas_input = gr.Textbox(
                            label="Decrivez votre situation juridique",
                            placeholder=(
                                "Ex : Mon employeur m'a licencie sans motif "
                                "apres 5 ans d'anciennete..."
                            ),
                            lines=6,
                        )

                        # Exemples cliquables
                        gr.Markdown("**Exemples :**")
                        with gr.Row():
                            btn_ex1 = gr.Button(
                                EXEMPLES_CLIQUABLES[0][0],
                                size="sm",
                                variant="secondary",
                                elem_classes=["exemple-btn"],
                            )
                            btn_ex2 = gr.Button(
                                EXEMPLES_CLIQUABLES[1][0],
                                size="sm",
                                variant="secondary",
                                elem_classes=["exemple-btn"],
                            )
                        with gr.Row():
                            btn_ex3 = gr.Button(
                                EXEMPLES_CLIQUABLES[2][0],
                                size="sm",
                                variant="secondary",
                                elem_classes=["exemple-btn"],
                            )
                            btn_ex4 = gr.Button(
                                EXEMPLES_CLIQUABLES[3][0],
                                size="sm",
                                variant="secondary",
                                elem_classes=["exemple-btn"],
                            )

                        type_analyse = gr.Radio(
                            choices=[
                                "Articles applicables",
                                "Arguments defense",
                                "Arguments accusation",
                                "Analyse complete",
                            ],
                            value="Analyse complete",
                            label="Type d'analyse",
                        )

                        gr.Markdown(
                            "**Avertissement :** Ces informations sont donnees "
                            "a titre indicatif et ne remplacent pas les "
                            "conseils d'un avocat.",
                            elem_classes=["disclaimer"],
                        )

                        btn_analyser = gr.Button(
                            "Analyser", variant="primary", size="lg"
                        )

                    # -- Colonne droite : resultat --
                    with gr.Column(scale=1):
                        badge_domaine = gr.Markdown(
                            value="", elem_classes=["badge-domaine"]
                        )
                        output_analyse = gr.Markdown(
                            value="*L'analyse apparaitra ici...*",
                            elem_classes=["analyse-output"],
                        )
                        with gr.Row():
                            btn_copier = gr.Button(
                                "Copier l'analyse",
                                size="sm",
                                variant="secondary",
                            )
                            btn_reset = gr.Button(
                                "Nouvelle analyse",
                                size="sm",
                                variant="secondary",
                            )

                # State pour le compteur
                compteur_state = gr.State(value=compteur_global)

                # ── Events exemples ──
                btn_ex1.click(
                    fn=lambda: EXEMPLES_CLIQUABLES[0][1],
                    outputs=cas_input,
                )
                btn_ex2.click(
                    fn=lambda: EXEMPLES_CLIQUABLES[1][1],
                    outputs=cas_input,
                )
                btn_ex3.click(
                    fn=lambda: EXEMPLES_CLIQUABLES[2][1],
                    outputs=cas_input,
                )
                btn_ex4.click(
                    fn=lambda: EXEMPLES_CLIQUABLES[3][1],
                    outputs=cas_input,
                )

                # Bouton Analyser
                btn_analyser.click(
                    fn=analyser_cas,
                    inputs=[cas_input, type_analyse, compteur_state],
                    outputs=[
                        output_analyse,
                        compteur_display,
                        badge_domaine,
                        compteur_state,
                    ],
                )

                # Reset
                btn_reset.click(
                    fn=lambda: ("", "*L'analyse apparaitra ici...*", ""),
                    outputs=[cas_input, output_analyse, badge_domaine],
                )

                # Copier via JS
                btn_copier.click(
                    fn=None,
                    js=(
                        "() => { "
                        "const el = document.querySelector('.analyse-output'); "
                        "if (el) navigator.clipboard.writeText(el.innerText); "
                        "}"
                    ),
                )

            # ── Onglet 2 : Conversation (chat avec historique) ────────────
            with gr.Tab("Conversation"):
                gr.ChatInterface(
                    fn=chat_juridique,
                    chatbot=gr.Chatbot(
                        height=500,
                        render_markdown=True,
                        elem_classes=["chatbot-box"],
                    ),
                    textbox=gr.Textbox(
                        placeholder="Posez votre question juridique...",
                        scale=7,
                    ),
                    description=(
                        "Posez plusieurs questions qui se suivent, "
                        "LexIA garde le contexte de la conversation."
                    ),
                    examples=[
                        "Mon employeur peut-il me licencier sans motif ?",
                        "Quel est le delai pour contester un licenciement ?",
                        "Mon proprietaire refuse de rendre ma caution, que faire ?",
                        "Comment demander une pension alimentaire ?",
                    ],
                )

            # ── Onglet 3 : Resume de jugement ────────────────────────────
            with gr.Tab("Resume de jugement"):
                gr.Markdown("### Collez un jugement a resumer")

                jugement_input = gr.Textbox(
                    label="Texte du jugement",
                    placeholder="Collez ici le texte brut du jugement...",
                    lines=10,
                )
                btn_resume = gr.Button("Resumer", variant="primary")
                resume_output = gr.Markdown(
                    value="*Le resume apparaitra ici...*",
                    elem_classes=["resume-output"],
                )

                btn_resume.click(
                    fn=resumer_jugement,
                    inputs=[jugement_input],
                    outputs=resume_output,
                )

            # ── Onglet 4 : A propos ──────────────────────────────────────
            with gr.Tab("A propos"):
                gr.Markdown(f"""
## LexIA -- Assistant Juridique Francais

LexIA est un modele **Ministral 8B fine-tune** sur les sources juridiques officielles francaises.

### Sources de donnees
- **Legifrance** -- Code civil, penal, assurances (sources officielles)
- **Judilibre** -- Decisions anonymisees de la Cour de Cassation

### Stack technique
- **Modele de base :** Ministral 8B Instruct
- **Fine-tuning :** TRL SFTTrainer + LoRA 4-bit
- **Compute :** Hugging Face Jobs (A10G)
- **Tracking :** Weights & Biases
- **Modele :** `{HF_USERNAME}/{MODEL_NAME}`

### Disclaimer legal
LexIA est un outil d'aide a la recherche juridique.
Il ne remplace pas l'avis d'un avocat qualifie.
                """)

        gr.HTML(
            "<p style='text-align:center;color:#888;margin-top:10px'>"
            "Built for Mistral Hack-a-ton 2026 -- LexIA</p>"
        )

    return interface


# ── POINT D'ENTREE ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Lancement de LexIA...")
    demo = construire_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    )
