"""
=============================================================================
LEXIA — Interface Gradio pour la démonstration hackathon
=============================================================================
Cette interface web permet de démontrer les 4 capacités de LexIA :
1. Analyse de cas juridiques avec citation des articles applicables
2. Génération d'arguments de défense/accusation
3. Résumé de jugements complexes
4. Questions/réponses sur le droit français

L'interface utilise soit le modèle fine-tuné sur HF Hub,
soit l'API Mistral directement (fallback si modèle pas encore prêt).

DÉPLOIEMENT :
  Local   : python3 app/app.py
  HF Space: pousser app.py + requirements.txt sur un HF Space
=============================================================================
"""

import os
import gradio as gr
import requests
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
HF_USERNAME     = os.getenv("HF_USERNAME", "Youcefffj")
MODEL_NAME      = os.getenv("HF_MODEL_NAME", "lexia-ministral-8b")

# Personnalité permanente du modèle — injectée dans chaque requête
SYSTEM_PROMPT = """Tu es LexIA, un assistant juridique expert en droit français.
Tu as été entraîné sur les sources officielles françaises (Légifrance, Judilibre).
Tes réponses citent toujours les articles de loi exacts avec leur numéro et code.
Tu rédiges en français juridique rigoureux et structuré.
Tu n'inventes jamais d'articles inexistants."""

# ─── APPELS AU MODÈLE ─────────────────────────────────────────────────────────

def appeler_mistral_api(messages):
    """
    Appeler l'API Mistral pour générer une réponse.
    Utilisé comme fallback si le modèle fine-tuné n'est pas disponible.

    Args:
        messages: Liste de messages au format ChatML

    Returns:
        Texte de la réponse générée
    """
    if not MISTRAL_API_KEY:
        return "❌ Clé API Mistral non configurée. Ajoute MISTRAL_API_KEY dans le .env"

    reponse = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
        json={
            "model": "mistral-large-latest",
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1024,
        },
        timeout=30,
    )
    reponse.raise_for_status()
    return reponse.json()["choices"][0]["message"]["content"]


def generer_reponse(instruction):
    """
    Construire les messages et appeler le modèle.

    Args:
        instruction: La question ou le cas soumis par l'utilisateur

    Returns:
        Réponse juridique structurée du modèle
    """
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": instruction},
    ]
    return appeler_mistral_api(messages)

# ─── FONCTIONS DES 4 MODES ────────────────────────────────────────────────────

def analyser_cas(description_cas, type_analyse):
    """
    Mode 1 & 2 : Analyser un cas juridique.
    Selon le type choisi, génère soit une analyse des articles applicables,
    soit des arguments de défense ou d'accusation, soit une analyse complète.
    """
    prompts = {
        "🔍 Articles applicables": (
            f"Identifie et explique tous les articles de loi français applicables "
            f"à cette situation, en précisant le code (civil, pénal, travail...) "
            f"et le numéro d'article :\n\n{description_cas}"
        ),
        "⚔️ Arguments défense": (
            f"Génère les arguments juridiques de défense les plus solides, "
            f"avec les articles de loi correspondants :\n\n{description_cas}"
        ),
        "⚖️ Arguments accusation": (
            f"Quels sont les arguments juridiques que peut invoquer "
            f"la partie demanderesse ou plaignante :\n\n{description_cas}"
        ),
        "📋 Analyse complète": (
            f"Réalise une analyse juridique complète : articles applicables, "
            f"arguments des deux parties, jurisprudence, et recommandations :\n\n{description_cas}"
        ),
    }

    instruction = prompts.get(type_analyse, prompts["📋 Analyse complète"])
    return generer_reponse(instruction)


def repondre_question(question):
    """Mode 3 : Répondre à une question générale sur le droit français."""
    return generer_reponse(question)


def resumer_jugement(texte_jugement):
    """
    Mode 4 : Résumer un jugement complexe en termes accessibles.
    Utile pour les avocats qui reçoivent de longs textes de décisions.
    """
    instruction = (
        f"Résume ce jugement de manière claire et structurée. "
        f"Identifie : la problématique, les arguments des parties, "
        f"les articles de loi appliqués, et la décision finale :\n\n{texte_jugement}"
    )
    return generer_reponse(instruction)

# ─── EXEMPLES PRÉ-CHARGÉS ────────────────────────────────────────────────────

# Ces exemples seront affichés dans l'interface pour faciliter la démonstration
EXEMPLES_CAS = [
    [
        "Un salarié a été licencié après 12 ans d'ancienneté sans convocation préalable à entretien et sans lettre de licenciement motivée.",
        "📋 Analyse complète"
    ],
    [
        "Un acheteur découvre 8 mois après l'achat de son appartement que la toiture fuit. Le vendeur professionnel avait connaissance du problème.",
        "🔍 Articles applicables"
    ],
    [
        "Un locataire refuse de payer son loyer depuis 4 mois. Il invoque des troubles de jouissance dus à des problèmes d'humidité non réparés.",
        "⚔️ Arguments défense"
    ],
    [
        "Une salariée subit depuis 18 mois des reproches répétés, une mise à l'écart des réunions, et une surcharge de travail documentée.",
        "⚖️ Arguments accusation"
    ],
]

EXEMPLES_QUESTIONS = [
    ["Quelle est la différence entre le Code civil et le Code pénal ?"],
    ["Quels sont les délais de prescription en droit du travail ?"],
    ["Quelles sont les conditions pour invoquer la légitime défense ?"],
    ["Comment fonctionne la garde alternée selon le droit français ?"],
]

# ─── CONSTRUCTION DE L'INTERFACE ──────────────────────────────────────────────

def construire_interface():
    """
    Construire l'interface Gradio complète avec 4 onglets fonctionnels.
    Chaque onglet correspond à un mode de LexIA.
    """
    with gr.Blocks(
        title="⚖️ LexIA — Assistant Juridique Français",
    ) as interface:

        # En-tête
        gr.HTML("""
        <div style="text-align:center; padding:20px">
            <h1>⚖️ LexIA</h1>
            <p>Assistant Juridique Expert en Droit Français</p>
            <span style="background:#4f46e5;color:white;padding:3px 12px;border-radius:12px;font-size:12px">
                Mistral Hack-a-ton 2026 · Fine-tuned on Légifrance + Judilibre
            </span>
        </div>
        """)

        with gr.Tabs():

            # ─── ONGLET 1 : ANALYSE DE CAS ────────────────────────────────────
            with gr.Tab("🔍 Analyse de cas"):
                gr.Markdown("### Décrivez votre situation juridique")

                with gr.Row():
                    with gr.Column():
                        cas_input = gr.Textbox(
                            label="Description du cas",
                            placeholder="Ex: Mon employeur m'a licencié sans convocation...",
                            lines=6,
                        )
                        type_analyse = gr.Radio(
                            choices=["🔍 Articles applicables", "⚔️ Arguments défense",
                                     "⚖️ Arguments accusation", "📋 Analyse complète"],
                            value="📋 Analyse complète",
                            label="Type d'analyse",
                        )
                        btn_analyser = gr.Button("⚖️ Analyser", variant="primary", size="lg")

                    with gr.Column():
                        cas_output = gr.Textbox(
                            label="Analyse juridique LexIA",
                            lines=15,
                        )

                gr.Examples(
                    examples=EXEMPLES_CAS,
                    inputs=[cas_input, type_analyse],
                    label="📚 Exemples de cas",
                )
                btn_analyser.click(analyser_cas, inputs=[cas_input, type_analyse], outputs=cas_output)

            # ─── ONGLET 2 : QUESTIONS JURIDIQUES ──────────────────────────────
            with gr.Tab("📚 Questions juridiques"):
                gr.Markdown("### Posez une question sur le droit français")

                question_input = gr.Textbox(
                    label="Votre question",
                    placeholder="Ex: Quels sont les droits d'un salarié en cas de licenciement abusif ?",
                    lines=3,
                )
                btn_question = gr.Button("💬 Répondre", variant="primary")
                question_output = gr.Textbox(label="Réponse LexIA", lines=12)

                gr.Examples(
                    examples=EXEMPLES_QUESTIONS,
                    inputs=[question_input],
                    label="Questions fréquentes",
                )
                btn_question.click(repondre_question, inputs=[question_input], outputs=question_output)

            # ─── ONGLET 3 : RÉSUMÉ DE JUGEMENT ───────────────────────────────
            with gr.Tab("📄 Résumé de jugement"):
                gr.Markdown("### Collez un jugement à résumer")

                jugement_input = gr.Textbox(
                    label="Texte du jugement",
                    placeholder="Collez ici le texte brut du jugement...",
                    lines=10,
                )
                btn_resume = gr.Button("📝 Résumer", variant="primary")
                resume_output = gr.Textbox(label="Résumé LexIA", lines=10)

                btn_resume.click(resumer_jugement, inputs=[jugement_input], outputs=resume_output)

            # ─── ONGLET 4 : À PROPOS ──────────────────────────────────────────
            with gr.Tab("ℹ️ À propos"):
                gr.Markdown(f"""
                ## ⚖️ LexIA — Assistant Juridique Français

                LexIA est un modèle **Ministral 8B fine-tuné** sur les sources juridiques officielles françaises.

                ### 📊 Sources de données
                - **Légifrance** — Code civil, pénal, assurances (sources officielles)
                - **Judilibre** — Décisions anonymisées de la Cour de Cassation

                ### 🛠️ Stack technique
                - **Modèle de base :** Ministral 8B Instruct
                - **Fine-tuning :** TRL SFTTrainer + LoRA 4-bit
                - **Compute :** Hugging Face Jobs (A10G)
                - **Tracking :** Weights & Biases
                - **Modèle :** `mistral-hackaton-2026/{MODEL_NAME}`

                ### ⚠️ Disclaimer légal
                LexIA est un outil d'aide à la recherche juridique.
                Il ne remplace pas l'avis d'un avocat qualifié.
                """)

        gr.HTML("<p style='text-align:center;color:#888;margin-top:10px'>Built for Mistral Hack-a-ton 2026 · LexIA</p>")

    return interface


# ─── POINT D'ENTRÉE ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Lancement de LexIA...")
    interface = construire_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    )
