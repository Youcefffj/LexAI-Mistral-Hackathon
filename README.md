# ⚖️ LexIA — Assistant Juridique Français

> Fine-tuned Mistral model for French legal analysis
> **Mistral Hack-a-ton 2026** | Track : Fine-tuning (W&B)

## 🎯 Résumé du projet

LexIA est un assistant juridique fine-tuné sur Ministral 8B, capable d'analyser
des cas juridiques complexes en citant les articles exacts du droit français.

## 🔗 Liens

- **Démo live :** https://huggingface.co/spaces/Youcefffj/lexia
- **Modèle :** https://huggingface.co/mistral-hackaton-2026/lexia-ministral-8b
- **Dataset :** https://huggingface.co/datasets/Youcefffj/lexia-french-legal
- **W&B Run :** https://wandb.ai/youcefffj-mistralhack/youcefffj-mistralhack

## 📊 Sources de données

| Source | Type | Volume |
|--------|------|--------|
| Légifrance (erdal/legifrance) | Articles de loi officiels | ~1500 articles |
| Exemples synthétiques | Cas types rédigés | ~10 exemples |

## 📈 Résultats du fine-tuning

| Métrique | Valeur |
|----------|--------|
| Modèle de base | Ministral 8B Instruct |
| Epochs | 3 |
| Train loss final | 0.5066 |
| Eval loss final | 0.4449 |
| Temps d'entraînement | ~2h20 |
| GPU | A10G (HF Jobs) |

## 🛠️ Stack

- **Fine-tuning :** TRL SFTTrainer + LoRA (r=16, 4-bit quantization)
- **Compute :** Hugging Face Jobs A10G
- **Tracking :** Weights & Biases
- **Demo :** Gradio sur HF Spaces

## 🚀 Installation

```bash
git clone https://github.com/Youcefffj/LexAI-Mistral-Hackathon
cd LexAI-Mistral-Hackathon
pip install -r requirements.txt
cp .env.example .env  # Remplir les clés API
python3 app/app.py
```

## ⚠️ Disclaimer

LexIA est un outil de recherche juridique. Il ne remplace pas un avocat.
