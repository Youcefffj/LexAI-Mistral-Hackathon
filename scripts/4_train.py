"""
=============================================================================
LEXIA — Script 4 : Fine-tuning Ministral 8B avec TRL SFT + LoRA
=============================================================================
Ce script fine-tune le modèle de base Ministral 8B sur le dataset juridique
LexIA. Il utilise :
- TRL SFTTrainer : framework de fine-tuning de Hugging Face
- LoRA : technique d'adaptation à faible rang (économise la VRAM)
- Quantization 4-bit : réduit la mémoire GPU nécessaire
- W&B : tracking des métriques d'entraînement en temps réel

ENTRÉE  : Dataset sur HF Hub ou local dans data/processed/
SORTIE  : Modèle fine-tuné sur HF Hub dans l'orga mistral-hackaton-2026

USAGE :
  python3 scripts/4_train.py                    # Configuration par défaut
  python3 scripts/4_train.py --small            # Modèle 3B (plus rapide)
  python3 scripts/4_train.py --dry-run          # Test sans entraîner
=============================================================================
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

HF_TOKEN     = os.getenv("HF_TOKEN")
HF_USERNAME  = os.getenv("HF_USERNAME", "Youcefffj")
BASE_MODEL   = os.getenv("BASE_MODEL", "mistralai/Ministral-8B-instruct")
DATASET_NAME = os.getenv("HF_DATASET_NAME", "lexia-french-legal")
MODEL_NAME   = os.getenv("HF_MODEL_NAME", "lexia-ministral-8b")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "youcefffj-mistralhack")

# ─── ENTRAÎNEMENT ─────────────────────────────────────────────────────────────

def lancer_entrainement(args):
    """
    Fonction principale d'entraînement.
    Charge le modèle, configure LoRA, lance l'entraînement, sauvegarde et push.
    """
    import torch
    import wandb
    from datasets import load_from_disk, load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from trl import SFTTrainer, SFTConfig
    from peft import LoraConfig
    from huggingface_hub import login

    print("=" * 60)
    print("⚖️  LEXIA — Fine-tuning TRL SFT")
    print("=" * 60)

    # Authentification Hugging Face
    if HF_TOKEN:
        login(token=HF_TOKEN)
        print("✅ Connecté à Hugging Face")

    # Initialiser W&B pour tracker l'entraînement
    wandb.init(
        project=WANDB_PROJECT,
        name=f"lexia-sft-{MODEL_NAME}",
        config={
            "model": args.model or BASE_MODEL,
            "dataset": DATASET_NAME,
            "lora_r": 16,
            "lora_alpha": 32,
            "epochs": 3,
        }
    )
    print(f"✅ W&B initialisé : projet '{WANDB_PROJECT}'")

    # ─── CHARGEMENT DU DATASET ────────────────────────────────────────────────

    print(f"\n📦 Chargement du dataset...")

    chemin_local = Path("data/processed/lexia_dataset")
    if chemin_local.exists():
        # Préférer le dataset local (plus rapide)
        dataset = load_from_disk(str(chemin_local))
        print(f"✅ Dataset local chargé")
    else:
        # Sinon charger depuis HF Hub
        dataset = load_dataset(f"{HF_USERNAME}/{DATASET_NAME}")
        print(f"✅ Dataset HF Hub chargé")

    print(f"   Train : {len(dataset['train'])} exemples")
    print(f"   Eval  : {len(dataset['eval'])} exemples")

    # ─── CHARGEMENT DU MODÈLE ─────────────────────────────────────────────────

    # Choisir le modèle selon les arguments
    if args.small:
        model_id = "mistralai/Ministral-3b-instruct"
        print(f"\n🤖 Mode rapide : Ministral 3B (adapté pour faible VRAM)")
    else:
        model_id = args.model or BASE_MODEL
        print(f"\n🤖 Chargement : {model_id}")

    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Padding à droite pour les modèles causaux
    print("✅ Tokenizer chargé")

    # Configurer la quantization 4-bit pour réduire l'utilisation de VRAM
    config_quantization = BitsAndBytesConfig(
        load_in_4bit=True,                    # Charger en 4-bit
        bnb_4bit_quant_type="nf4",           # Type de quantization NF4
        bnb_4bit_compute_dtype=torch.float16, # Calculs en float16
        bnb_4bit_use_double_quant=True,       # Double quantization
    )

    # Charger le modèle avec quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=config_quantization,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"✅ Modèle {model_id} chargé en 4-bit")

    # ─── CONFIGURATION LORA ───────────────────────────────────────────────────

    config_lora = LoraConfig(
        r=16,              # Rang des matrices LoRA
        lora_alpha=32,     # Facteur de scaling = 2 × r
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print("✅ Configuration LoRA prête")

    # ─── CONFIGURATION DE L'ENTRAÎNEMENT ──────────────────────────────────────

    dossier_sortie = f"./output/{MODEL_NAME}"

    args_entrainement = SFTConfig(
        output_dir=dossier_sortie,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        fp16=True,
        max_seq_length=2048,
        dataset_text_field="messages",
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        push_to_hub=True,
        hub_model_id=f"mistral-hackaton-2026/{MODEL_NAME}",
        report_to="wandb",
    )
    print("✅ Configuration d'entraînement prête")

    # ─── LANCEMENT DE L'ENTRAÎNEMENT ──────────────────────────────────────────

    print("\n🏋️  Initialisation du SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        args=args_entrainement,
        peft_config=config_lora,
    )

    model.print_trainable_parameters()

    print(f"\n🚀 Démarrage du fine-tuning...")
    print(f"   Modèle     : {model_id}")
    print(f"   Dataset    : {len(dataset['train'])} exemples")
    print(f"   Epochs     : {args_entrainement.num_train_epochs}")
    print(f"   W&B projet : {WANDB_PROJECT}")
    print()

    trainer.train()

    # ─── SAUVEGARDE ET PUSH ───────────────────────────────────────────────────

    print("\n💾 Sauvegarde du modèle fine-tuné...")
    trainer.save_model(dossier_sortie)
    tokenizer.save_pretrained(dossier_sortie)
    print(f"✅ Modèle sauvegardé localement → {dossier_sortie}")

    print(f"🚀 Push sur HF Hub...")
    trainer.push_to_hub()
    print(f"✅ Modèle disponible : https://huggingface.co/mistral-hackaton-2026/{MODEL_NAME}")

    wandb.finish()
    print("✅ Run W&B terminé")
    print("\n🎉 Fine-tuning terminé avec succès !")

    return trainer

# ─── POINT D'ENTRÉE ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LexIA Fine-tuning")
    parser.add_argument(
        "--model", type=str, default=None,
        help="ID du modèle de base sur HF Hub"
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Utiliser Ministral 3B (plus rapide, moins de VRAM)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Tester la configuration sans lancer l'entraînement"
    )
    args = parser.parse_args()

    if args.dry_run:
        print("🧪 DRY RUN — Vérification de la configuration")
        print(f"  Modèle    : {args.model or BASE_MODEL}")
        print(f"  Dataset   : {HF_USERNAME}/{DATASET_NAME}")
        print(f"  Output    : mistral-hackaton-2026/{MODEL_NAME}")
        print(f"  W&B       : {WANDB_PROJECT}")
        print("✅ Configuration valide")
    else:
        lancer_entrainement(args)
