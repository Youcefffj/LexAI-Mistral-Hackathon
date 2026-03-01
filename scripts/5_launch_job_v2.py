"""
=============================================================================
LEXIA — Script 5 : Lancement du fine-tuning v2 via HF Jobs
=============================================================================
Ce script lance un job de continued fine-tuning sur Hugging Face Jobs
en utilisant l'API Python huggingface_hub.

MODELE BASE  : mistral-hackaton-2026/lexia-ministral-8b
DATASET      : Youcefffj/lexia-french-legal-v2
MODELE CIBLE : mistral-hackaton-2026/lexia-ministral-8b-v2
HARDWARE     : A10G Large (24 GB VRAM)

CONFIGURATION :
  - LoRA r=16, alpha=32, dropout=0.05
  - 4-bit quantization (QLoRA)
  - 2 epochs, lr=1e-4, cosine scheduler
  - Suivi W&B
=============================================================================
"""

import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

# ── CONFIGURATION ────────────────────────────────────────────────────────────

HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "youcefffj-mistralhack")

# Modele de base (v1 fine-tune)
BASE_MODEL = "mistral-hackaton-2026/lexia-ministral-8b"

# Dataset enrichi v2
DATASET = "Youcefffj/lexia-french-legal-v2"

# Modele de sortie v2
OUTPUT_MODEL = "mistral-hackaton-2026/lexia-ministral-8b-v2"

# Image Docker avec PyTorch + CUDA
DOCKER_IMAGE = "nvcr.io/nvidia/pytorch:24.01-py3"

# Hardware
HARDWARE = "a10g-large"

# ── SCRIPT DE TRAINING ──────────────────────────────────────────────────────

# Commande bash qui sera executee dans le conteneur Docker
training_command = (
    "pip install --no-cache-dir "
    "typing_extensions>=4.12.0 "
    "trl==0.12.2 "
    "transformers==4.46.3 "
    "peft==0.13.2 "
    "datasets==2.21.0 "
    "accelerate==0.34.2 "
    "bitsandbytes==0.43.3 "
    "wandb==0.19.1 "
    "&& trl sft "
    f"--model_name_or_path {BASE_MODEL} "
    f"--dataset_name {DATASET} "
    "--dataset_test_split eval "
    "--output_dir ./output/lexia-ministral-8b-v2 "
    f"--hub_model_id {OUTPUT_MODEL} "
    "--push_to_hub "
    "--num_train_epochs 2 "
    "--per_device_train_batch_size 1 "
    "--gradient_accumulation_steps 8 "
    "--learning_rate 1e-4 "
    "--lr_scheduler_type cosine "
    "--warmup_ratio 0.05 "
    "--bf16 "
    "--gradient_checkpointing "
    "--optim paged_adamw_32bit "
    "--use_peft "
    "--load_in_4bit "
    "--lora_r 16 "
    "--lora_alpha 32 "
    "--lora_dropout 0.05 "
    "--lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj "
    "--dataset_text_field messages "
    "--max_seq_length 1024 "
    "--logging_steps 25 "
    "--eval_strategy steps "
    "--eval_steps 100 "
    "--save_steps 200 "
    "--report_to wandb"
)

# ── LANCEMENT ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("LEXIA — Lancement du fine-tuning v2")
    print("=" * 60)
    print(f"\nModele base   : {BASE_MODEL}")
    print(f"Dataset       : {DATASET}")
    print(f"Modele cible  : {OUTPUT_MODEL}")
    print(f"Hardware      : {HARDWARE}")
    print(f"Image Docker  : {DOCKER_IMAGE}")
    print(f"Epochs        : 2")
    print(f"LoRA          : r=16, alpha=32")
    print(f"W&B project   : {WANDB_PROJECT}")
    print()

    # Verifier les variables d'environnement
    if not HF_TOKEN:
        print("ERREUR : HF_TOKEN non configure dans .env")
        exit(1)
    if not WANDB_API_KEY:
        print("ATTENTION : WANDB_API_KEY non configure — le tracking W&B sera desactive")

    # Variables d'environnement pour le job
    env_vars = {
        "WANDB_PROJECT": WANDB_PROJECT,
    }

    # Secrets (ne sont pas loggues)
    secrets = {
        "HF_TOKEN": HF_TOKEN,
        "WANDB_API_KEY": WANDB_API_KEY or "",
    }

    print("Lancement du job HF via l'API Python...")
    print()

    try:
        api = HfApi(token=HF_TOKEN)

        job_info = api.run_job(
            image=DOCKER_IMAGE,
            command=["bash", "-c", training_command],
            env=env_vars,
            secrets=secrets,
            flavor=HARDWARE,
            timeout="4h",
        )

        print()
        print("=" * 60)
        print("Job lance avec succes !")
        print(f"Job ID        : {job_info.id}")
        print(f"Status        : {job_info.status}")
        print(f"URL           : {job_info.url}")
        print(f"W&B           : https://wandb.ai/{WANDB_PROJECT}")
        print("=" * 60)

    except Exception as e:
        print(f"ERREUR lors du lancement : {e}")
        print()
        print("Causes possibles :")
        print("  - Credits HF Jobs insuffisants")
        print("  - Token HF sans droits suffisants")
        print("  - Hardware non disponible")
        print()
        print("Verifiez sur https://huggingface.co/settings/billing")
