"""
Monitoring du job HF — check toutes les 10 minutes.
Affiche le status et les derniers logs si erreur.
"""
import time
import sys
from datetime import datetime
from huggingface_hub import HfApi

JOB_ID = sys.argv[1] if len(sys.argv) > 1 else "69a44efb5672f7593677057c"
INTERVAL = 600  # 10 minutes

api = HfApi()

print(f"Monitoring job {JOB_ID} toutes les {INTERVAL // 60} minutes...")
print("=" * 60)

while True:
    try:
        job = api.inspect_job(job_id=JOB_ID)
        stage = job.status.stage
        message = job.status.message or ""
        now = datetime.now().strftime("%H:%M:%S")

        print(f"[{now}] Status: {stage} {('— ' + message) if message else ''}")

        if stage == "COMPLETED":
            print("\nJob termine avec succes !")
            print(f"URL: {job.url}")
            break

        if stage == "ERROR":
            print(f"\nERREUR detectee : {message}")
            print("\nRecuperation des logs...")
            try:
                logs = api.fetch_job_logs(job_id=JOB_ID)
                log_lines = []
                for line in logs:
                    log_lines.append(str(line))
                # Afficher les 80 dernieres lignes
                for l in log_lines[-80:]:
                    print(l)
            except Exception as e:
                print(f"Impossible de recuperer les logs: {e}")
            break

        if stage in ("CANCELLED", "CANCELLING"):
            print("\nJob annule.")
            break

    except Exception as e:
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] Erreur de check: {e}")

    time.sleep(INTERVAL)
