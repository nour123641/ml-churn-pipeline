"""Pipeline Prefect : orchestre les étapes du projet churn."""

import subprocess
from pathlib import Path

from prefect import flow, task, get_run_logger

VENV_DIR = Path("venv")
PYTHON_BIN = VENV_DIR / "bin" / "python"
MAIN_PY = Path("main.py")

# L'énoncé demande de ne viser que ces 3 fichiers
CODE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def _run(cmd):
    """Exécute une commande, affiche sa sortie, et lève une erreur si elle échoue."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Échec de la commande : {' '.join(cmd)}")


# ---------- Tasks : environnement et qualité du code ----------


@task(name="install-dependencies")
def install_dependencies():
    logger = get_run_logger()
    _run([str(PYTHON_BIN), "-m", "pip", "install", "-r", "requirements.txt"])
    logger.info("Dépendances installées.")


@task(name="format-code")
def format_code():
    logger = get_run_logger()
    _run([str(PYTHON_BIN), "-m", "black"] + CODE_FILES)
    logger.info("Formatage du code terminé.")


@task(name="quality-code")
def quality_code():
    logger = get_run_logger()
    _run([str(PYTHON_BIN), "-m", "flake8", "--max-line-length=100"] + CODE_FILES)
    logger.info("Qualité du code vérifiée.")


@task(name="security-code")
def security_code():
    logger = get_run_logger()
    _run([str(PYTHON_BIN), "-m", "bandit", "-ll", "-q"] + CODE_FILES)
    logger.info("Sécurité du code vérifiée.")


@task(name="unit-tests")
def unit_tests():
    logger = get_run_logger()
    _run([str(PYTHON_BIN), "-m", "pytest", "-q", "test_model_pipeline.py"])
    logger.info("Tests unitaires réussis.")


# ---------- Tasks : données et modèle ----------


@task(name="prepare-data")
def prepare_data():
    logger = get_run_logger()
    _run([str(PYTHON_BIN), str(MAIN_PY), "--prepare"])
    logger.info("Préparation des données terminée.")


@task(name="train-model")
def train_model():
    logger = get_run_logger()
    _run([str(PYTHON_BIN), str(MAIN_PY), "--train"])
    logger.info("Entraînement terminé.")


@task(name="save-model")
def save_model():
    logger = get_run_logger()
    _run([str(PYTHON_BIN), str(MAIN_PY), "--save"])
    logger.info("Modèle sauvegardé.")


@task(name="load-model")
def load_model():
    logger = get_run_logger()
    _run([str(PYTHON_BIN), str(MAIN_PY), "--load"])
    logger.info("Modèle chargé.")


@task(name="evaluate-model")
def evaluate_model():
    logger = get_run_logger()
    _run([str(PYTHON_BIN), str(MAIN_PY), "--evaluate"])
    logger.info("Évaluation terminée.")


# ---------- Flows ----------


@flow(name="code", log_prints=True)
def flow_code():
    install_dependencies()
    format_code()
    quality_code()
    security_code()
    unit_tests()
    print("Flow CODE terminé avec succès")


@flow(name="all", log_prints=True)
def flow_all():
    install_dependencies()
    format_code()
    quality_code()
    security_code()
    unit_tests()
    prepare_data()
    train_model()
    save_model()
    evaluate_model()
    print("Flow ALL terminé avec succès")


@flow(name="train", log_prints=True)
def flow_train():
    prepare_data()
    train_model()
    print("Flow TRAIN terminé avec succès")


@flow(name="evaluate", log_prints=True)
def flow_evaluate():
    load_model()
    evaluate_model()
    print("Flow EVALUATE terminé avec succès")


FLOWS = {
    "all": flow_all,
    "code": flow_code,
    "train": flow_train,
    "entrainement": flow_train,
    "evaluate": flow_evaluate,
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline Prefect")
    parser.add_argument("--flow", choices=FLOWS.keys(), required=True)
    args = parser.parse_args()
    FLOWS[args.flow]()
