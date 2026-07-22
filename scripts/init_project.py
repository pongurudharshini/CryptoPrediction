"""
Project Initialization Script.
Creates the complete clean architecture directory structure for the Crypto Prediction pipeline.
"""

from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define root directory relative to this script
ROOT_DIR = Path(__file__).resolve().parent.parent

DIRECTORIES = [
    "app/api/endpoints",
    "app/core",
    "app/services",
    "frontend/pages",
    "frontend/components",
    "models/baselines",
    "models/tft",
    "models/pso",
    "data_pipeline/ingest",
    "data_pipeline/process",
    "data_pipeline/features",
    "sentiment/fetchers",
    "sentiment/analyzers",
    "training",
    "evaluation",
    "config",
    "database",
    "notebooks",
    "tests/unit",
    "tests/integration",
    "docker",
    "logs",
    "data/raw",
    "data/processed",
    "saved_models",
]

def create_structure() -> None:
    """Creates directory structure and __init__.py files where appropriate."""
    logging.info(f"Initializing project structure at: {ROOT_DIR}")
    
    for relative_path in DIRECTORIES:
        dir_path = ROOT_DIR / relative_path
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {relative_path}")
        
        # Add __init__.py to Python package directories
        if not relative_path.startswith(("notebooks", "docker", "logs", "data", "saved_models")):
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                logging.info(f"Created __init__.py in: {relative_path}")

    # Create root level __init__.py files for main modules
    root_modules = ["app", "models", "data_pipeline", "sentiment", "training", "evaluation", "config", "database"]
    for mod in root_modules:
        init_file = ROOT_DIR / mod / "__init__.py"
        if not init_file.exists():
            init_file.touch()

    logging.info("Project directory initialization complete!")

if __name__ == "__main__":
    create_structure()