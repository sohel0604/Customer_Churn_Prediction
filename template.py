import os

folders = [
    "src/components",
    "artifacts"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

files = [
    "src/__init__.py",
    "src/logger.py",
    "src/exception.py",
    "src/utils.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "main.py",
    "setup.py"
    
]

for file in files:
    if not os.path.exists(file):
        with open(file, "w") as f:
            pass
