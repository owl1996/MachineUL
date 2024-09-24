# MachineUL
Ce dépôt contient des implémentations et des benchmarks de différentes techniques de Machine Unlearning et des techniques d'attaques de modèle (Membership Inference, etc.) pour évaluer leur efficacité.
Machine-UL/
├── README.md
├── LICENSE
├── requirements.txt  # Les dépendances Python
├── src/              # Code source des modèles et attaques
│   ├── models/       # Modèles ML
│   ├── attacks/      # Implémentations des attaques
│   └── unlearning/   # Implémentations des méthodes de unlearning
├── benchmarks/       # Scripts pour évaluer les performances des techniques
│   └── membership_inference.py
├── notebooks/        # Notebooks explicatifs pour tests et démonstrations
│   └── unlearning_example.ipynb
├── data/             # Données d'entraînement (ou exemples de données)
│   └── README.md     # Explication des jeux de données utilisés
└── docs/             # Documentation plus poussée du projet
