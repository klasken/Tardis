# TARDIS : Predicting the Unpredictable

## Description du Projet
Ce projet a été réalisé pour le **SNCF Data Analysis Service**. L'objectif est d'analyser les données historiques des retards de trains pour identifier des patterns et construire un modèle de régression capable de prédire les futurs retards. Les résultats sont présentés via une application web interactive.

## Structure du Dépôt 

| Fichier | Description |
| :--- | :--- |
| `tardis_eda.ipynb` | Nettoyage des données, analyse exploratoire et feature engineering. |
| `cleaned_dataset.csv` | Jeu de données nettoyé exporté après l'étape d'EDA. |
| `tardis_model.ipynb` | Entraînement du modèle de régression, évaluation (RMSE, R²) et justification des choix. |
| `model.joblib` | Sauvegarde du modèle entraîné pour l'intégration dans le dashboard. |
| `tardis_dashboard.py` | Application Streamlit incluant les visualisations et l'interface de prédiction. |
| `requirements.txt` | Liste des bibliothèques nécessaires (pandas, scikit-learn, streamlit, etc.). |
| `README.md` | Documentation du projet, instructions d'installation et d'utilisation. |

## Méthodologie 
* **Nettoyage** : Suppression des doublons et gestion des valeurs manquantes.
* **Modèle** : Utilisation d'un modèle de régression pour prédire le retard en minutes.

## Installation et Configuration
1. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
