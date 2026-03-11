## TARDIS ##

# Description du Projet
Ce projet a été réalisé pour le SNCF Data Analysis Service. L'objectif est d'analyser les données historiques des retards de trains et de développer un modèle prédictif pour améliorer l'expérience des voyageurs.
+2

# Structure du Dépôt 

tardis_eda.ipynb : [À toi : Que fais-tu dans ce notebook ? (Nettoyage, analyse...)].

tardis_model.ipynb : Entraînement et évaluation du modèle de régression.

tardis_dashboard.py : Application interactive pour visualiser les données et prédire les retards.

requirements.txt : Liste des dépendances nécessaires.

model.joblib (ou .pkl) : Le modèle entraîné prêt à l'emploi.

# Installation et Utilisation
# Installation des dépendances :
pip install -r requirements.txt

# Lancement du Dashboard :
[À toi : Quelle est la commande spécifique pour lancer un script Streamlit ?].


# Méthodologie 
Nettoyage : Suppression des doublons et gestion des valeurs manquantes.
+1


Modèle : Utilisation d'un algorithme de [À toi : Quel type de modèle vas-tu utiliser ?] pour prédire le retard en minutes.

## TARDIS : Predicting the Unpredictable ##

## Description du Projet
Ce projet a été réalisé pour le **SNCF Data Analysis Service**[cite: 8]. L'objectif est d'analyser les données historiques des retards de trains pour identifier des patterns et construire un modèle de régression capable de prédire les futurs retards[cite: 9, 65]. Les résultats sont présentés via une application web interactive.

## Structure du Dépôt 
| Fichier | Description |
| :--- | :--- |
| `tardis_eda.ipynb` | Nettoyage des données, analyse exploratoire et feature engineering[cite: 17, 18]. |
| `cleaned_dataset.csv` | Jeu de données nettoyé exporté après l'étape d'EDA. |
| `tardis_model.ipynb` | Entraînement du modèle de régression, évaluation (RMSE, R²) et justification des choix[cite: 73, 74]. |
| `model.joblib` | Sauvegarde du modèle entraîné pour l'intégration dans le dashboard[cite: 79]. |
| `tardis_dashboard.py` | Application Streamlit incluant les visualisations et l'interface de prédiction[cite: 20, 91]. |
| `requirements.txt` | Liste des bibliothèques nécessaires (pandas, scikit-learn, streamlit, etc.). |
| `README.md` | Documentation du projet, instructions d'installation et d'utilisation[cite: 102]. |

## Installation et Configuration
1. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt

## Utilisation
# Préparation : Exécuter les notebooks tardis_eda.ipynb puis tardis_model.ipynb pour générer les fichiers nécessaires.

# Lancer le Dashboard :
streamlit run tardis_dashboard.py

## Fonctionnalités du Dashboard 
Statistiques clés : Affichage des moyennes de retards et du taux de ponctualité.
Visualisations : Graphiques de distribution et de comparaison des retards.
Interface de prédiction : Saisie des paramètres du trajet pour obtenir une estimation du retard en minutes.