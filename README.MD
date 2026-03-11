# TARDIS : Predicting the Unpredictable

## Description du Projet
Ce projet a été réalisé pour le **SNCF Data Analysis Service**. L'objectif est d'analyser les données historiques des retards de trains afin d'identifier des patterns et de construire un modèle de régression capable de prédire les futurs retards. Les résultats sont présentés via une application web interactive.

## Structure du Dépôt
* **tardis_eda.ipynb** : Notebook dédié au nettoyage des données et à l'analyse exploratoire.
* **tardis_model.ipynb** : Entraînement du modèle de régression et sauvegarde du modèle.
* **tardis_dashboard.py** : Interface utilisateur développée avec Streamlit.
* **model.joblib** : Le modèle entraîné (généré après exécution du notebook).

---

## Guide d'Installation et Lancemen

### 1. Préparation de l'environnement virtuel

**Sur Linux / macOS :**
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
