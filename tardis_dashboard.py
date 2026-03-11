import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Config

st.set_page_config(
    page_title="TARDIS - SNCF Delay Prediction",
    page_icon="🚆",
    layout="wide",
)

st.title("🚆 TARDIS - Train Delay Prediction")
st.markdown("SNCF Data Analysis Service")


# Load

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.csv")
    df.columns = df.columns.str.strip()
    return df


@st.cache_resource
def load_model():
    model = joblib.load("model.joblib")
    return model

@st.cache_resource
def load_encoders():
    return joblib.load("encoders.joblib")

df = load_data()
model = load_model()
encoders = load_encoders()

#  Clean data

delay_column = "Retard moyen de tous les trains à l'arrivée"

df[delay_column] = (
    df[delay_column]
    .astype(str)
    .str.replace(",", ".", regex=False)
)

df[delay_column] = pd.to_numeric(
    df[delay_column],
    errors="coerce",
)

df = df.dropna(subset=[delay_column])


# Sidebar pour filtrer
st.sidebar.header("Filters")

selected_year = st.sidebar.selectbox(
    "Select Year",
    sorted(df["Annee"].dropna().unique()),
)

filtered_df = df[df["Annee"] == selected_year]

st.header("Statistiques")

col1, col2, col3 = st.columns(3)

avg_delay = filtered_df[delay_column].mean()
total_trips = len(filtered_df)
punctuality_rate = (
    (filtered_df[delay_column] <= 5).mean() * 100
)

col1.metric("Retard moyen", f"{avg_delay:.2f} min")
col2.metric("Voyage total", total_trips)
col3.metric("Taux de ponctualité (≤5 min)", f"{punctuality_rate:.1f}%")

st.divider()

st.header("Analyse des retards")

col_graph1, col_graph2 = st.columns(2)

# Histogramme de retard
with col_graph1:
    st.subheader("Retard")

    fig1, ax1 = plt.subplots(figsize=(3, 2))
    sns.histplot(filtered_df[delay_column], bins=30, kde=True, ax=ax1)

    ax1.set_xlabel("Retard (min)", fontsize=8)
    ax1.set_ylabel("Nombre de retard", fontsize=8)
    ax1.tick_params(axis="both", labelsize=6)

    st.pyplot(fig1, use_container_width=False)


# Graphique en barre des retards par service
with col_graph2:
    st.subheader("Retard par service")

    service_delay = (
        filtered_df.groupby("Service")[delay_column]
        .mean()
    )

    fig2, ax2 = plt.subplots(figsize=(3, 1.4))
    service_delay.plot(kind="bar", ax=ax2)

    ax2.set_xlabel("Service", fontsize=8)
    ax2.set_ylabel("Retard moyen (min)", fontsize=8)
    ax2.tick_params(axis="both", labelsize=6)

    st.pyplot(fig2, use_container_width=False)

# Section pred
st.divider()

st.header("Prédiction de retard")

col_pred1, col_pred2 = st.columns(2)

# Récupérer les listes de gares connues
gares_depart = sorted(encoders["depart"].classes_)
gares_arrivee = sorted(encoders["arrivee"].classes_)
services = sorted(encoders["service"].classes_)

# Inputs
with col_pred1:
    gare_dep = st.selectbox("Gare de départ", gares_depart)
    gare_arr = st.selectbox("Gare d'arrivée", gares_arrivee)
    service = st.selectbox("Service", services)

with col_pred2:
    mois = st.slider("Mois", 1, 12, 6)
    annee = st.selectbox("Année", sorted(df["Annee"].dropna().unique()))
    duree = st.number_input("Durée moyenne du trajet (min)", min_value=10, max_value=500, value=120)

if st.button("🚆 Prédire le retard"):
    # Encoder les valeurs
    dep_encoded = encoders["depart"].transform([gare_dep])[0]
    arr_encoded = encoders["arrivee"].transform([gare_arr])[0]
    service_encoded = encoders["service"].transform([service])[0]


    nb_circ = int(df["Nombre de circulations prévues"].median())

    # Stocker les valeurs demandés par l'utilisateur
    input_data = pd.DataFrame([{
        "Gare de départ": dep_encoded,
        "Gare d'arrivée": arr_encoded,
        "Mois": mois,
        "Service": service_encoded,
        "Durée moyenne du trajet": duree,
        "Annee": annee,
        "Nombre de circulations prévues": nb_circ,
    }])

    prediction = model.predict(input_data)[0]

    # Affichage du résultat
    if prediction <= 5:
        st.success(f"Retard prédit : **{prediction:.1f} minutes** (train ponctuel)")
    elif prediction <= 15:
        st.warning(f"Retard prédit : **{prediction:.1f} minutes** (léger retard)")
    else:
        st.error(f"Retard prédit : **{prediction:.1f} minutes** (retard important)")


# Footer
st.markdown("---")
st.markdown("TARDIS Project - Lois, Marius, Antoine")
