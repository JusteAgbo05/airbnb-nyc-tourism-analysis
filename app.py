from __future__ import annotations

from pathlib import Path
import io
import base64

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------------------------------------------------------
# 1. Configuration générale
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Airbnb NYC | Tourism Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ASSETS_DIR = ROOT_DIR / "assets"
DATASET_FILENAME = "AB_NYC_2019.csv"
DATASET_SLUG = "dgomonov/new-york-city-airbnb-open-data"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Palette partagée avec le README et la présentation.
PETROL = "#123B3A"
TEAL = "#269B8A"
GOLD = "#D6A451"
INK = "#35434A"
MUTED = "#71817F"
PAPER = "#F7F8F5"


# -----------------------------------------------------------------------------
# 2. Style personnalisé
# -----------------------------------------------------------------------------

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
      .stApp {{ background: {PAPER}; color: {INK}; }}
      [data-testid="stHeader"] {{ background: rgba(247,248,245,0.88); }}
      [data-testid="stSidebar"] {{ background: #102F2E; }}
      [data-testid="stSidebar"] * {{ color: #F7F8F5 !important; }}
      [data-testid="stSidebar"] [data-testid="stRadio"] label {{
        padding: 0.45rem 0.55rem; border-radius: 9px; font-weight: 500;
      }}
      [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{ background: rgba(255,255,255,.08); }}
      h1, h2, h3 {{ font-family: 'Manrope', sans-serif !important; color: {PETROL}; letter-spacing: -0.035em; }}
      h1 {{ font-weight: 800 !important; }}
      .block-container {{ padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1440px; }}
      .hero {{
        background: linear-gradient(115deg, #123B3A 0%, #185B55 58%, #269B8A 100%);
        border-radius: 22px; padding: 2.2rem 2.4rem; color: white;
        box-shadow: 0 16px 40px rgba(18,59,58,.14); margin-bottom: 1.4rem;
        position: relative; overflow: hidden;
      }}
      .hero:after {{ content:''; position:absolute; width:280px; height:280px; right:-75px; top:-115px;
        border:1px solid rgba(255,255,255,.22); border-radius:50%; box-shadow:0 0 0 28px rgba(255,255,255,.04),0 0 0 58px rgba(255,255,255,.035); }}
      .hero-kicker {{ text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; color:#D6A451; }}
      .hero h1 {{ color:white !important; font-size:clamp(2rem,4vw,3.35rem); margin:.5rem 0 .6rem; line-height:1.06; }}
      .hero p {{ color:rgba(255,255,255,.82); max-width:760px; font-size:1.02rem; margin:0; }}
      .section-kicker {{ color:{TEAL}; text-transform:uppercase; letter-spacing:.14em; font-size:.7rem; font-weight:800; margin-bottom:.2rem; }}
      .section-title {{ font-family:'Manrope',sans-serif; color:{PETROL}; font-size:1.55rem; font-weight:800; margin:0 0 .2rem; }}
      .section-caption {{ color:{MUTED}; font-size:.92rem; margin-bottom:1.1rem; }}
      .metric-card {{ background:white; border:1px solid #E4EAE5; border-radius:16px; padding:1rem 1.15rem; min-height:112px; box-shadow:0 4px 14px rgba(18,59,58,.035); }}
      .metric-label {{ color:{MUTED}; font-size:.76rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }}
      .metric-value {{ color:{PETROL}; font-family:'Manrope',sans-serif; font-size:1.8rem; font-weight:800; margin-top:.28rem; line-height:1.2; }}
      .metric-help {{ color:{MUTED}; font-size:.76rem; margin-top:.2rem; }}
      .insight {{ background:#EAF3EF; border-left:4px solid {TEAL}; padding:.9rem 1rem; border-radius:0 12px 12px 0; color:#244A45; margin:.6rem 0 1rem; }}
      .warning {{ background:#FBF3E4; border-left:4px solid {GOLD}; padding:.9rem 1rem; border-radius:0 12px 12px 0; color:#66502B; margin:.6rem 0 1rem; }}
      .mini-label {{ color:{MUTED}; font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; font-weight:800; }}
      div[data-testid="stPlotlyChart"] {{ background:white; border:1px solid #E7ECE8; border-radius:16px; padding:.35rem; }}
      div[data-testid="stDataFrame"] {{ border:1px solid #E7ECE8; border-radius:12px; overflow:hidden; }}
      .footer {{ border-top:1px solid #DDE5DF; padding-top:1rem; margin-top:2.5rem; color:{MUTED}; font-size:.78rem; }}
      .stButton > button {{ background:{PETROL}; color:white; border:0; border-radius:10px; font-weight:700; padding:.55rem 1rem; }}
      .stButton > button:hover {{ background:{TEAL}; color:white; border:0; }}
      [data-testid="stFormSubmitButton"] button {{ background:{TEAL}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 3. Chargement des données : local, KaggleHub ou dépôt manuel
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_dataset(uploaded_bytes: bytes | None = None) -> pd.DataFrame:
    """Charge le CSV depuis le dossier local ou télécharge la version publique Kaggle."""
    # Si un fichier est importé depuis l'interface, il est prioritaire sur le fichier local.
    if uploaded_bytes is not None:
        return pd.read_csv(io.BytesIO(uploaded_bytes), low_memory=False)

    local_candidates = [
        RAW_DIR / DATASET_FILENAME,
        PROCESSED_DIR / DATASET_FILENAME,
        ROOT_DIR / DATASET_FILENAME,
    ]

    for candidate in local_candidates:
        if candidate.exists():
            return pd.read_csv(candidate, low_memory=False)

    # KaggleHub permet de récupérer le dataset public sans intégrer de jeton secret au projet.
    try:
        import kagglehub

        downloaded_dir = Path(kagglehub.dataset_download(DATASET_SLUG))
        matches = list(downloaded_dir.rglob(DATASET_FILENAME))
        if matches:
            return pd.read_csv(matches[0], low_memory=False)
    except Exception as exc:
        raise RuntimeError(
            "Le téléchargement automatique depuis Kaggle a échoué. "
            "Vérifiez la connexion Internet ou utilisez le champ d'import du CSV dans la barre latérale. "
            f"Détail technique : {exc}"
        ) from exc

    raise FileNotFoundError(
        f"{DATASET_FILENAME} est introuvable. Placez-le dans data/raw/ ou importez-le depuis la barre latérale."
    )


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Applique les règles essentielles du notebook pour l'exploration de l'application."""
    df = data.copy()

    # Normaliser les espaces des champs textuels sans inventer de catégories.
    text_columns = ["name", "host_name", "neighbourhood_group", "neighbourhood", "room_type"]
    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].apply(lambda value: value.strip() if isinstance(value, str) else value)

    # Les prix nuls ou négatifs ne sont pas exploitables pour une analyse tarifaire.
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.loc[df["price"].notna() & (df["price"] > 0)].copy()

    # Conversion explicite des variables numériques utilisées par les graphiques et le modèle.
    numeric_columns = [
        "latitude", "longitude", "minimum_nights", "number_of_reviews",
        "reviews_per_month", "calculated_host_listings_count", "availability_365",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Le notebook ne supprime pas les annonces simplement parce qu'elles n'ont pas d'avis.
    return df


@st.cache_data(show_spinner=False)
def prepare_data(uploaded_bytes: bytes | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_dataset(uploaded_bytes)
    clean = clean_dataset(raw)
    return raw, clean


# -----------------------------------------------------------------------------
# 4. Modèle de régression : même famille de variables que le notebook
# -----------------------------------------------------------------------------

CATEGORICAL_FEATURES = ["neighbourhood_group", "neighbourhood", "room_type"]
NUMERIC_FEATURES = [
    "latitude", "longitude", "minimum_nights", "number_of_reviews",
    "reviews_per_month", "calculated_host_listings_count", "availability_365", "has_reviews",
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def make_model_frame(data: pd.DataFrame) -> pd.DataFrame:
    features = data[ CATEGORICAL_FEATURES + NUMERIC_FEATURES[:-1] ].copy()
    reviews = pd.to_numeric(features["number_of_reviews"], errors="coerce")
    features["has_reviews"] = np.where(reviews.isna(), np.nan, (reviews > 0).astype(float))
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype(float)
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype("object")
    return features


@st.cache_resource(show_spinner="Entraînement et évaluation des modèles…")
def train_demo_model(data: pd.DataFrame):
    """Entraîne Ridge et Random Forest et les compare à la baseline du notebook."""
    X = make_model_frame(data)
    y = data["price"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    numeric_pipeline_ridge = Pipeline(steps=[
        ("imputation", SimpleImputer(strategy="median")),
        ("standardisation", StandardScaler()),
    ])
    numeric_pipeline_forest = Pipeline(steps=[
        ("imputation", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputation", SimpleImputer(strategy="most_frequent")),
        ("encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    def build_preprocessor(numeric_pipeline):
        return ColumnTransformer(transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ])

    ridge_model = Pipeline(steps=[
        ("preparation", build_preprocessor(numeric_pipeline_ridge)),
        ("model", Ridge(alpha=10.0)),
    ])
    forest_model = Pipeline(steps=[
        ("preparation", build_preprocessor(numeric_pipeline_forest)),
        ("model", RandomForestRegressor(
            n_estimators=60, max_depth=18, min_samples_leaf=3,
            max_features=0.8, random_state=42, n_jobs=-1,
        )),
    ])

    y_train_log = np.log1p(y_train)
    ridge_model.fit(X_train, y_train_log)
    forest_model.fit(X_train, y_train_log)

    predictions = {
        "Baseline — médiane": np.full(len(y_test), y_train.median(), dtype=float),
        "Régression Ridge": np.expm1(np.maximum(ridge_model.predict(X_test), 0)),
        "Forêt aléatoire": np.expm1(np.maximum(forest_model.predict(X_test), 0)),
    }

    rows = []
    for model_name, predicted_values in predictions.items():
        rows.append({
            "Modèle": model_name,
            "MAE (USD)": mean_absolute_error(y_test, predicted_values),
            "RMSE (USD)": np.sqrt(mean_squared_error(y_test, predicted_values)),
            "MedAE (USD)": median_absolute_error(y_test, predicted_values),
            "R²": r2_score(y_test, predicted_values),
        })

    comparison = pd.DataFrame(rows).sort_values("MAE (USD)").reset_index(drop=True)
    forest_metrics = comparison.loc[comparison["Modèle"] == "Forêt aléatoire"].iloc[0].to_dict()
    forest_metrics["test_size"] = len(y_test)
    return forest_model, comparison, forest_metrics


def predict_price(model, values: dict) -> float:
    row = pd.DataFrame([values], columns=MODEL_FEATURES)
    predicted_log = model.predict(row)[0]
    return float(np.expm1(max(predicted_log, 0)))


# -----------------------------------------------------------------------------
# 5. Composants d'interface
# -----------------------------------------------------------------------------

def render_project_banner():
    """Affiche la bannière SVG du projet, également utilisée dans le README."""
    banner_path = ASSETS_DIR / "banner.svg"
    if banner_path.exists():
        encoded = base64.b64encode(banner_path.read_bytes()).decode("utf-8")
        st.markdown(
            f'<img src="data:image/svg+xml;base64,{encoded}" '
            'style="width:100%;height:auto;border-radius:18px;margin-bottom:1rem;" />',
            unsafe_allow_html=True,
        )


def section_heading(kicker: str, title: str, caption: str | None = None):
    st.markdown(
        f'<div class="section-kicker">{kicker}</div><div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, help_text: str = ""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-help">{help_text}</div></div>',
        unsafe_allow_html=True,
    )


def usd(value: float) -> str:
    return f"${value:,.0f}"


def apply_plotly_theme(fig, height: int = 390):
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=55, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=INK),
        title_font=dict(family="Manrope, sans-serif", size=16, color=PETROL),
        legend_title_font=dict(size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E9EEEA", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E9EEEA", zeroline=False)
    return fig


# -----------------------------------------------------------------------------
# 6. Barre latérale et chargement initial
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("<div style='font-family:Manrope;font-size:1.1rem;font-weight:800;letter-spacing:-.03em'>AIRBNB <span style='color:#D6A451'>NYC</span></div>", unsafe_allow_html=True)
    st.caption("TOURISM DATA ANALYTICS · 2019")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Vue d'ensemble", "Explorer les données", "Qualité des données", "Marché touristique", "Analyse géographique", "Modèles & performances", "Prédire un prix", "À propos"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("**Source des données**")
    st.caption("New York City Airbnb Open Data · 2019")
    uploaded_file = st.file_uploader("Importer le CSV si nécessaire", type=["csv"])
    uploaded_bytes = uploaded_file.getvalue() if uploaded_file else None
    st.markdown("---")
    st.caption("Projet académique · Domaine Tourisme")
    st.caption("Vivien AGBO & ADANLAO Laurinda")

try:
    with st.spinner("Chargement et préparation des annonces Airbnb…"):
        df_raw, df = prepare_data(uploaded_bytes)
except Exception as error:
    st.markdown(
        '<div class="hero"><div class="hero-kicker">Tourism data analytics</div>'
        '<h1>Airbnb New York<br>2019</h1><p>Une exploration du marché des hébergements, de la qualité des données et de la prédiction des prix.</p></div>',
        unsafe_allow_html=True,
    )
    st.error(str(error))
    st.info("Pour lancer la démonstration, placez AB_NYC_2019.csv dans data/raw/ ou importez le fichier depuis la barre latérale.")
    st.stop()


# -----------------------------------------------------------------------------
# 7. Pages de l'application
# -----------------------------------------------------------------------------

if page == "Vue d'ensemble":
    render_project_banner()
    st.markdown(
        '<div class="hero"><div class="hero-kicker">Tourism data analytics · New York City · 2019</div>'
        '<h1>Comprendre le marché.<br>Explorer les données.</h1>'
        '<p>Une expérience interactive autour des annonces Airbnb de New York : qualité des données, dynamiques tarifaires et premier modèle de prédiction.</p></div>',
        unsafe_allow_html=True,
    )

    total = len(df)
    boroughs = df["neighbourhood_group"].nunique()
    median_price = df["price"].median()
    mean_price = df["price"].mean()
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Annonces analysées", f"{total:,}".replace(",", " "), "Après exclusion des prix non positifs")
    with c2: metric_card("Prix médian / nuit", usd(median_price), "Indicateur moins sensible aux extrêmes")
    with c3: metric_card("Prix moyen / nuit", usd(mean_price), "En dollars américains")
    with c4: metric_card("Arrondissements", str(boroughs), "Les cinq boroughs de New York")

    st.write("")
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        section_heading("01 · Offre", "Où se concentrent les annonces ?", "Répartition des annonces par grand arrondissement.")
        borough_counts = df["neighbourhood_group"].value_counts().rename_axis("Arrondissement").reset_index(name="Annonces")
        fig = px.bar(borough_counts.sort_values("Annonces"), x="Annonces", y="Arrondissement", orientation="h", text="Annonces", color_discrete_sequence=[TEAL])
        fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
        st.plotly_chart(apply_plotly_theme(fig, 340), use_container_width=True)
    with right:
        section_heading("02 · Hébergement", "Quels logements sont proposés ?", "Composition de l'offre selon le type d'hébergement.")
        room_counts = df["room_type"].value_counts().rename_axis("Type").reset_index(name="Annonces")
        fig = px.pie(room_counts, names="Type", values="Annonces", hole=.62, color_discrete_sequence=[PETROL, TEAL, GOLD])
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(apply_plotly_theme(fig, 340), use_container_width=True)

    borough_share = df["neighbourhood_group"].value_counts(normalize=True)
    top_two_share = borough_share.iloc[:2].sum() * 100
    st.markdown(
        f'<div class="insight"><b>À retenir.</b> Les deux arrondissements les plus représentés regroupent '
        f'<b>{top_two_share:.2f} %</b> des annonces du fichier nettoyé. Le marché observé est donc fortement concentré géographiquement.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="warning"><b>Repère temporel.</b> Ces données décrivent les annonces de 2019. Elles ne constituent pas une mesure du marché Airbnb actuel.</div>', unsafe_allow_html=True)

elif page == "Explorer les données":
    section_heading("DATA EXPLORER", "Explorer les annonces", "Filtrez les logements et consultez les caractéristiques enregistrées dans le dataset.")
    f1, f2, f3 = st.columns(3)
    with f1:
        borough_options = ["Tous"] + sorted(df["neighbourhood_group"].dropna().unique().tolist())
        selected_borough = st.selectbox("Arrondissement", borough_options)
    with f2:
        room_options = ["Tous"] + sorted(df["room_type"].dropna().unique().tolist())
        selected_room = st.selectbox("Type d'hébergement", room_options)
    with f3:
        max_price = int(min(df["price"].quantile(.99), 2000))
        selected_max_price = st.slider("Prix maximal affiché (USD)", 0, max(100, max_price), max(100, max_price), step=25)

    filtered = df.copy()
    if selected_borough != "Tous": filtered = filtered[filtered["neighbourhood_group"] == selected_borough]
    if selected_room != "Tous": filtered = filtered[filtered["room_type"] == selected_room]
    filtered = filtered[filtered["price"] <= selected_max_price]

    a, b, c = st.columns(3)
    with a: metric_card("Annonces filtrées", f"{len(filtered):,}".replace(",", " "), "Après application des filtres")
    with b: metric_card("Prix médian", usd(filtered["price"].median()) if not filtered.empty else "—", "Sur la sélection courante")
    with c: metric_card("Quartiers représentés", str(filtered["neighbourhood"].nunique()) if not filtered.empty else "0", "Nombre de quartiers distincts")

    st.write("")
    if filtered.empty:
        st.warning("Aucune annonce ne correspond à ces filtres. Modifiez les critères pour afficher des résultats.")
    else:
        fig = px.histogram(filtered, x="price", color="room_type", nbins=55, marginal="box", opacity=.78,
                           color_discrete_sequence=[PETROL, TEAL, GOLD], title="Distribution des prix dans la sélection")
        st.plotly_chart(apply_plotly_theme(fig, 400), use_container_width=True)
        display_columns = [c for c in ["name", "neighbourhood_group", "neighbourhood", "room_type", "price", "minimum_nights", "number_of_reviews", "availability_365"] if c in filtered.columns]
        st.dataframe(filtered[display_columns].sort_values("price", ascending=False), use_container_width=True, hide_index=True, height=360)
        csv = filtered[display_columns].to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger la sélection (CSV)", csv, "airbnb_nyc_selection.csv", "text/csv")

elif page == "Qualité des données":
    section_heading("DATA QUALITY", "Évaluer la qualité des données", "Comparer le fichier initial au jeu utilisé dans l'application et visualiser les valeurs manquantes.")
    removed = len(df_raw) - len(df)
    retained = len(df) / len(df_raw) * 100 if len(df_raw) else 0
    q1, q2, q3, q4 = st.columns(4)
    with q1: metric_card("Lignes initiales", f"{len(df_raw):,}".replace(",", " "), "Fichier source")
    with q2: metric_card("Lignes après filtre", f"{len(df):,}".replace(",", " "), "Prix strictement positif")
    with q3: metric_card("Lignes écartées", f"{removed:,}".replace(",", " "), "Prix nul, négatif ou non numérique")
    with q4: metric_card("Taux conservé", f"{retained:.2f} %", "Part des lignes utilisables")

    missing = pd.DataFrame({
        "Variable": df_raw.columns,
        "Manquants": df_raw.isna().sum().values,
        "Part (%)": (df_raw.isna().mean().values * 100),
    }).sort_values("Manquants", ascending=False)
    missing = missing[missing["Manquants"] > 0]
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        section_heading("VALEURS ABSENTES", "Où se trouvent les données manquantes ?")
        if not missing.empty:
            fig = px.bar(missing.sort_values("Part (%)"), x="Part (%)", y="Variable", orientation="h",
                         color_discrete_sequence=[TEAL], text=missing.sort_values("Part (%)")["Part (%)"].map(lambda x: f"{x:.1f}%"))
            fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(apply_plotly_theme(fig, 390), use_container_width=True)
    with right:
        section_heading("DIAGNOSTIC", "Valeurs manquantes par variable")
        st.dataframe(missing.style.format({"Part (%)": "{:.2f}%"}), use_container_width=True, hide_index=True, height=390)

    st.markdown('<div class="insight"><b>Interprétation.</b> Les champs liés aux avis peuvent être absents lorsqu’une annonce n’a encore reçu aucun avis. Il ne faut donc pas remplacer systématiquement ces absences par une valeur arbitraire : leur signification dépend du contexte métier.</div>', unsafe_allow_html=True)
    st.caption("Cette page applique dans l'application le filtre tarifaire essentiel du notebook. Les autres diagnostics détaillés (doublons, valeurs atypiques et contrôles de cohérence) restent documentés dans le notebook du projet.")

elif page == "Marché touristique":
    section_heading("MARKET INSIGHTS", "Lire les dynamiques tarifaires", "Comparer les prix affichés selon l'arrondissement et le type d'hébergement.")
    p1, p2 = st.columns(2)
    with p1:
        order_borough = df.groupby("neighbourhood_group")["price"].median().sort_values(ascending=False).index.tolist()
        fig = px.box(df[df["price"] <= df["price"].quantile(.99)], x="neighbourhood_group", y="price", category_orders={"neighbourhood_group": order_borough}, color="neighbourhood_group", color_discrete_sequence=[PETROL, TEAL, GOLD, "#83B8AA", "#B7C9C0"], title="Prix par nuit selon l'arrondissement")
        fig.update_layout(showlegend=False)
        st.plotly_chart(apply_plotly_theme(fig, 420), use_container_width=True)
    with p2:
        order_room = df.groupby("room_type")["price"].median().sort_values(ascending=False).index.tolist()
        fig = px.box(df[df["price"] <= df["price"].quantile(.99)], x="room_type", y="price", category_orders={"room_type": order_room}, color="room_type", color_discrete_sequence=[PETROL, TEAL, GOLD], title="Prix par nuit selon le type de logement")
        fig.update_layout(showlegend=False)
        st.plotly_chart(apply_plotly_theme(fig, 420), use_container_width=True)

    borough_stats = df.groupby("neighbourhood_group")["price"].agg(annonces="count", median="median", mean="mean").sort_values("median", ascending=False).reset_index()
    room_stats = df.groupby("room_type")["price"].agg(annonces="count", median="median", mean="mean").sort_values("median", ascending=False).reset_index()
    t1, t2 = st.columns(2)
    with t1:
        section_heading("COMPARAISON", "Indicateurs par arrondissement")
        st.dataframe(borough_stats.rename(columns={"neighbourhood_group":"Arrondissement", "annonces":"Annonces", "median":"Médiane (USD)", "mean":"Moyenne (USD)"}).style.format({"Médiane (USD)":"${:,.2f}", "Moyenne (USD)":"${:,.2f}"}), use_container_width=True, hide_index=True)
    with t2:
        section_heading("COMPARAISON", "Indicateurs par hébergement")
        st.dataframe(room_stats.rename(columns={"room_type":"Type", "annonces":"Annonces", "median":"Médiane (USD)", "mean":"Moyenne (USD)"}).style.format({"Médiane (USD)":"${:,.2f}", "Moyenne (USD)":"${:,.2f}"}), use_container_width=True, hide_index=True)

    top_borough = borough_stats.iloc[0]
    top_room = room_stats.iloc[0]
    st.markdown(
        f'<div class="insight"><b>Lecture métier.</b> Dans ce fichier de 2019, {top_borough["neighbourhood_group"]} présente la médiane la plus élevée '
        f'({usd(top_borough["median"])} par nuit). Le type « {top_room["room_type"]} » présente la médiane la plus élevée parmi les catégories d’hébergement. '
        'Ces écarts sont descriptifs et ne démontrent pas une relation de causalité.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="warning"><b>Précaution de lecture.</b> Les graphiques de boîtes sont limités visuellement au 99e percentile pour rester lisibles. Les observations ne sont pas supprimées du jeu de données utilisé pour les indicateurs.</div>', unsafe_allow_html=True)

elif page == "Analyse géographique":
    section_heading("GEOGRAPHIC VIEW", "Explorer la répartition spatiale", "Chaque point représente une annonce disposant de coordonnées géographiques.")
    geo = df.dropna(subset=["latitude", "longitude"]).copy()
    if len(geo) > 18000:
        geo = geo.sample(18000, random_state=42)
    fig = px.scatter_mapbox(
        geo, lat="latitude", lon="longitude", color="neighbourhood_group",
        hover_name="neighbourhood", hover_data={"price":":$,.0f", "room_type":True, "latitude":False, "longitude":False},
        zoom=9, center={"lat":40.72,"lon":-73.95}, height=650,
        color_discrete_sequence=[PETROL, TEAL, GOLD, "#83B8AA", "#B7C9C0"],
        title="Localisation des annonces Airbnb à New York",
    )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=50,b=0), legend_title="Arrondissement")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight"><b>Interprétation.</b> La carte permet d’observer la concentration spatiale des annonces et leur présence dans les différents arrondissements. Elle ne représente ni le nombre de réservations ni la fréquentation touristique réelle.</div>', unsafe_allow_html=True)
    st.caption(f"{len(geo):,} points affichés au maximum. Si le jeu contient davantage d'annonces géolocalisées, un échantillon reproductible est utilisé pour préserver la fluidité.")

elif page == "Modèles & performances":
    section_heading("MODEL EVALUATION", "Comparer les approches de régression", "Même jeu de test, mêmes variables : la comparaison permet de mesurer l'apport des modèles par rapport à une référence simple.")
    try:
        model, model_comparison, forest_metrics = train_demo_model(df)
    except Exception as error:
        st.error(f"Impossible d'entraîner les modèles : {error}")
        st.stop()

    best_row = model_comparison.iloc[0]
    baseline_mae = float(model_comparison.loc[model_comparison["Modèle"] == "Baseline — médiane", "MAE (USD)"].iloc[0])
    improvement = (1 - float(best_row["MAE (USD)"]) / baseline_mae) * 100

    k1, k2, k3 = st.columns(3)
    with k1: metric_card("Meilleure MAE", usd(float(best_row["MAE (USD)"])), str(best_row["Modèle"]))
    with k2: metric_card("Gain vs baseline", f"{improvement:.1f} %", "Réduction de la MAE")
    with k3: metric_card("R² forêt", f'{forest_metrics["R²"]:.3f}', "Part de variabilité expliquée, selon R²")

    left, right = st.columns([1.1, .9], gap="large")
    with left:
        section_heading("TABLEAU COMPARATIF", "Résultats sur le jeu de test")
        st.dataframe(model_comparison.style.format({"MAE (USD)":"${:,.2f}", "RMSE (USD)":"${:,.2f}", "MedAE (USD)":"${:,.2f}", "R²":"{:.3f}"}), use_container_width=True, hide_index=True)
    with right:
        section_heading("ERREUR ABSOLUE", "MAE par modèle")
        chart = px.bar(model_comparison.sort_values("MAE (USD)", ascending=True), x="MAE (USD)", y="Modèle", orientation="h", text=model_comparison.sort_values("MAE (USD)", ascending=True)["MAE (USD)"].map(lambda x: f"${x:,.2f}"), color_discrete_sequence=[TEAL])
        chart.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(apply_plotly_theme(chart, 320), use_container_width=True)

    st.markdown(f'<div class="insight"><b>Interprétation.</b> {best_row["Modèle"]} obtient la MAE la plus faible sur cette séparation entraînement/test. La réduction de MAE par rapport à la baseline est de {improvement:.1f} %. Ce résultat indique un gain sur cet échantillon, pas une garantie de performance sur des annonces actuelles.</div>', unsafe_allow_html=True)
    st.markdown('<div class="warning"><b>Limite.</b> Le R² de la forêt aléatoire reste modeste dans cette expérience. Les erreurs importantes, les prix extrêmes et les variables absentes du dataset limitent la précision de l’estimation.</div>', unsafe_allow_html=True)

elif page == "Prédire un prix":
    section_heading("MACHINE LEARNING", "Estimer le prix d'une annonce", "Un premier essai de régression supervisée, fondé sur la forêt aléatoire du notebook.")
    st.markdown('<div class="warning"><b>Usage pédagogique.</b> Le modèle est entraîné sur des annonces de 2019. Son estimation ne constitue pas un conseil tarifaire actuel et ne garantit pas le prix réellement payé ou obtenu.</div>', unsafe_allow_html=True)
    try:
        model, model_comparison, metrics = train_demo_model(df)
    except Exception as error:
        st.error(f"Impossible d'entraîner le modèle : {error}")
        st.stop()

    m1, m2, m3 = st.columns(3)

    with m1: metric_card("MAE de test", usd(metrics["MAE (USD)"]), "Erreur absolue moyenne")
    with m2: metric_card("R² de test", f'{metrics["R²"]:.3f}', "Pouvoir explicatif limité")
    with m3: metric_card("Annonces de test", f'{metrics["test_size"]:,}'.replace(",", " "), "Échantillon de test séparé")

    boroughs = sorted(df["neighbourhood_group"].dropna().unique().tolist())
    rooms = sorted(df["room_type"].dropna().unique().tolist())
    neighborhoods_by_borough = {
        borough: sorted(df.loc[df["neighbourhood_group"] == borough, "neighbourhood"].dropna().unique().tolist())
        for borough in boroughs
    }

    with st.form("prediction_form"):
        st.markdown("#### 1. Localisation et hébergement")
        c1, c2, c3 = st.columns(3)
        with c1: borough = st.selectbox("Arrondissement", boroughs)
        with c2:
            neighborhood_options = neighborhoods_by_borough.get(borough, [])
            neighborhood = st.selectbox("Quartier", neighborhood_options)
        with c3: room_type = st.selectbox("Type de logement", rooms)

        st.markdown("#### 2. Caractéristiques de l'annonce")
        c4, c5, c6 = st.columns(3)
        with c4: latitude = st.number_input("Latitude", min_value=40.45, max_value=40.95, value=40.72, step=0.001, format="%.5f")
        with c5: longitude = st.number_input("Longitude", min_value=-74.30, max_value=-73.65, value=-73.95, step=0.001, format="%.5f")
        with c6: minimum_nights = st.number_input("Séjour minimum (nuits)", min_value=1, max_value=365, value=2, step=1)

        c7, c8, c9 = st.columns(3)
        with c7: number_of_reviews = st.number_input("Nombre total d'avis", min_value=0, max_value=1000, value=10, step=1)
        with c8: reviews_per_month = st.number_input("Avis par mois", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
        with c9: host_listings = st.number_input("Annonces de l'hôte", min_value=1, max_value=100, value=1, step=1)
        availability = st.slider("Disponibilité annuelle (jours)", 0, 365, 180)
        submitted = st.form_submit_button("Estimer le prix par nuit", use_container_width=True)

    if submitted:
        input_values = {
            "neighbourhood_group": borough,
            "neighbourhood": neighborhood,
            "room_type": room_type,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "minimum_nights": float(minimum_nights),
            "number_of_reviews": float(number_of_reviews),
            "reviews_per_month": float(reviews_per_month),
            "calculated_host_listings_count": float(host_listings),
            "availability_365": float(availability),
            "has_reviews": float(number_of_reviews > 0),
        }
        estimate = predict_price(model, input_values)
        st.markdown(
            f'<div class="hero"><div class="hero-kicker">Résultat de la simulation</div>'
            f'<h1>{usd(estimate)} <span style="font-size:1.2rem;font-weight:600">/ nuit</span></h1>'
            f'<p>Estimation indicative pour un logement de type {room_type.lower()} situé dans {neighborhood}, {borough}.</p></div>',
            unsafe_allow_html=True,
        )
        st.caption("Cette estimation provient d'un modèle entraîné sur les annonces de 2019. Elle doit être interprétée avec prudence, notamment pour les prix extrêmes et les informations absentes du dataset.")

elif page == "À propos":
    render_project_banner()
    st.markdown(
        '<div class="hero"><div class="hero-kicker">Projet académique · Domaine tourisme</div>'
        '<h1>Airbnb NYC<br>Tourism Analytics</h1>'
        '<p>Du diagnostic de qualité des données à un premier essai de prédiction du prix des hébergements à New York.</p></div>',
        unsafe_allow_html=True,
    )
    section_heading("LE PROJET", "Une démarche complète de Data Science")
    st.markdown("""
    Ce projet étudie le jeu **New York City Airbnb Open Data (2019)** selon trois axes :

    - **Nettoyage :** identifier les valeurs manquantes, les incohérences et les prix non exploitables, puis justifier les traitements.
    - **Analyse exploratoire :** décrire la répartition géographique de l'offre, les types d'hébergement et les écarts de prix.
    - **Modélisation :** comparer une baseline, une régression Ridge et une forêt aléatoire pour estimer le prix affiché par nuit.

    **Auteurs :** Vivien AGBO et ADANLAO Laurinda  
    **Domaine :** Tourisme  
    **Jeu de données :** New York City Airbnb Open Data — 2019
    """)
    section_heading("MÉTHODOLOGIE", "Ce qu'il faut garder à l'esprit")
    st.markdown("""
    - Le dataset décrit des annonces de 2019 ; il ne reflète pas nécessairement le marché actuel.
    - Le prix correspond au prix affiché par nuit, et non au coût total payé par un voyageur.
    - La disponibilité ne doit pas être assimilée à un taux d'occupation.
    - Les associations statistiques et l'importance des variables ne prouvent pas une causalité.
    - Le modèle de régression constitue une expérimentation pédagogique et non un outil de tarification prêt à l'emploi.
    """)
    st.markdown(f"**Dépôt GitHub :** [airbnb-nyc-tourism-analysis](https://github.com/JusteAgbo05/airbnb-nyc-tourism-analysis)")


# -----------------------------------------------------------------------------
# 8. Pied de page
# -----------------------------------------------------------------------------

st.markdown(
    '<div class="footer">AIRBNB NYC · TOURISM DATA ANALYTICS · 2019 &nbsp;|&nbsp; '
    'Projet académique réalisé par Vivien AGBO &amp; ADANLAO Laurinda</div>',
    unsafe_allow_html=True,
)
