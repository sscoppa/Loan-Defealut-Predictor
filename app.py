import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# --------------------------------------------------------
# PAGE CONFIG (must be first Streamlit command)
# --------------------------------------------------------
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    layout="centered",
)

# --------------------------------------------------------
# GLOBAL STYLE / CUSTOM CSS (ASU-themed)
# --------------------------------------------------------
st.markdown(
    """
    <style>
    /* Overall page background */
    .stApp {
        background: radial-gradient(circle at top, #8C1D40 0, #111827 40%, #020617 80%);
        color: #e5e7eb;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif;
    }

    /* Center content and limit width */
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1 {
        font-weight: 750 !important;
        letter-spacing: 0.04em;
    }

    /* Section cards */
    .section-card {
        background: rgba(15, 23, 42, 0.95);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 18px 40px rgba(0,0,0,0.45);
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }

    .section-subtitle {
        font-size: 0.82rem;
        color: #9ca3af;
        margin-bottom: 0.75rem;
    }

    /* Button styling - ASU maroon & gold */
    .stButton>button {
        background: linear-gradient(135deg, #8C1D40, #FFC627);
        color: white;
        border: none;
        padding: 0.55rem 1.4rem;
        border-radius: 999px;
        font-weight: 600;
        letter-spacing: 0.04em;
        box-shadow: 0 12px 30px rgba(140,29,64,0.6);
    }

    .stButton>button:hover {
        filter: brightness(1.05);
        box-shadow: 0 16px 40px rgba(255,198,39,0.75);
    }

    /* Risk pill */
    .risk-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.18rem 0.7rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .risk-pill.high {
        background: rgba(239, 68, 68, 0.15);
        color: #fecaca;
        border: 1px solid rgba(248, 113, 113, 0.6);
    }

    .risk-pill.medium {
        background: rgba(251, 191, 36, 0.12);
        color: #fef3c7;
        border: 1px solid rgba(252, 211, 77, 0.6);
    }

    .risk-pill.low {
        background: rgba(34, 197, 94, 0.15);
        color: #bbf7d0;
        border: 1px solid rgba(74, 222, 128, 0.6);
    }

    .prob-label {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-bottom: 0.1rem;
    }

    .prob-value {
        font-weight: 600;
        margin-top: 0.15rem;
        font-size: 0.95rem;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------
# LOAD MODEL
# --------------------------------------------------------
MODEL_FILENAME = "loan_default_xgb.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILENAME)

model = load_model()

# Get the column structure the pipeline expects
try:
    REQUIRED_COLUMNS = list(model.feature_names_in_)
except AttributeError:
    REQUIRED_COLUMNS = [
        "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
        "CNT_CHILDREN", "CNT_FAM_MEMBERS", "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "DAYS_EMPLOYED", "DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "REGION_POPULATION_RELATIVE",
    ]

# --------------------------------------------------------
# HEADER (with your name + program)
# --------------------------------------------------------
st.markdown(
    """
    <div class="section-card" style="margin-bottom:1.6rem;">
        <div style="font-size:0.78rem; letter-spacing:0.18em; text-transform:uppercase; color:#FCD34D; margin-bottom:0.25rem;">
            MS-AIB · CIS 508 · Term Project
        </div>
        <h1>Loan Default Risk Predictor</h1>
        <p style="font-size:0.9rem; color:#e5e7eb; max-width:720px; margin-top:0.35rem;">
            Interactive demo of a machine learning model trained on the Home Credit Default Risk dataset.
            Adjust the borrower profile below to see how the estimated probability of default changes.
        </p>
        <p style="font-size:0.82rem; color:#d1d5db; margin-top:0.6rem;">
            <strong>Created by:</strong> Shane Scoppa · MS in Artificial Intelligence in Business, W. P. Carey School of Business
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------
# APPLICANT PROFILE
# --------------------------------------------------------
st.markdown(
    """
    <div class="section-card">
        <div class="section-title">Applicant Profile</div>
        <div class="section-subtitle">
            Core demographic and household information about the applicant.
        </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    NAME_CONTRACT_TYPE = st.selectbox(
        "Contract Type",
        ["Cash loans", "Revolving loans"],
        index=0,
    )
    CODE_GENDER = st.selectbox("Gender", ["M", "F"], index=0)
    FLAG_OWN_CAR = st.selectbox("Owns a Car?", ["Y", "N"], index=0)
    FLAG_OWN_REALTY = st.selectbox("Owns Real Estate?", ["Y", "N"], index=0)
    NAME_EDUCATION_TYPE = st.selectbox(
        "Education Level",
        [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree",
        ],
        index=1,
    )

with col2:
    NAME_FAMILY_STATUS = st.selectbox(
        "Family Status",
        [
            "Single / not married",
            "Married",
            "Civil marriage",
            "Separated",
            "Widow",
        ],
        index=0,
    )
    NAME_HOUSING_TYPE = st.selectbox(
        "Housing Type",
        [
            "House / apartment",
            "With parents",
            "Municipal apartment",
            "Rented apartment",
            "Office apartment",
            "Co-op apartment",
        ],
        index=0,
    )
    CNT_CHILDREN = st.number_input("Number of Children", 0, 20, 0, 1)
    CNT_FAM_MEMBERS = st.number_input(
        "Number of People You Live With",
        min_value=1.0,
        max_value=20.0,
        value=2.0,
        step=1.0,
        help="Total number of people in your household (including you).",
    )

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# FINANCIAL INFORMATION
# --------------------------------------------------------
st.markdown(
    """
    <div class="section-card">
        <div class="section-title">Financial Information</div>
        <div class="section-subtitle">
            Income, loan size, and payment structure for the current application.
        </div>
    """,
    unsafe_allow_html=True,
)

col3, col4 = st.columns(2)

with col3:
    AMT_INCOME_TOTAL = st.number_input(
        "Annual Income (USD)",
        0.0, 5_000_000.0, 150_000.0, 5_000.0,
    )
    AMT_ANNUITY = st.number_input(
        "Annuity Amount (per period, USD)",
        0.0, 500_000.0, 25_000.0, 1_000.0,
    )

with col4:
    AMT_CREDIT = st.number_input(
        "Total Loan Amount (USD)",
        0.0, 5_000_000.0, 500_000.0, 10_000.0,
        help="Total principal amount you are borrowing.",
    )
    AMT_GOODS_PRICE = st.number_input(
        "Goods Price (if applicable, USD)",
        0.0, 5_000_000.0, 500_000.0, 10_000.0,
        help="If the loan is for a specific purchase (e.g., car), enter that price.",
    )

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# TIMELINE + EXTERNAL SCORES (friendlier)
# --------------------------------------------------------
st.markdown(
    """
    <div class="section-card">
        <div class="section-title">History & External Scores</div>
        <div class="section-subtitle">
            Approximate timing and overall external risk signals associated with the applicant.
        </div>
    """,
    unsafe_allow_html=True,
)

col5, col6 = st.columns(2)

with col5:
    age_years = st.number_input("Applicant Age (years)", 18, 90, 35, 1)
    years_employed = st.number_input("Years Employed", 0, 60, 5, 1)
    years_since_registration = st.number_input(
        "Years Since Registration", 0, 60, 5, 1
    )
    years_since_id_publish = st.number_input(
        "Years Since ID Issued", 0, 60, 3, 1
    )

with col6:
    credit_profile = st.select_slider(
        "Overall Credit Profile",
        options=[
            "Very weak / limited",
            "Weak",
            "Average",
            "Good",
            "Excellent",
        ],
        value="Average",
        help="A simplified way to represent the applicant's overall creditworthiness.",
    )

    location_type = st.selectbox(
        "Where do you live?",
        [
            "Rural / small town",
            "Suburban area",
            "Urban / city",
            "Major metro / very dense",
        ],
        index=1,
        help="General type of area where the applicant lives.",
    )

# Convert to negative days (dataset convention)
DAYS_BIRTH = -int(age_years * 365)
DAYS_EMPLOYED = -int(years_employed * 365)
DAYS_REGISTRATION = -int(years_since_registration * 365)
DAYS_ID_PUBLISH = -int(years_since_id_publish * 365)

# Map friendly controls -> model features for external scores
credit_mapping = {
    "Very weak / limited": (0.10, 0.10, 0.10),
    "Weak": (0.25, 0.25, 0.25),
    "Average": (0.50, 0.50, 0.50),
    "Good": (0.70, 0.70, 0.70),
    "Excellent": (0.90, 0.90, 0.90),
}
EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3 = credit_mapping[credit_profile]

# Map location type -> region population relative
region_mapping = {
    "Rural / small town": 0.02,
    "Suburban area": 0.15,
    "Urban / city": 0.40,
    "Major metro / very dense": 0.70,
}
REGION_POPULATION_RELATIVE = region_mapping[location_type]

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# BUILD INPUT ROW
# --------------------------------------------------------
base_input = {
    "NAME_CONTRACT_TYPE": NAME_CONTRACT_TYPE,
    "CODE_GENDER": CODE_GENDER,
    "FLAG_OWN_CAR": FLAG_OWN_CAR,
    "FLAG_OWN_REALTY": FLAG_OWN_REALTY,
    "CNT_CHILDREN": CNT_CHILDREN,
    "CNT_FAM_MEMBERS": CNT_FAM_MEMBERS,
    "NAME_EDUCATION_TYPE": NAME_EDUCATION_TYPE,
    "NAME_FAMILY_STATUS": NAME_FAMILY_STATUS,
    "NAME_HOUSING_TYPE": NAME_HOUSING_TYPE,
    "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
    "AMT_CREDIT": AMT_CREDIT,
    "AMT_ANNUITY": AMT_ANNUITY,
    "AMT_GOODS_PRICE": AMT_GOODS_PRICE,
    "DAYS_EMPLOYED": DAYS_EMPLOYED,
    "DAYS_BIRTH": DAYS_BIRTH,
    "DAYS_REGISTRATION": DAYS_REGISTRATION,
    "DAYS_ID_PUBLISH": DAYS_ID_PUBLISH,
    "EXT_SOURCE_1": EXT_SOURCE_1,
    "EXT_SOURCE_2": EXT_SOURCE_2,
    "EXT_SOURCE_3": EXT_SOURCE_3,
    "REGION_POPULATION_RELATIVE": REGION_POPULATION_RELATIVE,
}

input_df = pd.DataFrame([base_input])

# Ensure all required columns exist
for col in REQUIRED_COLUMNS:
    if col not in input_df.columns:
        input_df[col] = np.nan

input_df = input_df[REQUIRED_COLUMNS]

# --------------------------------------------------------
# PREDICTION CARD (with gauge)
# --------------------------------------------------------
st.markdown(
    """
    <div class="section-card" style="margin-top:1.4rem;">
    """,
    unsafe_allow_html=True,
)

top_cols = st.columns([0.4, 0.6])

with top_cols[0]:
    predict_clicked = st.button("Predict Default Risk")

with top_cols[1]:
    st.write("")

if predict_clicked:
    try:
        proba = float(model.predict_proba(input_df)[0, 1])
        proba_pct = proba * 100

        # Determine label
        if proba >= 0.5:
            label, pill_class = "High Risk", "high"
        elif proba >= 0.2:
            label, pill_class = "Medium Risk", "medium"
        else:
            label, pill_class = "Low Risk", "low"

        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                <div style="font-size:1.05rem; font-weight:600;">Prediction Result</div>
                <div class="risk-pill {pill_class}">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------- Gauge (Plotly) --------
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=proba_pct,
                number={
                    "suffix": "%",
                    "font": {"color": "#F9FAFB", "size": 26},
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "#E5E7EB",
                    },
                    "bar": {"color": "#FFC627"},
                    "bgcolor": "rgba(15,23,42,1)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 20], "color": "rgba(34,197,94,0.35)"},
                        {"range": [20, 50], "color": "rgba(249,115,22,0.35)"},
                        {"range": [50, 100], "color": "rgba(239,68,68,0.35)"},
                    ],
                },
            )
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(15,23,42,0)",
        )

        st.markdown(
            '<div class="prob-label">Estimated probability of default</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f'<div class="prob-value">Model estimate: {proba:.3f}</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Note: This is a probabilistic model trained on historical loan data. "
            "It should support, not replace, human credit decisions."
        )

    except Exception as e:
        st.error("There was an error when calling the model. Details:")
        st.code(str(e))
else:
    st.write(
        "Adjust the borrower profile above, then click **Predict Default Risk** "
        "to see the model output."
    )

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# ABOUT THE MODEL (expander)
# --------------------------------------------------------
with st.expander("About the Model & Methodology", expanded=False):
    st.markdown(
        """
        **Dataset**

        - Home Credit Default Risk dataset (Kaggle)
        - Contains demographic, employment, financial, and application-level variables
        - Target variable: whether a borrower defaulted (`TARGET` = 1)

        **Modeling Approach**

        - Train/test split on historical applications
        - XGBoost classifier wrapped in a full scikit-learn preprocessing pipeline
        - Pipeline includes:
            - Numeric imputation and scaling
            - One-hot encoding of categorical variables
            - Model fitting and probability output calibration

        **Model Selection Metric**

        - **F1-score was the primary evaluation metric** used to select the final model.
        - F1 balances:
            - **Precision**: avoiding false approvals
            - **Recall**: catching risky applicants
        - This is appropriate because the dataset is **highly imbalanced** (few borrowers default).

        **Key Test Metrics**

        - **F1-score ≈ 0.31** (final model selected using this metric)
        - **ROC AUC ≈ 0.76** (moderate ability to distinguish high-risk vs low-risk borrowers)
        - **Accuracy ≈ 0.87**  
          *(Not used for model selection due to class imbalance—most borrowers do not default.)*

        **Interpretation**

        - The model outputs a **probability of default**, allowing borrowers to be **ranked by risk**.
        - It is **not** intended to be a single “approve or deny” decision system.
        - In practice, lenders would apply business thresholds, human oversight, fairness checks, and additional data sources.
        """
    )

