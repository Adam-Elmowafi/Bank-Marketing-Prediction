"""
Bank Deposit Classification — Streamlit Application

This app matches the final notebook training pipeline:
- Loads bank.csv
- Drops 'duration' to prevent Data Leakage
- Binary encoding: default, housing, loan (yes/no → 1/0)
- One-hot encoding: job, marital, education, contact, month, poutcome (drop_first=True)
- Total 41 features after encoding
- StandardScaler normalization
- SMOTE applied exclusively on training data for class balance
- Four models: Logistic Regression, SVM, Random Forest, XGBoost
- Target: deposit (yes/no)
- Optimal Decision Threshold: 0.47 (XGBoost)
"""

import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Bank Deposit Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Constants
# ============================================================================
DATASET_FILE = "bank.csv"
THRESHOLD = 0.47  # Optimized via Bayesian Optimization (Optuna)

# Job categories from notebook
JOBS = [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown",
]
MARITAL_OPTIONS = ["divorced", "married", "single"]
EDUCATION_OPTIONS = ["primary", "secondary", "tertiary", "unknown"]
CONTACT_OPTIONS = ["cellular", "telephone", "unknown"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
POUTCOME_OPTIONS = ["failure", "other", "success", "unknown"]


# ============================================================================
# Theme & Styling
# ============================================================================
def apply_theme() -> None:
    """Apply dark theme CSS through Streamlit's Markdown renderer."""
    st.markdown(
        """
        <style>
        .stApp { 
            background: linear-gradient(135deg, #0e1117 0%, #161b22 100%); 
            color: #c9d1d9; 
        }
        [data-testid="stSidebar"] { 
            background: #161b22; 
            border-right: 1px solid #30363d; 
        }
        [data-testid="stMetricValue"] { 
            color: #79c0ff; 
            font-size: 24px !important;
            font-weight: bold;
        }
        h1, h2, h3 { 
            color: #79c0ff;
            font-weight: 600;
        }
        .stButton > button {
            background: #7928ca; 
            color: white; 
            border: 0; 
            border-radius: 7px;
            font-weight: 600;
            padding: 10px 20px;
            transition: all 0.3s ease;
        }
        .stButton > button:hover { 
            background: #a855f7; 
            color: white;
            transform: scale(1.02);
        }
        .stSelectbox > div > div {
            background-color: #0d1117;
        }
        .stSlider {
            padding: 20px 0px 20px 0px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# Data Loading & Preprocessing
# ============================================================================
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and preprocess the bank.csv dataset."""
    try:
        df = pd.read_csv(DATASET_FILE, sep=",")
    except FileNotFoundError:
        st.error(f"❌ File {DATASET_FILE} not found. Please place it next to app.py")
        st.stop()
    
    return df


@st.cache_data
def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the dataframe:
    1. Identify target column
    2. Drop 'duration' to prevent data leakage
    3. Convert binary columns to 0/1
    4. One-hot encode categorical columns (excluding target)
    5. Prepare X and y
    """
    df_copy = df.copy()
    
    # Identify target column FIRST
    target_col = 'deposit' if 'deposit' in df_copy.columns else 'y'
    
    # Convert target column to numeric
    if df_copy[target_col].dtype == 'object':
        df_copy[target_col] = df_copy[target_col].astype(str).str.strip().str.lower().map({'no': 0, 'yes': 1})
    
    # CRITICAL: Drop 'duration' to prevent Data Leakage
    if 'duration' in df_copy.columns:
        df_copy = df_copy.drop(columns=['duration'])
        
    # Separate features and target BEFORE encoding
    X = df_copy.drop(columns=[target_col])
    y = df_copy[target_col]
    
    # Convert binary columns in X only
    binary_cols = ['default', 'housing', 'loan']
    for col in binary_cols:
        if col in X.columns:
            X[col] = X[col].astype(str).str.strip().str.lower().replace({'yes': 1, 'no': 0}).fillna(0).astype(int)
    
    # One-hot encoding for categorical columns in X only (drop_first=True)
    categorical_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    existing_cats = [c for c in categorical_cols if c in X.columns]
    X_encoded = pd.get_dummies(X, columns=existing_cats, drop_first=True, dtype=int)
    
    return X_encoded, y, df_copy, target_col


@st.cache_resource
def train_models(X_train, X_test, y_train, y_test, _scaler):
    """Train Models with SMOTE applied strictly on Training Data."""
    # Fit the scaler explicitly on training data
    X_train_scaled = _scaler.fit_transform(X_train)
    X_test_scaled = _scaler.transform(X_test)
    
    # Apply SMOTE to training data only to prevent leakage
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # 1. Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_resampled, y_train_resampled)
    
    # 2. SVM
    svm_model = SVC(random_state=42, probability=True)
    svm_model.fit(X_train_resampled, y_train_resampled)
    
    # 3. Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_resampled, y_train_resampled)
    
    # 4. XGBoost (Best Model)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train_resampled, y_train_resampled)
    
    return {
        'lr': lr_model,
        'svm': svm_model,
        'rf': rf_model,
        'xgb': xgb_model,
        'scaler': _scaler
    }

# ============================================================================
# Prediction & Evaluation
# ============================================================================
def prepare_inference_data(user_inputs: dict, feature_names: list[str]) -> np.ndarray:
    """Create an encoded row aligned with training feature columns."""
    encoded = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Numerical values (Note: 'duration' is excluded)
    numerical_values = {
        "age": user_inputs["age"],
        "balance": user_inputs["balance"],
        "day": user_inputs["day"],
        "campaign": user_inputs["campaign"],
        "pdays": user_inputs["pdays"],
        "previous": user_inputs["previous"],
    }
    for column, value in numerical_values.items():
        if column in encoded.columns:
            encoded.loc[0, column] = value
    
    # Binary values
    binary_values = {
        "default": int(user_inputs["default"] == "yes"),
        "housing": int(user_inputs["housing"] == "yes"),
        "loan": int(user_inputs["loan"] == "yes"),
    }
    for column, value in binary_values.items():
        if column in encoded.columns:
            encoded.loc[0, column] = value
    
    # Categorical values (one-hot encoded)
    for column in ("job", "marital", "education", "contact", "month", "poutcome"):
        dummy_column = f"{column}_{user_inputs[column]}"
        if dummy_column in encoded.columns:
            encoded.loc[0, dummy_column] = 1
    
    return encoded.values


def predict_deposit(user_inputs: dict, model, scaler, feature_names: list[str], model_name: str) -> dict:
    """Make a prediction using the selected model."""
    encoded_input = prepare_inference_data(user_inputs, feature_names)
    scaled_input = scaler.transform(encoded_input)
    
    if hasattr(model, 'predict_proba'):
        probability = float(model.predict_proba(scaled_input)[0, 1])
    else:
        probability = float(model.decision_function(scaled_input)[0])
        probability = 1 / (1 + np.exp(-probability))  # Sigmoid
    
    # Use global optimized threshold
    prediction = int(probability >= THRESHOLD)
    
    return {
        "probability": probability,
        "prediction": prediction,
        "model": model_name,
    }


# ============================================================================
# UI Components
# ============================================================================
def build_sidebar() -> dict:
    """Collect customer profile inputs from sidebar."""
    st.sidebar.title("🏦 Customer Profile")
    
    st.sidebar.subheader("👤 Demographics")
    age = st.sidebar.slider("Age (years)", 18, 95, 45)
    
    st.sidebar.subheader("💰 Financial Profile")
    balance = st.sidebar.slider("Account Balance", -10000, 100000, 1500, step=500)
    job = st.sidebar.selectbox("Occupation", JOBS, index=4)
    marital = st.sidebar.selectbox("Marital Status", MARITAL_OPTIONS, index=1)
    education = st.sidebar.selectbox("Education Level", EDUCATION_OPTIONS, index=1)
    
    st.sidebar.subheader("📋 Credit Profile")
    default = st.sidebar.selectbox("Credit in Default?", ["no", "yes"])
    housing = st.sidebar.selectbox("Housing Loan?", ["no", "yes"], index=1)
    loan = st.sidebar.selectbox("Personal Loan?", ["no", "yes"])
    
    st.sidebar.subheader("📞 Campaign Information")
    contact = st.sidebar.selectbox("Contact Type", CONTACT_OPTIONS)
    day = st.sidebar.slider("Day of Month", 1, 31, 15)
    month = st.sidebar.selectbox("Month of Contact", MONTHS, index=4)
    campaign = st.sidebar.slider("Contacts in Campaign", 1, 50, 2)
    pdays = st.sidebar.slider("Days Since Previous Campaign", -1, 999, -1)
    previous = st.sidebar.slider("Previous Campaign Contacts", 0, 20, 0)
    poutcome = st.sidebar.selectbox("Previous Campaign Result", POUTCOME_OPTIONS, index=3)
    
    return {
        "age": age, "balance": balance, "job": job, "marital": marital,
        "education": education, "default": default, "housing": housing,
        "loan": loan, "contact": contact, "day": day, "month": month,
        "campaign": campaign, "pdays": pdays, "previous": previous,
        "poutcome": poutcome,
    }


# ============================================================================
# Tab Content Functions
# ============================================================================
def show_overview(df_original: pd.DataFrame, num_features: int) -> None:
    """Show dataset overview metrics."""
    st.title("📊 Bank Deposit Predictor")
    st.caption("🎯 Predicting whether a customer will subscribe to a term deposit")
    
    deposit_col = 'deposit' if 'deposit' in df_original.columns else 'y'
    subscription_rate = (df_original[deposit_col] == 'yes').sum() / len(df_original)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📈 Total Records", f"{len(df_original):,}")
    c2.metric("🔢 Raw Features", "16", "Without Duration")
    c3.metric("✨ Encoded Features", num_features)
    c4.metric("📊 Subscription Rate", f"{subscription_rate * 100:.1f}%")
    c5.metric("⚖️ Class Imbalance", "SMOTE", "Training Only")
    
    st.divider()


def show_prediction_tab(user_inputs, models_dict, feature_names) -> None:
    """Show prediction interface and results."""
    st.header("🎯 Predict Subscription Likelihood")

    col1, col2 = st.columns(2)

    with col1:
        model_choice = st.radio(
            "🤖 Select Model:",
            ("XGBoost (Best Performance)", "Random Forest", "SVM", "Logistic Regression"),
            index=0
        )

        if "XGBoost" in model_choice:
            selected_model = models_dict["xgb"]
            model_name = "XGBoost"
        elif "Random Forest" in model_choice:
            selected_model = models_dict["rf"]
            model_name = "Random Forest"
        elif "SVM" in model_choice:
            selected_model = models_dict["svm"]
            model_name = "SVM"
        else:
            selected_model = models_dict["lr"]
            model_name = "Logistic Regression"

    with col2:
        if st.button(
            "🔮 Predict Subscription",
            use_container_width=True,
            key="predict_btn"
        ):
            result = predict_deposit(
                user_inputs,
                selected_model,
                models_dict["scaler"],
                feature_names,
                model_name
            )
            st.session_state.prediction_result = result

    if "prediction_result" in st.session_state:
        result = st.session_state.prediction_result
        probability = result["probability"]

        left, right = st.columns(2)

        with left:
            if result["prediction"] == 1:
                st.success(
                    "✅ **LIKELY TO SUBSCRIBE**",
                    icon="✅"
                )
            else:
                st.warning(
                    "❌ **UNLIKELY TO SUBSCRIBE**",
                    icon="⚠️"
                )

            st.metric(
                "Subscription Probability",
                f"{probability:.2%}",
                delta=(
                    f"{probability - THRESHOLD:.2%}"
                    if probability > THRESHOLD
                    else ""
                )
            )

            st.caption(
                f"Decision Rule: Probability ≥ {THRESHOLD:.0%} → Subscribe"
            )

        with right:
            # Gauge chart
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%", "font": {"size": 28}},
                    title={
                        "text": "Subscription Probability",
                        "font": {"size": 16}
                    },
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#7928ca"},
                        "threshold": {
                            "line": {"color": "#ff4b4b", "width": 4},
                            "value": THRESHOLD * 100
                        },
                        "steps": [
                            {"range": [0, THRESHOLD * 100], "color": "#f0f0f0"},
                            {"range": [THRESHOLD * 100, 100], "color": "#e0e0e0"}
                        ]
                    }
                )
            )

            fig.update_layout(
                template="plotly_dark",
                height=300,
                margin=dict(l=20, r=20, t=60, b=10),
                paper_bgcolor="#161b22",
                font=dict(color="#c9d1d9")
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


def show_analytics_tab(df_original: pd.DataFrame) -> None:
    """Show data analytics and visualizations."""
    st.header("📈 Data Analytics")
    
    # Convert yes/no to numeric for visualization
    deposit_col = 'deposit' if 'deposit' in df_original.columns else 'y'
    df_viz = df_original.copy()
    df_viz['subscription'] = (df_viz[deposit_col] == 'yes').astype(int)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age_chart = px.histogram(
            df_viz, x="age", color="subscription",
            nbins=30,
            title="Age Distribution by Subscription",
            labels={"subscription": "Subscribed", "age": "Age (years)"},
            template="plotly_dark",
            color_discrete_map={0: "#ff4b4b", 1: "#51cf66"}
        )
        st.plotly_chart(age_chart, use_container_width=True)
    
    with col2:
        balance_chart = px.histogram(
            df_viz, x="balance", nbins=50,
            title="Account Balance Distribution",
            labels={"balance": "Balance", "count": "Count"},
            template="plotly_dark"
        )
        balance_chart.update_traces(marker_color="#7928ca")
        st.plotly_chart(balance_chart, use_container_width=True)
    
    # Subscription rate by job
    job_rates = df_viz.groupby("job", as_index=False)["subscription"].agg(['sum', 'count'])
    job_rates['rate'] = (job_rates['sum'] / job_rates['count'] * 100).round(2)
    job_rates = job_rates.sort_values('rate', ascending=False)
    
    job_chart = px.bar(
        job_rates, x="job", y="rate",
        title="Subscription Rate by Occupation",
        labels={"rate": "Subscription Rate (%)", "job": "Occupation"},
        template="plotly_dark"
    )
    job_chart.update_traces(marker_color="#51cf66")
    st.plotly_chart(job_chart, use_container_width=True)


def show_model_comparison_tab() -> None:
    """Show hardcoded model comparison metrics mirroring the defense presentation."""
    st.header("🏆 Model Comparison (Official Validation Results)")
    
    st.markdown("Metrics reflect **realistic, leakage-free** evaluation (duration dropped) and **SMOTE applied exclusively to training data**.")
    
    # Official metrics from the project presentation
    comparison_data = {
        'Model': ['Logistic Regression', 'SVM', 'Random Forest', 'XGBoost'],
        'Train Acc': ['70.31%', '75.17%', '78.78%', '75.30%'],
        'Test Acc': ['69.50%', '72.68%', '72.91%', '73.67%'],
        'Train Precision': ['71.54%', '77.22%', '79.40%', '75.90%'],
        'Test Precision': ['70.15%', '73.94%', '73.30%', '74.07%'],
        'Train Recall': ['70.31%', '75.17%', '78.78%', '75.30%'],
        'Test Recall': ['69.50%', '72.68%', '72.91%', '73.67%'],
        'Train F1': ['69.88%', '74.69%', '78.67%', '75.15%'],
        'Test F1': ['68.94%', '72.02%', '72.61%', '73.39%']
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.success(
        "✨ **XGBoost Selected**: Achieved the highest test accuracy (73.67%) and test F1-score (73.39%) "
        "while maintaining stability and guarding against overfitting."
    )


def show_documentation_tab() -> None:
    """Show engineering documentation."""
    st.header("📚 Engineering Documentation")
    
    st.markdown("""
    ### 🔧 Data Pipeline
    
    **1. Data Loading & Leakage Prevention**
    - File: `bank.csv` (delimiter: `,`)
    - Records: 11,162 samples
    - ⚠️ **CRITICAL: Dropped `duration` feature to prevent data leakage (call duration is unknown prior to a call).**
    
    **2. Feature Engineering**
    - **Binary Encoding**: `default`, `housing`, `loan` → `yes/no` to `1/0`
    - **One-Hot Encoding**: `job`, `marital`, `education`, `contact`, `month`, `poutcome`
      - Applied with `drop_first=True` to avoid multicollinearity
    - **Total Encoded Features**: 41
    
    **3. Data Preprocessing & Balancing**
    - Train/Test Split: 80/20 with stratification
    - Scaler: StandardScaler (fitted on training data only)
    - **SMOTE**: Applied *exclusively* to the training set to resolve class imbalance without leaking test patterns.
    
    **4. Model Optimization**
    - Optimal Decision Threshold: **0.47**
    - Optimized via Optuna & Youden's J statistic for XGBoost to perfectly balance Precision and Recall.
    
    ### 📊 Features Used (41 Total)
    
    **Numerical Features (6)**
    - age, balance, day, campaign, pdays, previous
    
    **Binary Features (3)**
    - default, housing, loan
    
    **Categorical Features (6) → One-Hot Encoded (32 Dummies)**
    - job (11 categories)
    - marital (2 categories)
    - education (3 categories)
    - contact (2 categories)
    - month (11 categories)
    - poutcome (3 categories)
    """)


# ============================================================================
# Main Application
# ============================================================================
def main() -> None:
    """Main application entry point."""
    apply_theme()
    
    # Load and preprocess data
    df_original = load_data()
    X, y, df_encoded, target_col = preprocess_data(df_original)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train with scaler
    scaler = StandardScaler()
    models_dict = train_models(X_train, X_test, y_train, y_test, scaler)
    
    feature_names = X.columns.tolist()
    
    # Build UI
    user_inputs = build_sidebar()
    show_overview(df_original, len(feature_names))
    
    # Tabs
    prediction_tab, analytics_tab, comparison_tab, docs_tab = st.tabs(
        ["🎯 Predict", "📈 Analytics", "🏆 Compare Models", "📚 Documentation"]
    )
    
    with prediction_tab:
        show_prediction_tab(user_inputs, models_dict, feature_names)
    
    with analytics_tab:
        show_analytics_tab(df_original)
    
    with comparison_tab:
        show_model_comparison_tab()
    
    with docs_tab:
        show_documentation_tab()
    
    st.divider()
    st.caption("🏦 Bank Marketing Classification System | End-to-End ML Pipeline | Leakage-Free | XGBoost")


if __name__ == "__main__":
    main()
