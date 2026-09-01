import os
import warnings
import sqlite3
from datetime import datetime
from io import StringIO

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# Suppress only the known serialized-model feature-name warning.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but RandomForestClassifier was fitted with feature names",
    category=UserWarning,
)


CHURN_THRESHOLD = 0.40
PREPROCESSOR_FILE = "models/preprocessor.joblib"
MODEL_FILE = "models/best_random_forest_tuned.pkl"
DATABASE_FILE = "telecom_churn.db"
SEED_FILE = "prototype_seed_customers.csv"

st.set_page_config(
    page_title="Telecom Churn AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(59,130,246,.18), transparent 28%),
            radial-gradient(circle at 85% 15%, rgba(139,92,246,.16), transparent 28%),
            linear-gradient(135deg, #07111f 0%, #0b1730 48%, #111827 100%);
        color: #f8fafc;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #081225 0%, #101b35 100%);
    }
    [data-testid="stSidebar"] * { color: #f8fafc !important; }
    .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }
    .hero {
        padding: 28px;
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 20px;
        background: rgba(255,255,255,.06);
        box-shadow: 0 15px 45px rgba(0,0,0,.25);
        margin-bottom: 22px;
    }
    .hero h1 { margin-bottom: 4px; color: #ffffff; }
    .muted { color: #b8c3d6; }
    .card {
        padding: 18px;
        border-radius: 16px;
        background: rgba(255,255,255,.07);
        border: 1px solid rgba(255,255,255,.10);
        box-shadow: 0 10px 30px rgba(0,0,0,.18);
    }
    .risk-card {
        text-align: center;
        padding: 20px;
        border-radius: 18px;
        background: rgba(255,255,255,.07);
        border: 1px solid rgba(255,255,255,.10);
    }
    .circle-inner {
        width: 118px;
        height: 118px;
        border-radius: 50%;
        background: #0b1730;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .circle-value { font-size: 1.65rem; font-weight: 800; color: #ffffff; }
    .circle-label { font-size: .75rem; color: #b8c3d6; }
    .section-title {
        margin-top: 24px;
        margin-bottom: 10px;
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
    }
    .status-churn {
        padding: 15px;
        border-radius: 12px;
        background: rgba(239,68,68,.16);
        border: 1px solid rgba(239,68,68,.45);
        color: #fecaca;
        font-weight: 700;
        text-align: center;
    }
    .status-retain {
        padding: 15px;
        border-radius: 12px;
        background: rgba(34,197,94,.16);
        border: 1px solid rgba(34,197,94,.45);
        color: #bbf7d0;
        font-weight: 700;
        text-align: center;
    }
    .threshold {
        text-align: center;
        padding: 12px;
        border-radius: 12px;
        background: rgba(59,130,246,.12);
        border: 1px solid rgba(59,130,246,.30);
        color: #bfdbfe;
        font-weight: 700;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.06);
        border: 1px solid rgba(255,255,255,.09);
        padding: 12px;
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_connection():
    return sqlite3.connect(DATABASE_FILE, check_same_thread=False)


def initialize_database():
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS customers (
            customerID TEXT PRIMARY KEY,
            gender TEXT,
            SeniorCitizen TEXT,
            Partner TEXT,
            Dependents TEXT,
            tenure INTEGER,
            PhoneService TEXT,
            MultipleLines TEXT,
            InternetService TEXT,
            OnlineSecurity TEXT,
            OnlineBackup TEXT,
            DeviceProtection TEXT,
            TechSupport TEXT,
            StreamingTV TEXT,
            StreamingMovies TEXT,
            Contract TEXT,
            PaperlessBilling TEXT,
            PaymentMethod TEXT,
            MonthlyCharges REAL,
            TotalCharges REAL,
            actual_churn TEXT,
            source TEXT,
            created_at TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customerID TEXT,
            churn_probability REAL,
            prediction INTEGER,
            threshold REAL,
            risk_level TEXT,
            scanned_at TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS system_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            level TEXT,
            event TEXT,
            customerID TEXT,
            details TEXT
        )
        """
    )
    connection.commit()
    connection.close()


def log_event(level, event, customer_id="", details=""):
    connection = get_connection()
    connection.execute(
        """
        INSERT INTO system_logs(timestamp, level, event, customerID, details)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            level,
            event,
            customer_id,
            details,
        ),
    )
    connection.commit()
    connection.close()


def insert_customer(customer, actual_churn, source):
    connection = get_connection()
    connection.execute(
        """
        INSERT OR REPLACE INTO customers (
            customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
            PhoneService, MultipleLines, InternetService, OnlineSecurity,
            OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
            StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
            MonthlyCharges, TotalCharges, actual_churn, source, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(customer["customerID"]),
            str(customer["gender"]),
            str(customer["SeniorCitizen"]),
            str(customer["Partner"]),
            str(customer["Dependents"]),
            int(customer["tenure"]),
            str(customer["PhoneService"]),
            str(customer["MultipleLines"]),
            str(customer["InternetService"]),
            str(customer["OnlineSecurity"]),
            str(customer["OnlineBackup"]),
            str(customer["DeviceProtection"]),
            str(customer["TechSupport"]),
            str(customer["StreamingTV"]),
            str(customer["StreamingMovies"]),
            str(customer["Contract"]),
            str(customer["PaperlessBilling"]),
            str(customer["PaymentMethod"]),
            float(customer["MonthlyCharges"]),
            float(customer["TotalCharges"]),
            actual_churn,
            source,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    connection.commit()
    connection.close()


def seed_customer_registry():
    connection = get_connection()
    count = connection.execute(
        "SELECT COUNT(*) FROM customers"
    ).fetchone()[0]
    connection.close()

    if count > 0 or not os.path.exists(SEED_FILE):
        return

    seed_df = pd.read_csv(SEED_FILE)

    for _, customer in seed_df.iterrows():
        insert_customer(
            customer,
            str(customer["Churn"]),
            "Historical seed dataset",
        )

    log_event(
        "INFO",
        "Customer registry seeded",
        details=f"{len(seed_df)} historical customers loaded.",
    )


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    model = joblib.load(MODEL_FILE)
    return preprocessor, model


MODEL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def build_model_input(
    tenure,
    monthly_charges,
    total_charges,
    gender,
    senior_citizen,
    partner,
    dependents,
    phone_service,
    multiple_lines,
    internet,
    online_security,
    online_backup,
    device_protect,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless,
    payment_method,
):
    return pd.DataFrame(
        [{
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "gender": gender,
            "SeniorCitizen": "1" if senior_citizen == "Yes (1)" else "0",
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protect,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
        }]
    )


def predict_customer(input_df, preprocessor, model):
    transformed = preprocessor.transform(input_df)
    probabilities = model.predict_proba(transformed)[0]
    churn_probability = float(probabilities[1])
    prediction = int(churn_probability >= CHURN_THRESHOLD)

    if churn_probability >= 0.60:
        risk_level = "High"
    elif churn_probability >= CHURN_THRESHOLD:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return prediction, churn_probability, risk_level


def get_churn_probability(input_df, preprocessor, model):
    transformed = preprocessor.transform(input_df)
    return float(model.predict_proba(transformed)[0][1])


@st.cache_data(show_spinner=False)
def _get_customer_signals_cached(input_json):
    # Reconstruct the exact one-row customer DataFrame from the stable JSON key.
    pd.read_json(StringIO(input_json), orient="split")

    # Use the artifacts loaded by the application initializer.
    base_probability = get_churn_probability(
        customer_df,
        preprocessor,
        model,
    )
    signals = []

    # Evaluate each model feature by comparing the observed value with alternatives.
    for feature in MODEL_FEATURES:
        original_value = customer_df.iloc[0][feature]
        alternatives = []

        if feature == "tenure":
            alternatives = [0, 24, 48, 72]
        elif feature == "MonthlyCharges":
            alternatives = [
                max(0.0, float(original_value) - 20.0),
                float(original_value) + 20.0,
            ]
        elif feature == "TotalCharges":
            alternatives = [
                max(0.0, float(original_value) * 0.75),
                float(original_value) * 1.25,
            ]
        elif original_value in ["Yes", "No"]:
            alternatives = ["Yes" if original_value == "No" else "No"]
        elif original_value in ["0", "1"]:
            alternatives = ["1" if original_value == "0" else "0"]
        elif feature == "Contract":
            alternatives = [
                value for value in ["Month-to-month", "One year", "Two year"]
                if value != original_value
            ]
        elif feature == "PaymentMethod":
            alternatives = [
                value for value in [
                    "Electronic check", "Mailed check",
                    "Bank transfer (automatic)", "Credit card (automatic)"
                ]
                if value != original_value
            ]
        elif feature == "InternetService":
            alternatives = [
                value for value in ["DSL", "Fiber optic", "No"]
                if value != original_value
            ]
        elif feature == "MultipleLines":
            alternatives = [
                value for value in ["No", "Yes", "No phone service"]
                if value != original_value
            ]

        changes = []
        for alternative in alternatives:
            modified = customer_df.copy()
            modified.loc[0, feature] = alternative
            try:
                modified_probability = get_churn_probability(
                    modified,
                    preprocessor,
                    model,
                )
                changes.append(abs(base_probability - modified_probability))
            except Exception:
                pass

        if changes:
            sensitivity = max(changes)
            if sensitivity >= 0.15:
                strength = "Strong"
            elif sensitivity >= 0.07:
                strength = "Moderate"
            else:
                strength = "Supporting"

            signals.append({
                "Feature": feature,
                "Observed value": str(original_value),
                "Signal strength": strength,
                "Sensitivity": round(sensitivity, 4),
            })

    if not signals:
        return pd.DataFrame()

    return pd.DataFrame(signals).sort_values(
        "Sensitivity",
        ascending=False,
    ).head(5)


def get_customer_signals(input_df, preprocessor, model):
    # Convert the input into a stable cache key so repeated Streamlit reruns reuse the result.
    input_json = input_df.to_json(orient="split")

    # Calculate or retrieve the cached customer-specific signals.
    return _get_customer_signals_cached(input_json)


def risk_color(probability):
    # Return a green-to-yellow-to-red color according to churn probability.
    if probability < CHURN_THRESHOLD:
        ratio = probability / CHURN_THRESHOLD
        red = int(34 + (245 - 34) * ratio)
        green = int(197 + (158 - 197) * ratio)
        return f"rgb({red}, {green}, 94)"

    ratio = min((probability - CHURN_THRESHOLD) / 0.60, 1.0)
    red = int(245 + (239 - 245) * ratio)
    green = int(158 + (68 - 158) * ratio)
    return f"rgb({red}, {green}, 94)"


def circular_meter(value, label, title, color=None):
    # Clamp the displayed value to the valid range.
    value = max(0.0, min(1.0, float(value)))

    # Convert the percentage into degrees for the circular progress ring.
    degrees = value * 360

    # Select the supplied color or the calculated risk-sensitive color.
    meter_color = color or risk_color(value)

    # Render the circular meter.
    st.markdown(
        f"""
        <div class="risk-card">
            <div style="
                width:150px;
                height:150px;
                border-radius:50%;
                margin:10px auto 14px auto;
                background:conic-gradient({meter_color} {degrees}deg, #26354f 0deg);
                display:flex;
                align-items:center;
                justify-content:center;">
                <div class="circle-inner">
                    <div class="circle-value">{value * 100:.1f}%</div>
                    <div class="circle-label">{label}</div>
                </div>
            </div>
            <div style="font-weight:700;">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_customer(customer_id):
    connection = get_connection()
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        "SELECT * FROM customers WHERE customerID = ?",
        (customer_id,),
    ).fetchone()
    connection.close()
    return dict(row) if row else None


def get_latest_scan(customer_id):
    # Open the database connection.
    connection = get_connection()

    # Configure named-column access for the returned database row.
    connection.row_factory = sqlite3.Row

    # Retrieve the customer's most recent stored prediction.
    row = connection.execute(
        """
        SELECT *
        FROM scans
        WHERE customerID = ?
        ORDER BY scan_id DESC
        LIMIT 1
        """,
        (customer_id,),
    ).fetchone()

    # Close the database connection.
    connection.close()

    # Return the stored scan as a dictionary when it exists.
    return dict(row) if row else None


def load_customer_for_prediction(customer):
    # Convert a stored customer profile into the model's expected feature frame.
    return pd.DataFrame([{
        feature: customer[feature]
        for feature in MODEL_FEATURES
    }])


def suggested_retention_actions(customer, prediction):
    # Create the list of actions to display to the retention officer.
    actions = []

    # Only recommend intervention when the customer is flagged for churn.
    if prediction == 1:
        # Address month-to-month contract risk.
        if customer["Contract"] == "Month-to-month":
            actions.append(
                "Offer a discounted One-Year or Two-Year contract to encourage longer-term retention."
            )

        # Address missing online security for fiber customers.
        if (
            customer["InternetService"] == "Fiber optic"
            and customer["OnlineSecurity"] == "No"
        ):
            actions.append(
                "Offer Online Security as a discounted bundle to improve perceived service value."
            )

        # Address missing technical support.
        if customer["TechSupport"] == "No":
            actions.append(
                "Offer a free Tech Support trial or assisted-support package."
            )

        # Address electronic-check payment behavior.
        if customer["PaymentMethod"] == "Electronic check":
            actions.append(
                "Encourage migration to automatic bank transfer or credit-card payment."
            )

        # Address relatively high monthly charges.
        if float(customer["MonthlyCharges"]) > 70:
            actions.append(
                "Review the customer's monthly plan and consider a targeted loyalty discount."
            )

        # Prioritize customers early in their lifecycle.
        if int(customer["tenure"]) < 12:
            actions.append(
                "Prioritize early-lifecycle outreach and assign a retention specialist."
            )

        # Provide a general fallback if no profile rule fires.
        if not actions:
            actions.append(
                "Assign a retention specialist to understand the customer's concerns and offer a suitable retention intervention."
            )

    # Give a lower-intensity recommendation when the customer is not flagged.
    else:
        actions.append(
            "Continue standard engagement and reassess if the customer's usage, contract, or billing profile changes."
        )

    # Return the completed recommendations.
    return actions


def save_scan(customer_id, probability, prediction, risk_level):
    connection = get_connection()
    connection.execute(
        """
        INSERT INTO scans(
            customerID,
            churn_probability,
            prediction,
            threshold,
            risk_level,
            scanned_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            customer_id,
            probability,
            prediction,
            CHURN_THRESHOLD,
            risk_level,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    connection.commit()
    connection.close()


def generate_seed_scans(preprocessor, model):
    # Retrieve all historical customers loaded from the 70-customer seed file.
    connection = get_connection()
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT *
        FROM customers
        WHERE source = 'Historical seed dataset'
        """
    ).fetchall()
    connection.close()

    # Process every historical customer that does not already have a scan.
    for row in rows:
        customer = dict(row)
        customer_id = str(customer["customerID"])

        # Prevent duplicate scans when Streamlit reruns the application.
        if get_latest_scan(customer_id):
            continue

        # Reconstruct the customer's model input.
        model_input = load_customer_for_prediction(customer)

        # Transform the input using the saved preprocessing pipeline.
        transformed = preprocessor.transform(model_input)

        # Obtain the model's class probabilities.
        probabilities = model.predict_proba(transformed)[0]

        # Extract the churn probability.
        probability = float(probabilities[1])

        # Apply the team's locked 40% threshold.
        prediction = int(probability >= CHURN_THRESHOLD)

        # Assign the application's risk band.
        if probability >= 0.60:
            risk_level = "High"
        elif probability >= CHURN_THRESHOLD:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Persist the historical customer's completed scan.
        save_scan(
            customer_id,
            probability,
            prediction,
            risk_level,
        )

        # Record the operation in the system log.
        log_event(
            "INFO",
            "Historical customer scanned",
            customer_id=customer_id,
            details=(
                f"Probability={probability:.4f}; "
                f"threshold={CHURN_THRESHOLD:.2f}; "
                f"prediction={prediction}; "
                f"actual_churn={customer['actual_churn']}"
            ),
        )


def ensure_historical_scans(preprocessor, model):
    # Count the historical customers currently stored in the registry.
    connection = get_connection()
    total_historical = connection.execute(
        "SELECT COUNT(*) FROM customers WHERE source = 'Historical seed dataset'"
    ).fetchone()[0]

    # Count historical customers that already have at least one stored scan.
    scanned_historical = connection.execute(
        """
        SELECT COUNT(DISTINCT s.customerID)
        FROM scans AS s
        INNER JOIN customers AS c
            ON s.customerID = c.customerID
        WHERE c.source = 'Historical seed dataset'
        """
    ).fetchone()[0]
    connection.close()

    # Skip the expensive seed scan when every historical customer is already scanned.
    if total_historical == 0 or scanned_historical >= total_historical:
        return

    # Generate only the missing historical scans when the registry page is opened.
    generate_seed_scans(preprocessor, model)


@st.cache_resource
def initialize_application():
    # Initialize the local database schema once for the Streamlit process.
    initialize_database()

    # Load the historical registry records when the database is empty.
    seed_customer_registry()

    # Load the saved preprocessing pipeline and tuned Random Forest model.
    loaded_preprocessor, loaded_model = load_artifacts()

    # Return the artifacts without performing historical inference during startup.
    return loaded_preprocessor, loaded_model


try:
    preprocessor, model = initialize_application()
    artifacts_loaded = True
except Exception as error:
    preprocessor = None
    model = None
    artifacts_loaded = False
    artifact_error = str(error)
    log_event(
        "ERROR",
        "Model artifacts could not be loaded",
        details=artifact_error,
    )


with st.sidebar:
    st.markdown("## 📡 Telecom Churn AI")
    st.caption("Group 2 • Retention Intelligence")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "Dashboard",
            "Customer Registry",
            "Analytics",
            "System Logs",
        ],
    )

    st.markdown("---")
    st.markdown("### Customer Scan")

    customer_id = st.text_input(
        "Customer ID",
        placeholder="e.g. 7590-VHVEG",
    )

    tenure = st.slider("Tenure (months)", 0, 72, 12)

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=65.0,
        step=0.5,
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=float(tenure * monthly_charges),
        step=1.0,
    )

    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"],
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"],
    )

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No (0)", "Yes (1)"],
    )
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"],
    )
    internet = st.selectbox(
        "Internet Service",
        ["Fiber optic", "DSL", "No"],
    )
    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"],
    )
    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"],
    )
    device_protect = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"],
    )
    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"],
    )
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"],
    )
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"],
    )

    predict_button = st.button(
        "🔍 Run Prediction",
        use_container_width=True,
    )


st.markdown(
    """
    <div class="hero">
        <h1>📡 Telecom Customer Churn AI</h1>
        <div class="muted">
            Retention intelligence for identifying customers who may be at risk
            of leaving the service.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="threshold">🔒 Classification threshold locked at {CHURN_THRESHOLD:.0%}</div>',
    unsafe_allow_html=True,
)


if page == "Dashboard":

    st.markdown(
        '<div class="section-title">🔎 Find Customer</div>',
        unsafe_allow_html=True,
    )

    search_col1, search_col2 = st.columns([4, 1])

    with search_col1:
        search_id = st.text_input(
            "Search customer by unique ID",
            placeholder="Enter customerID",
            label_visibility="collapsed",
        )

    with search_col2:
        search_button = st.button(
            "Search",
            use_container_width=True,
        )

    if search_button and search_id.strip():
        searched_customer = get_customer(search_id.strip())

        if searched_customer:
            st.session_state["selected_customer"] = searched_customer
            log_event(
                "INFO",
                "Customer searched",
                customer_id=search_id.strip(),
            )
            st.success(
                f"Customer {search_id.strip()} loaded from the registry."
            )

            # Retrieve the customer's already-generated historical scan.
            stored_scan = get_latest_scan(search_id.strip())

            # Display the stored result when the customer has been pre-scanned.
            if stored_scan:
                st.markdown(
                    '<div class="section-title">📌 Stored Customer Result</div>',
                    unsafe_allow_html=True,
                )

                result_cols = st.columns(4)

                with result_cols[0]:
                    st.metric(
                        "Actual Historical Churn",
                        searched_customer["actual_churn"],
                    )

                with result_cols[1]:
                    st.metric(
                        "Model Prediction",
                        "Churn" if stored_scan["prediction"] == 1 else "No Churn",
                    )

                with result_cols[2]:
                    st.metric(
                        "Churn Probability",
                        f"{stored_scan['churn_probability']:.1%}",
                    )

                with result_cols[3]:
                    st.metric(
                        "Risk Level",
                        stored_scan["risk_level"],
                    )

                # Display the threshold used by the stored prediction.
                st.markdown(
                    f'<div class="threshold">🔒 Decision threshold: {CHURN_THRESHOLD:.0%}</div>',
                    unsafe_allow_html=True,
                )

                # Reconstruct the customer's model input for signal analysis.
                searched_input = load_customer_for_prediction(
                    searched_customer
                )

                # Calculate the customer's key model sensitivity signals.
                searched_signals = get_customer_signals(
                    searched_input,
                    preprocessor,
                    model,
                )

                st.markdown(
                    '<div class="section-title">🎯 Key Customer Signals</div>',
                    unsafe_allow_html=True,
                )

                if searched_signals.empty:
                    st.info("No customer-specific signals could be calculated.")
                else:
                    st.dataframe(
                        searched_signals,
                        use_container_width=True,
                        hide_index=True,
                    )

                # Display the same retention actions used for newly scanned customers.
                st.markdown(
                    '<div class="section-title">💡 Suggested Retention Actions</div>',
                    unsafe_allow_html=True,
                )

                for action in suggested_retention_actions(
                    searched_customer,
                    int(stored_scan["prediction"]),
                ):
                    st.markdown(f"- {action}")

            customer = searched_customer
            st.info(
                "Customer found. Review the profile below and use the "
                "sidebar values to run a fresh prediction."
            )

        else:
            st.warning("Customer ID was not found in the registry.")

    if not artifacts_loaded:
        st.error("Model artifacts could not be loaded.")
        st.code(artifact_error)
        st.stop()

    if predict_button:
        if not customer_id.strip():
            st.warning(
                "Please enter a Customer ID before running the prediction."
            )
        else:
            input_data = build_model_input(
                tenure,
                monthly_charges,
                total_charges,
                gender,
                senior_citizen,
                partner,
                dependents,
                phone_service,
                multiple_lines,
                internet,
                online_security,
                online_backup,
                device_protect,
                tech_support,
                streaming_tv,
                streaming_movies,
                contract,
                paperless,
                payment_method,
            )

            prediction, churn_probability, risk_level = predict_customer(
                input_data,
                preprocessor,
                model,
            )

            new_customer = input_data.iloc[0].to_dict()
            new_customer["customerID"] = customer_id.strip()

            insert_customer(
                new_customer,
                "Unknown",
                "New application scan",
            )

            save_scan(
                customer_id.strip(),
                churn_probability,
                prediction,
                risk_level,
            )

            log_event(
                "INFO",
                "Customer prediction generated",
                customer_id=customer_id.strip(),
                details=(
                    f"Probability={churn_probability:.4f}; "
                    f"threshold={CHURN_THRESHOLD:.2f}; "
                    f"prediction={prediction}; "
                    f"risk={risk_level}"
                ),
            )

            st.session_state["latest_input"] = input_data
            st.session_state["latest_customer_id"] = customer_id.strip()
            st.session_state["latest_prediction"] = prediction
            st.session_state["latest_probability"] = churn_probability
            st.session_state["latest_risk"] = risk_level

    if "latest_probability" in st.session_state:
        latest_id = st.session_state["latest_customer_id"]
        churn_probability = st.session_state["latest_probability"]
        retain_probability = 1.0 - churn_probability
        prediction = st.session_state["latest_prediction"]
        risk_level = st.session_state["latest_risk"]

        if prediction == 1:
            st.markdown(
                '<div class="status-churn">⚠️ CHURN FLAG — RETENTION INTERVENTION RECOMMENDED</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-retain">✅ NO CHURN FLAG — STANDARD MONITORING</div>',
                unsafe_allow_html=True,
            )

        meter1, meter2, meter3 = st.columns(3)

        with meter1:
            circular_meter(
                churn_probability,
                "Churn probability",
                "Churn Risk",
                risk_color(churn_probability),
            )

        with meter2:
            circular_meter(
                retain_probability,
                "Retention probability",
                "Retention",
                "#22c55e",
            )

        with meter3:
            circular_meter(
                CHURN_THRESHOLD,
                "Locked threshold",
                "Decision Boundary",
            )

        st.markdown(
            f"""
            <div class="card">
                <b>Customer:</b> {latest_id}
                &nbsp;&nbsp; | &nbsp;&nbsp;
                <b>Risk level:</b> {risk_level}
                &nbsp;&nbsp; | &nbsp;&nbsp;
                <b>Decision threshold:</b> {CHURN_THRESHOLD:.0%}
            </div>
            """,
            unsafe_allow_html=True,
        )

        signals = get_customer_signals(
            st.session_state["latest_input"],
            preprocessor,
            model,
        )

        st.markdown(
            '<div class="section-title">🎯 Key Customer Signals</div>',
            unsafe_allow_html=True,
        )

        if signals.empty:
            st.info(
                "No customer-specific signals could be calculated for this profile."
            )
        else:
            st.dataframe(
                signals,
                use_container_width=True,
                hide_index=True,
            )

        # Display the retention recommendations for the new scan.
        st.markdown(
            '<div class="section-title">💡 Suggested Retention Actions</div>',
            unsafe_allow_html=True,
        )

        # Retrieve the exact customer profile used for the completed prediction.
        scanned_customer = st.session_state["latest_input"].iloc[0].to_dict()

        # Render each profile-specific retention action.
        for action in suggested_retention_actions(
            scanned_customer,
            prediction,
        ):
            st.markdown(f"- {action}")

        st.markdown(
            '<div class="section-title">📋 Scanned Customer Profile</div>',
            unsafe_allow_html=True,
        )

        profile = st.session_state["latest_input"].T.rename(
            columns={0: "Value"}
        )
        profile.index.name = "Feature"

        st.dataframe(
            profile,
            use_container_width=True,
        )

    else:
        st.info(
            "Enter a customer ID and customer profile in the sidebar, "
            "then select Run Prediction."
        )

elif page == "Customer Registry":

    # Complete missing historical scans only when the registry is opened.
    ensure_historical_scans(preprocessor, model)

    st.markdown(
        '<div class="section-title">🗃️ Customer Registry</div>',
        unsafe_allow_html=True,
    )

    connection = get_connection()

    registry_df = pd.read_sql_query(
        """
        SELECT
            customerID,
            tenure,
            Contract,
            InternetService,
            MonthlyCharges,
            actual_churn,
            source,
            created_at
        FROM customers
        ORDER BY created_at DESC
        """,
        connection,
    )

    connection.close()

    total_customers = len(registry_df)
    known_outcomes = int(
        (registry_df["actual_churn"] != "Unknown").sum()
    )
    scanned_customers = int(
        (registry_df["source"] == "New application scan").sum()
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Customers in Registry", total_customers)

    with col2:
        st.metric("Known Historical Outcomes", known_outcomes)

    with col3:
        st.metric("Newly Scanned", scanned_customers)

    st.dataframe(
        registry_df,
        use_container_width=True,
        hide_index=True,
    )

elif page == "Analytics":

    st.markdown(
        '<div class="section-title">📈 Scan Analytics</div>',
        unsafe_allow_html=True,
    )

    connection = get_connection()

    scans_df = pd.read_sql_query(
        "SELECT * FROM scans ORDER BY scanned_at ASC",
        connection,
    )

    customers_df = pd.read_sql_query(
        "SELECT customerID, actual_churn FROM customers",
        connection,
    )

    connection.close()

    if scans_df.empty:
        st.info("No application scans have been recorded yet.")
    else:
        total_scans = len(scans_df)
        churn_flags = int(scans_df["prediction"].sum())
        non_churn_flags = total_scans - churn_flags
        average_probability = float(
            scans_df["churn_probability"].mean()
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Scans", total_scans)

        with col2:
            st.metric("Churn Flags", churn_flags)

        with col3:
            st.metric("No-Churn Flags", non_churn_flags)

        with col4:
            st.metric(
                "Average Churn Probability",
                f"{average_probability:.1%}",
            )

        st.markdown(
            f'<div class="threshold">Decision threshold: {CHURN_THRESHOLD:.0%}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">Prediction Distribution</div>',
            unsafe_allow_html=True,
        )

        distribution = pd.DataFrame(
            {
                "Prediction": ["Churn", "No Churn"],
                "Customers": [churn_flags, non_churn_flags],
            }
        ).set_index("Prediction")

        st.bar_chart(distribution)

        st.markdown(
            '<div class="section-title">Churn Probability by Scan</div>',
            unsafe_allow_html=True,
        )

        probability_chart = scans_df[
            ["scan_id", "churn_probability"]
        ].set_index("scan_id")

        st.line_chart(probability_chart)

        outcome_df = scans_df.merge(
            customers_df,
            on="customerID",
            how="left",
        )

        known = outcome_df[
            outcome_df["actual_churn"].isin(["Yes", "No"])
        ].copy()

        if not known.empty:
            known["actual_binary"] = (
                known["actual_churn"] == "Yes"
            ).astype(int)

            known["correct"] = (
                known["prediction"] == known["actual_binary"]
            )

            st.markdown(
                '<div class="section-title">Historical Outcome Comparison</div>',
                unsafe_allow_html=True,
            )

            st.metric(
                "Accuracy on Known Historical Outcomes",
                f"{known['correct'].mean():.1%}",
            )

            st.caption(
                "This comparison applies only to historical customers with a known "
                "Churn label. Newly scanned customers do not have a future outcome."
            )

        st.warning(
            "Scan analytics describe model usage and predictions. They do not "
            "establish future churn outcomes for newly scanned customers."
        )

elif page == "System Logs":

    st.markdown(
        '<div class="section-title">🛠️ System Logs</div>',
        unsafe_allow_html=True,
    )

    connection = get_connection()

    logs_df = pd.read_sql_query(
        """
        SELECT timestamp, level, event, customerID, details
        FROM system_logs
        ORDER BY log_id DESC
        LIMIT 500
        """,
        connection,
    )

    connection.close()

    st.metric("Recorded Events", len(logs_df))

    st.dataframe(
        logs_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "The log records application startup, database seeding, customer searches, "
        "predictions, and model-loading errors."
    )
