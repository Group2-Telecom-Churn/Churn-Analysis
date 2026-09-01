"""
Retention IQ - Telecom Customer Retention Intelligence
Developed by Group 2.

The application turns a churn model into a retention workflow:
Identify -> Score -> Explain -> Prioritise -> Intervene -> Track -> Measure.
"""

import os
import sqlite3
from datetime import datetime
from string import Template

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st


PRODUCT_NAME = "Retention IQ"
PRODUCT_TAGLINE = "Telecom Customer Retention Intelligence"
BUILT_BY = "Developed by Group 2"

# Probability at or above which a customer is flagged for retention action.
CHURN_THRESHOLD = 0.40

# Upper band boundary: at or above this the case is treated as High risk.
HIGH_RISK_THRESHOLD = 0.60

# Assumed months of revenue preserved by a successful intervention. Used to
# convert a saved customer into a headline figure; stated in the UI so the
# number is never mistaken for a measured outcome.
RETENTION_HORIZON_MONTHS = 12

# Resolve every path against this file instead of the working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_file(*candidates):
    # Return the first candidate that exists, otherwise the first candidate so
    # that any resulting error message names a sensible expected location.
    for candidate in candidates:
        full_path = os.path.join(BASE_DIR, candidate)
        if os.path.exists(full_path):
            return full_path
    return os.path.join(BASE_DIR, candidates[0])


# The repo has both a model/ and a models/ folder, so check each spelling
# before falling back to a file sitting next to app.py.
PREPROCESSOR_FILE = find_file(
    "models/preprocessor.joblib",
    "model/preprocessor.joblib",
    "preprocessor.joblib",
)
MODEL_FILE = find_file(
    "models/best_random_forest_tuned.pkl",
    "model/best_random_forest_tuned.pkl",
    "best_random_forest_tuned.pkl",
)
DATABASE_FILE = os.path.join(BASE_DIR, "telecom_churn.db")

# Prefer a full Telco extract when one is committed, otherwise fall back to the
# 70-row prototype sample. Dropping the full CSV into the repo scales every KPI
# and chart with no code change.
SEED_FILE = find_file(
    "data/telco_customer_churn.csv",
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "prototype_seed_customers.csv",
    "data/prototype_seed_customers.csv",
)

st.set_page_config(
    page_title=f"{PRODUCT_NAME} - Retention Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================================
# Theming
#
# .streamlit/config.toml defines both a light and a dark theme. Streamlit
# exposes no CSS custom properties in the main app DOM, so the palette has to
# be resolved in Python via st.context.theme and substituted into the
# stylesheet rather than read from CSS.
# ==========================================================================

# Risk uses the status palette, which is deliberately NOT themed - the same
# three steps in both modes. Amber sits below 3:1 on a light surface by design,
# so every risk mark in this app carries an icon and a text label and every
# risk chart carries direct value labels. Colour never encodes risk alone.
RISK_COLORS = {"High": "#d03b3b", "Medium": "#fab219", "Low": "#0ca30c"}
RISK_ICONS = {"High": "▲", "Medium": "◆", "Low": "●"}
RISK_ORDER = ["High", "Medium", "Low"]

DARK_PALETTE = {
    "appBackground": (
        "radial-gradient(circle at 15% 10%, rgba(59,130,246,.18), transparent 28%),"
        "radial-gradient(circle at 85% 15%, rgba(139,92,246,.16), transparent 28%),"
        "linear-gradient(135deg, #07111f 0%, #0b1730 48%, #111827 100%)"
    ),
    "sidebarBackground": "linear-gradient(180deg, #081225 0%, #101b35 100%)",
    "text": "#f8fafc",
    "heading": "#ffffff",
    "muted": "#b8c3d6",
    "surface": "rgba(255,255,255,.07)",
    "surfaceSoft": "rgba(255,255,255,.06)",
    "border": "rgba(255,255,255,.10)",
    "shadow": "0 15px 45px rgba(0,0,0,.25)",
    "cardShadow": "0 10px 30px rgba(0,0,0,.18)",
    "circleBackground": "#0b1730",
    "meterTrack": "#26354f",
    # Chart chrome, stepped for the dark surface.
    "series1": "#3987e5",
    "series2": "#d95926",
    "chartGrid": "#22304a",
    "chartAxis": "#3a4a68",
    "chartMuted": "#93a4bf",
}

LIGHT_PALETTE = {
    "appBackground": (
        "radial-gradient(circle at 15% 10%, rgba(59,130,246,.10), transparent 28%),"
        "radial-gradient(circle at 85% 15%, rgba(139,92,246,.09), transparent 28%),"
        "linear-gradient(135deg, #f8fafc 0%, #eef2f9 48%, #e8edf6 100%)"
    ),
    "sidebarBackground": "linear-gradient(180deg, #eef2f9 0%, #e2e8f0 100%)",
    "text": "#0f172a",
    "heading": "#0f172a",
    "muted": "#475569",
    "surface": "rgba(255,255,255,.78)",
    "surfaceSoft": "rgba(255,255,255,.65)",
    "border": "rgba(15,23,42,.12)",
    "shadow": "0 15px 45px rgba(15,23,42,.08)",
    "cardShadow": "0 10px 30px rgba(15,23,42,.07)",
    "circleBackground": "#ffffff",
    "meterTrack": "#dbe2ec",
    "series1": "#2a78d6",
    "series2": "#eb6834",
    "chartGrid": "#e2e8f0",
    "chartAxis": "#cbd5e1",
    "chartMuted": "#64748b",
}


def active_palette():
    # st.context.theme.type reports the theme the viewer is actually using.
    # It can briefly be wrong on first load or mid theme-switch, so anything
    # unexpected falls back to dark, which is the configured default.
    try:
        theme_type = st.context.theme.type
    except Exception:
        theme_type = None

    return LIGHT_PALETTE if theme_type == "light" else DARK_PALETTE


PALETTE = active_palette()

# Template rather than an f-string: CSS is full of braces, and Template only
# treats $name as a placeholder so the stylesheet stays readable.
STYLESHEET = Template(
    """
    <style>
    .stApp { background: $appBackground; color: $text; }

    /* Streamlit paints its own text colour on these, so restate it against
       whichever background the active theme gives us. */
    .stApp, .stApp p, .stApp li, .stApp label,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] { color: $text; }

    [data-testid="stSidebar"] { background: $sidebarBackground; }
    [data-testid="stSidebar"] * { color: $text !important; }
    .block-container { padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1500px; }

    /* ---------- Masthead ---------- */
    .masthead {
        display: flex;
        align-items: baseline;
        gap: 14px;
        flex-wrap: wrap;
        padding: 0 0 6px 0;
        margin-bottom: 4px;
    }
    .masthead .brand {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -.02em;
        color: $heading;
        line-height: 1.1;
    }
    .masthead .tagline {
        font-size: .95rem;
        color: $muted;
        font-weight: 500;
    }
    .page-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: $heading;
        margin: 4px 0 2px 0;
        letter-spacing: -.01em;
    }
    .page-sub { color: $muted; font-size: .9rem; margin-bottom: 18px; }
    .rule { height: 1px; background: $border; margin: 6px 0 20px 0; }

    /* ---------- Section headings: a real hierarchy ---------- */
    .section-title {
        margin-top: 30px;
        margin-bottom: 2px;
        font-size: 1.05rem;
        font-weight: 700;
        color: $heading;
        letter-spacing: -.01em;
    }
    .section-question {
        margin-bottom: 12px;
        font-size: .84rem;
        color: $muted;
        font-style: italic;
    }

    /* ---------- KPI tiles ---------- */
    .kpi {
        padding: 16px 18px;
        border-radius: 14px;
        background: $surface;
        border: 1px solid $border;
        box-shadow: $cardShadow;
        height: 100%;
    }
    .kpi-label {
        font-size: .72rem;
        text-transform: uppercase;
        letter-spacing: .08em;
        color: $muted;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: $heading;
        line-height: 1.1;
        letter-spacing: -.02em;
    }
    .kpi-note { font-size: .76rem; color: $muted; margin-top: 5px; }
    .kpi-critical { border-left: 4px solid #d03b3b; }
    .kpi-warning  { border-left: 4px solid #fab219; }
    .kpi-good     { border-left: 4px solid #0ca30c; }
    .kpi-neutral  { border-left: 4px solid $series1; }

    /* ---------- Risk banner ---------- */
    .risk-banner {
        padding: 20px 24px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        margin: 6px 0 4px 0;
    }
    .risk-banner .rb-left { display: flex; align-items: center; gap: 16px; }
    .risk-banner .rb-icon { font-size: 1.6rem; }
    .risk-banner .rb-band {
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: .02em;
    }
    .risk-banner .rb-cust { font-size: .85rem; color: $muted; }
    .risk-banner .rb-prob { font-size: 2.3rem; font-weight: 800; line-height: 1; }

    .card {
        padding: 18px;
        border-radius: 14px;
        background: $surface;
        border: 1px solid $border;
        box-shadow: $cardShadow;
    }
    .muted { color: $muted; }
    .fineprint { font-size: .78rem; color: $muted; }

    /* ---------- Action list ---------- */
    .action {
        padding: 12px 14px;
        border-radius: 10px;
        background: $surfaceSoft;
        border: 1px solid $border;
        border-left: 3px solid $series1;
        margin-bottom: 8px;
        font-size: .92rem;
    }
    .action b { color: $heading; }

    /* ---------- Buttons: give the primary action real weight ---------- */
    div[data-testid="stMetric"] {
        background: $surfaceSoft;
        border: 1px solid $border;
        padding: 12px;
        border-radius: 12px;
    }
    </style>
    """
).safe_substitute(PALETTE)

st.markdown(STYLESHEET, unsafe_allow_html=True)


# Altair chrome matching the active theme. Charts render on the app's
# translucent surface, so the view background stays transparent.
def style_chart(chart, height=260):
    return (
        chart.properties(height=height, background="transparent")
        .configure_view(strokeWidth=0, fill=None)
        .configure_axis(
            labelColor=PALETTE["chartMuted"],
            titleColor=PALETTE["chartMuted"],
            gridColor=PALETTE["chartGrid"],
            domainColor=PALETTE["chartAxis"],
            tickColor=PALETTE["chartAxis"],
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight="normal",
        )
        .configure_legend(
            labelColor=PALETTE["text"],
            titleColor=PALETTE["chartMuted"],
            labelFontSize=11,
            titleFontSize=11,
        )
    )


def section(title, question=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if question:
        st.markdown(
            f'<div class="section-question">{question}</div>',
            unsafe_allow_html=True,
        )


def kpi_tile(label, value, note="", tone="neutral"):
    st.markdown(
        f"""
        <div class="kpi kpi-{tone}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def money(amount):
    return f"${amount:,.0f}"


# ==========================================================================
# Database
# ==========================================================================

CASE_STATUSES = [
    "Open",
    "Contacted",
    "Offer Made",
    "Retained",
    "Lost",
]

OFFER_TYPES = [
    "Annual contract migration + incentive",
    "15% loyalty discount",
    "Complimentary tech support (3 months)",
    "Online security bundle",
    "Automatic payment migration incentive",
    "Plan right-sizing review",
    "No offer - relationship call only",
]

RETENTION_OFFICERS = [
    "Unassigned",
    "A. Mensah",
    "B. Okonkwo",
    "C. Adeyemi",
    "D. Ncube",
]


def get_connection():
    return sqlite3.connect(DATABASE_FILE, check_same_thread=False)


def initialize_database():
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS customers (
            customerID TEXT PRIMARY KEY,
            gender TEXT, SeniorCitizen TEXT, Partner TEXT, Dependents TEXT,
            tenure INTEGER, PhoneService TEXT, MultipleLines TEXT,
            InternetService TEXT, OnlineSecurity TEXT, OnlineBackup TEXT,
            DeviceProtection TEXT, TechSupport TEXT, StreamingTV TEXT,
            StreamingMovies TEXT, Contract TEXT, PaperlessBilling TEXT,
            PaymentMethod TEXT, MonthlyCharges REAL, TotalCharges REAL,
            actual_churn TEXT, source TEXT, created_at TEXT
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
            timestamp TEXT, level TEXT, event TEXT,
            customerID TEXT, details TEXT
        )
        """
    )
    # The retention workflow. One row per case, so a customer can be worked
    # more than once over time and the outcome history is preserved.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS interventions (
            case_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customerID TEXT,
            opened_at TEXT,
            updated_at TEXT,
            risk_level TEXT,
            churn_probability REAL,
            monthly_value REAL,
            main_driver TEXT,
            assigned_to TEXT,
            status TEXT,
            offer_type TEXT,
            contact_attempted INTEGER DEFAULT 0,
            offer_made INTEGER DEFAULT 0,
            customer_accepted INTEGER DEFAULT 0,
            notes TEXT
        )
        """
    )
    # Indexes matter once the registry holds thousands of rows.
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_scans_customer ON scans(customerID)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_cases_customer ON interventions(customerID)"
    )
    connection.commit()
    connection.close()


def log_event(level, event, customer_id="", details="", connection=None):
    # When a connection is supplied the caller owns the transaction, so this
    # function neither commits nor closes. Each commit is an fsync, and doing
    # one per row is what made cold start crawl on Streamlit Cloud's disk.
    owns_connection = connection is None
    if owns_connection:
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

    if owns_connection:
        connection.commit()
        connection.close()


def insert_customer(customer, actual_churn, source, connection=None):
    owns_connection = connection is None
    if owns_connection:
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

    if owns_connection:
        connection.commit()
        connection.close()


def save_scan(customer_id, probability, prediction, risk_level, connection=None):
    owns_connection = connection is None
    if owns_connection:
        connection = get_connection()

    connection.execute(
        """
        INSERT INTO scans(
            customerID, churn_probability, prediction,
            threshold, risk_level, scanned_at
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

    if owns_connection:
        connection.commit()
        connection.close()


# ---------- Retention case CRUD ----------


def get_open_case(customer_id):
    connection = get_connection()
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        """
        SELECT * FROM interventions
        WHERE customerID = ?
        ORDER BY case_id DESC
        LIMIT 1
        """,
        (customer_id,),
    ).fetchone()
    connection.close()
    return dict(row) if row else None


def open_case(customer_id, risk_level, probability, monthly_value, main_driver,
              assigned_to="Unassigned", status="Open", offer_type=""):
    """Create a case, or return the existing one so a double click cannot
    create duplicate work for the same customer."""
    existing = get_open_case(customer_id)
    if existing and existing["status"] not in ("Retained", "Lost"):
        return existing["case_id"], False

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    connection = get_connection()
    cursor = connection.execute(
        """
        INSERT INTO interventions(
            customerID, opened_at, updated_at, risk_level, churn_probability,
            monthly_value, main_driver, assigned_to, status, offer_type,
            contact_attempted, offer_made, customer_accepted, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, '')
        """,
        (
            customer_id, now, now, risk_level, float(probability),
            float(monthly_value), main_driver, assigned_to, status, offer_type,
        ),
    )
    case_id = cursor.lastrowid
    log_event(
        "INFO", "Retention case opened", customer_id=customer_id,
        details=f"case={case_id}; risk={risk_level}; status={status}",
        connection=connection,
    )
    connection.commit()
    connection.close()
    return case_id, True


def update_case(case_id, **fields):
    if not fields:
        return
    fields["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    assignments = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values()) + [case_id]

    connection = get_connection()
    connection.execute(
        f"UPDATE interventions SET {assignments} WHERE case_id = ?", values
    )
    log_event(
        "INFO", "Retention case updated",
        details=f"case={case_id}; " + "; ".join(f"{k}={v}" for k, v in fields.items()),
        connection=connection,
    )
    connection.commit()
    connection.close()


# ==========================================================================
# Model layer
# ==========================================================================

MODEL_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges", "gender", "SeniorCitizen",
    "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

# Human labels for the model's raw feature names.
FEATURE_LABELS = {
    "tenure": "Tenure", "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Lifetime spend", "gender": "Gender",
    "SeniorCitizen": "Senior citizen", "Partner": "Has partner",
    "Dependents": "Has dependents", "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines", "InternetService": "Internet service",
    "OnlineSecurity": "Online security", "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection", "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
    "Contract": "Contract type", "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
}


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    model = joblib.load(MODEL_FILE)
    return preprocessor, model


def transform_features(input_df, preprocessor):
    # The preprocessor returns a bare NumPy array, but the classifier was
    # fitted on a named DataFrame. Feeding it the raw array makes scikit-learn
    # emit "X does not have valid feature names" on every prediction.
    transformed = preprocessor.transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return pd.DataFrame(transformed, columns=preprocessor.get_feature_names_out())


def risk_band(probability):
    if probability >= HIGH_RISK_THRESHOLD:
        return "High"
    if probability >= CHURN_THRESHOLD:
        return "Medium"
    return "Low"


def score_frame(input_df, preprocessor, model):
    """Vectorised scoring: one predict_proba call for any number of rows."""
    transformed = transform_features(input_df, preprocessor)
    return model.predict_proba(transformed)[:, 1]


def predict_customer(input_df, preprocessor, model):
    probability = float(score_frame(input_df, preprocessor, model)[0])
    return int(probability >= CHURN_THRESHOLD), probability, risk_band(probability)


def get_churn_probability(input_df, preprocessor, model):
    return float(score_frame(input_df, preprocessor, model)[0])


@st.cache_data(show_spinner=False)
def global_churn_drivers(_model, _preprocessor):
    """Aggregate the forest's importances over one-hot columns back to the
    original business features, so 'Top Churn Drivers' is the model's own
    ranking rather than a hand-written opinion."""
    try:
        importances = _model.feature_importances_
        names = list(_preprocessor.get_feature_names_out())
    except Exception:
        return pd.DataFrame()

    totals = {}
    for name, importance in zip(names, importances):
        # Names look like "numerical__tenure" or "categorical__Contract_Two year".
        bare = name.split("__", 1)[-1]
        matched = None
        for feature in MODEL_FEATURES:
            if bare == feature or bare.startswith(feature + "_"):
                # Prefer the longest match so "Streaming TV" cannot be
                # shadowed by a shorter feature name.
                if matched is None or len(feature) > len(matched):
                    matched = feature
        if matched:
            totals[matched] = totals.get(matched, 0.0) + float(importance)

    if not totals:
        return pd.DataFrame()

    total = sum(totals.values()) or 1.0
    frame = pd.DataFrame(
        [
            {"Driver": FEATURE_LABELS.get(k, k), "Importance": v / total}
            for k, v in totals.items()
        ]
    )
    return frame.sort_values("Importance", ascending=False).reset_index(drop=True)


# The per-customer driver ladder. Each entry is (feature, test, label).
# Running the full sensitivity analysis for every customer would mean roughly
# 40 model calls each - fine for one customer on the detail page, far too slow
# for a whole portfolio. So the portfolio uses this ladder, ordered at runtime
# by the model's own global importances, and reports the highest-ranked risk
# condition the customer actually meets.
DRIVER_RULES = [
    ("Contract", lambda c: c.get("Contract") == "Month-to-month",
     "Month-to-month contract"),
    ("tenure", lambda c: float(c.get("tenure") or 0) < 12,
     "Low tenure (under 12 months)"),
    ("InternetService", lambda c: c.get("InternetService") == "Fiber optic",
     "Fiber optic service"),
    ("PaymentMethod", lambda c: c.get("PaymentMethod") == "Electronic check",
     "Electronic check payment"),
    ("TechSupport", lambda c: c.get("TechSupport") == "No",
     "No tech support"),
    ("OnlineSecurity", lambda c: c.get("OnlineSecurity") == "No",
     "No online security"),
    ("MonthlyCharges", lambda c: float(c.get("MonthlyCharges") or 0) > 70,
     "High monthly charges"),
    ("PaperlessBilling", lambda c: c.get("PaperlessBilling") == "Yes",
     "Paperless billing"),
]


def build_driver_ladder(drivers_frame):
    """Order the rules by the model's global importance for their feature."""
    if drivers_frame is None or drivers_frame.empty:
        return DRIVER_RULES

    rank = {row["Driver"]: row["Importance"] for _, row in drivers_frame.iterrows()}
    return sorted(
        DRIVER_RULES,
        key=lambda rule: rank.get(FEATURE_LABELS.get(rule[0], rule[0]), 0.0),
        reverse=True,
    )


def primary_driver(customer, ladder):
    for _feature, test, label in ladder:
        try:
            if test(customer):
                return label
        except Exception:
            continue
    return "No dominant risk factor"


def get_customer_signals(input_df, preprocessor, model):
    """Per-customer sensitivity analysis: how much would the probability move
    if this one attribute changed? Expensive, so detail pages only."""
    base_probability = get_churn_probability(input_df, preprocessor, model)
    signals = []

    for feature in MODEL_FEATURES:
        original_value = input_df.iloc[0][feature]
        alternatives = []

        if feature == "tenure":
            alternatives = [0, 24, 48, 72]
        elif feature == "MonthlyCharges":
            alternatives = [max(0.0, float(original_value) - 20.0),
                            float(original_value) + 20.0]
        elif feature == "TotalCharges":
            alternatives = [max(0.0, float(original_value) * 0.75),
                            float(original_value) * 1.25]
        elif original_value in ["Yes", "No"]:
            alternatives = ["Yes" if original_value == "No" else "No"]
        elif original_value in ["0", "1"]:
            alternatives = ["1" if original_value == "0" else "0"]
        elif feature == "Contract":
            alternatives = [v for v in ["Month-to-month", "One year", "Two year"]
                            if v != original_value]
        elif feature == "PaymentMethod":
            alternatives = [v for v in ["Electronic check", "Mailed check",
                                        "Bank transfer (automatic)",
                                        "Credit card (automatic)"]
                            if v != original_value]
        elif feature == "InternetService":
            alternatives = [v for v in ["DSL", "Fiber optic", "No"]
                            if v != original_value]
        elif feature == "MultipleLines":
            alternatives = [v for v in ["No", "Yes", "No phone service"]
                            if v != original_value]

        best = None
        for alternative in alternatives:
            modified = input_df.copy()
            modified.loc[modified.index[0], feature] = alternative
            try:
                modified_probability = get_churn_probability(
                    modified, preprocessor, model
                )
            except Exception:
                continue
            delta = base_probability - modified_probability
            if best is None or abs(delta) > abs(best[1]):
                best = (alternative, delta)

        if best is not None and abs(best[1]) > 0.001:
            alternative, delta = best
            signals.append({
                "Factor": FEATURE_LABELS.get(feature, feature),
                "Current value": str(original_value),
                "Best alternative": str(alternative),
                # Positive means changing it REDUCES churn probability.
                "Risk reduction": round(delta, 4),
            })

    if not signals:
        return pd.DataFrame()

    frame = pd.DataFrame(signals)
    frame["abs"] = frame["Risk reduction"].abs()
    return (
        frame.sort_values("abs", ascending=False)
        .drop(columns="abs")
        .head(6)
        .reset_index(drop=True)
    )


def load_customer_for_prediction(customer):
    return pd.DataFrame([{f: customer[f] for f in MODEL_FEATURES}])


def build_model_input(values):
    """values is a dict of the sidebar/form widgets keyed by model feature."""
    row = dict(values)
    row["SeniorCitizen"] = "1" if row.get("SeniorCitizen") == "Yes (1)" else "0"
    return pd.DataFrame([{f: row[f] for f in MODEL_FEATURES}])


# ---------- Costed retention playbook ----------

def retention_playbook(customer, probability):
    """Return a list of (action, rationale, monthly_cost) for this customer.
    Costs are illustrative planning assumptions, labelled as such in the UI."""
    monthly = float(customer.get("MonthlyCharges") or 0.0)
    actions = []

    if probability < CHURN_THRESHOLD:
        actions.append((
            "Maintain standard engagement",
            "Below the action threshold - monitor and re-score if the "
            "contract, usage or billing profile changes.",
            0.0,
        ))
        return actions

    if customer.get("Contract") == "Month-to-month":
        actions.append((
            "Offer migration to an annual contract with a retention incentive",
            "Month-to-month is the single strongest churn driver in the model.",
            round(monthly * 0.10, 2),
        ))
    if customer.get("TechSupport") == "No":
        actions.append((
            "Offer 3 months complimentary tech support",
            "Customers without tech support churn materially more often.",
            round(monthly * 0.05, 2),
        ))
    if (customer.get("InternetService") == "Fiber optic"
            and customer.get("OnlineSecurity") == "No"):
        actions.append((
            "Bundle online security at a discount",
            "Fiber customers without security show elevated dissatisfaction.",
            round(monthly * 0.04, 2),
        ))
    if customer.get("PaymentMethod") == "Electronic check":
        actions.append((
            "Migrate to automatic bank transfer or card payment",
            "Electronic-check payers churn more than automatic-payment customers.",
            0.0,
        ))
    if monthly > 70:
        actions.append((
            "Run a plan right-sizing review with a loyalty discount",
            f"At {money(monthly)} per month this customer is above the "
            "portfolio average and price sensitive.",
            round(monthly * 0.15, 2),
        ))
    if float(customer.get("tenure") or 0) < 12:
        actions.append((
            "Assign to a retention specialist within 24 hours",
            "Early-lifecycle customers are the most winnable and the most volatile.",
            0.0,
        ))

    if not actions:
        actions.append((
            "Assign a retention specialist for a discovery call",
            "No single dominant driver - diagnose before making an offer.",
            0.0,
        ))
    return actions


# ==========================================================================
# Seeding and bootstrap
# ==========================================================================

def seed_customer_registry():
    connection = get_connection()
    try:
        count = connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        if count > 0 or not os.path.exists(SEED_FILE):
            return

        seed_df = pd.read_csv(SEED_FILE)

        # The published Telco extract carries TotalCharges as text with blanks
        # for brand-new accounts; coerce so the model never sees a NaN string.
        if "TotalCharges" in seed_df.columns:
            seed_df["TotalCharges"] = pd.to_numeric(
                seed_df["TotalCharges"], errors="coerce"
            ).fillna(0.0)
        if "Churn" not in seed_df.columns:
            seed_df["Churn"] = "Unknown"

        for _, customer in seed_df.iterrows():
            insert_customer(
                customer, str(customer["Churn"]),
                "Historical seed dataset", connection=connection,
            )

        log_event(
            "INFO", "Customer registry seeded",
            details=f"{len(seed_df)} customers loaded from {os.path.basename(SEED_FILE)}.",
            connection=connection,
        )
        connection.commit()
    finally:
        connection.close()


def generate_seed_scans(preprocessor, model):
    connection = get_connection()
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT * FROM customers WHERE source = 'Historical seed dataset'"
        ).fetchall()

        already_scanned = {
            row["customerID"]
            for row in connection.execute(
                "SELECT DISTINCT customerID FROM scans"
            ).fetchall()
        }

        pending = [
            dict(row) for row in rows
            if str(row["customerID"]) not in already_scanned
        ]
        if not pending:
            return

        # Score everything in one vectorised call rather than one per customer.
        model_input = pd.concat(
            [load_customer_for_prediction(c) for c in pending], ignore_index=True
        )
        probabilities = score_frame(model_input, preprocessor, model)

        for customer, probability in zip(pending, probabilities):
            probability = float(probability)
            save_scan(
                str(customer["customerID"]), probability,
                int(probability >= CHURN_THRESHOLD), risk_band(probability),
                connection=connection,
            )

        log_event(
            "INFO", "Portfolio scored",
            details=f"{len(pending)} customers scored at threshold "
                    f"{CHURN_THRESHOLD:.0%}.",
            connection=connection,
        )
        connection.commit()
    finally:
        connection.close()


@st.cache_resource(show_spinner="Scoring the customer portfolio...")
def bootstrap():
    # Streamlit re-executes the whole script on every interaction, so this
    # runs once per container rather than once per click.
    initialize_database()
    seed_customer_registry()
    try:
        preprocessor, model = load_artifacts()
        generate_seed_scans(preprocessor, model)
        return preprocessor, model, True, ""
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        log_event("ERROR", "Model artifacts could not be loaded", details=message)
        return None, None, False, message


preprocessor, model, artifacts_loaded, artifact_error = bootstrap()

DRIVERS_FRAME = (
    global_churn_drivers(model, preprocessor) if artifacts_loaded else pd.DataFrame()
)
DRIVER_LADDER = build_driver_ladder(DRIVERS_FRAME)


# ==========================================================================
# Portfolio data layer
# ==========================================================================

def data_version():
    """Cheap change token so cached frames refresh after a write without
    caching stale results for the life of the container."""
    connection = get_connection()
    try:
        scans = connection.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        cases = connection.execute(
            "SELECT COUNT(*) FROM interventions"
        ).fetchone()[0]
        touched = connection.execute(
            "SELECT COALESCE(MAX(updated_at), '') FROM interventions"
        ).fetchone()[0]
    finally:
        connection.close()
    return f"{scans}-{cases}-{touched}"


@st.cache_data(show_spinner=False)
def portfolio(version):
    """One row per customer: profile + latest score + latest case status."""
    connection = get_connection()
    try:
        frame = pd.read_sql_query(
            """
            SELECT c.*,
                   s.churn_probability,
                   s.risk_level,
                   s.prediction,
                   s.scanned_at
            FROM customers c
            LEFT JOIN (
                SELECT s1.*
                FROM scans s1
                JOIN (
                    SELECT customerID, MAX(scan_id) AS scan_id
                    FROM scans GROUP BY customerID
                ) latest
                ON s1.scan_id = latest.scan_id
            ) s ON s.customerID = c.customerID
            """,
            connection,
        )
        cases = pd.read_sql_query(
            """
            SELECT i1.customerID, i1.status, i1.assigned_to, i1.offer_type,
                   i1.case_id, i1.customer_accepted
            FROM interventions i1
            JOIN (
                SELECT customerID, MAX(case_id) AS case_id
                FROM interventions GROUP BY customerID
            ) latest ON i1.case_id = latest.case_id
            """,
            connection,
        )
    finally:
        connection.close()

    if frame.empty:
        return frame

    frame = frame.merge(cases, on="customerID", how="left")
    frame["churn_probability"] = frame["churn_probability"].fillna(0.0)
    frame["risk_level"] = frame["risk_level"].fillna("Low")
    frame["MonthlyCharges"] = pd.to_numeric(
        frame["MonthlyCharges"], errors="coerce"
    ).fillna(0.0)
    frame["tenure"] = pd.to_numeric(frame["tenure"], errors="coerce").fillna(0)

    frame["monthly_value"] = frame["MonthlyCharges"]
    # Expected loss: the risk-adjusted figure a finance team would accept.
    frame["expected_loss"] = frame["MonthlyCharges"] * frame["churn_probability"]
    frame["at_risk"] = frame["churn_probability"] >= CHURN_THRESHOLD
    frame["status"] = frame["status"].fillna("No case")
    frame["assigned_to"] = frame["assigned_to"].fillna("Unassigned")

    # Per-customer driver via the importance-ordered rule ladder.
    frame["main_driver"] = [
        primary_driver(row, DRIVER_LADDER)
        for row in frame.to_dict(orient="records")
    ]

    # Segments used by the analytics page.
    frame["tenure_band"] = pd.cut(
        frame["tenure"],
        bins=[-1, 6, 12, 24, 48, 1000],
        labels=["0-6 m", "7-12 m", "13-24 m", "25-48 m", "49+ m"],
    ).astype(str)
    frame["charge_band"] = pd.cut(
        frame["MonthlyCharges"],
        bins=[-1, 35, 60, 80, 100, 10000],
        labels=["<$35", "$35-60", "$60-80", "$80-100", "$100+"],
    ).astype(str)

    def segment(row):
        if row["MonthlyCharges"] >= 80 and row["tenure"] >= 24:
            return "High value / loyal"
        if row["MonthlyCharges"] >= 80:
            return "High value / new"
        if row["tenure"] >= 24:
            return "Standard / loyal"
        return "Standard / new"

    frame["segment"] = frame.apply(segment, axis=1)
    return frame


@st.cache_data(show_spinner=False)
def case_log(version):
    connection = get_connection()
    try:
        return pd.read_sql_query(
            "SELECT * FROM interventions ORDER BY case_id DESC", connection
        )
    finally:
        connection.close()


def portfolio_kpis(frame):
    total = len(frame)
    if total == 0:
        return {}

    at_risk = frame[frame["at_risk"]]
    gross = float(at_risk["monthly_value"].sum())
    expected = float(frame["expected_loss"].sum())

    cases = frame[frame["status"] != "No case"]
    retained = int((frame["status"] == "Retained").sum())
    resolved = int(frame["status"].isin(["Retained", "Lost"]).sum())

    known = frame[frame["actual_churn"].isin(["Yes", "No"])]
    base_rate = (
        float((known["actual_churn"] == "Yes").mean()) if len(known) else None
    )

    return {
        "total": total,
        "known": len(known),
        "base_rate": base_rate,
        "at_risk": len(at_risk),
        "risk_rate": len(at_risk) / total,
        "gross_exposure": gross,
        "expected_loss": expected,
        "open_cases": int((~frame["status"].isin(["No case", "Retained", "Lost"])).sum()),
        "cases": len(cases),
        "retained": retained,
        "resolved": resolved,
        "win_rate": (retained / resolved) if resolved else 0.0,
        "revenue_protected": float(
            frame.loc[frame["status"] == "Retained", "monthly_value"].sum()
        ),
    }


# ==========================================================================
# Charts
#
# Form follows the data's job. Risk bands use the fixed status palette and
# always carry a text label, because amber is deliberately low-contrast on a
# light surface. Everything else is single-series magnitude, so it uses one
# hue and needs no legend - the title names the series.
# ==========================================================================

TENURE_ORDER = ["0-6 m", "7-12 m", "13-24 m", "25-48 m", "49+ m"]
CHARGE_ORDER = ["<$35", "$35-60", "$60-80", "$80-100", "$100+"]


def _grouped(frame, dimension):
    grouped = (
        frame.groupby(dimension)
        .agg(
            customers=("customerID", "count"),
            at_risk=("at_risk", "sum"),
            revenue=("monthly_value", "sum"),
            expected=("expected_loss", "sum"),
        )
        .reset_index()
    )
    grouped["at_risk"] = grouped["at_risk"].astype(int)
    grouped["rate"] = grouped["at_risk"] / grouped["customers"].replace(0, np.nan)
    grouped["rate"] = grouped["rate"].fillna(0.0)
    grouped = grouped.rename(columns={dimension: "category"})
    grouped["category"] = grouped["category"].astype(str)
    return grouped


def risk_rate_chart(frame, dimension, order=None, height=260):
    """Share of customers at or above the action threshold, by segment."""
    grouped = _grouped(frame, dimension)
    if grouped.empty:
        return None

    sort = order if order else "-y"
    x = alt.X("category:N", title=None, sort=sort,
              axis=alt.Axis(labelAngle=0, labelLimit=140))

    base = alt.Chart(grouped)
    bars = base.mark_bar(cornerRadiusEnd=4, size=38,
                         color=PALETTE["series1"]).encode(
        x=x,
        y=alt.Y("rate:Q", title="At-risk share",
                axis=alt.Axis(format="%"), scale=alt.Scale(domainMin=0)),
        tooltip=[
            alt.Tooltip("category:N", title="Segment"),
            alt.Tooltip("rate:Q", title="At-risk share", format=".1%"),
            alt.Tooltip("at_risk:Q", title="At-risk customers", format=","),
            alt.Tooltip("customers:Q", title="Customers", format=","),
            alt.Tooltip("revenue:Q", title="Monthly revenue", format="$,.0f"),
        ],
    )
    labels = base.mark_text(dy=-8, fontSize=11, fontWeight=600,
                            color=PALETTE["text"]).encode(
        x=x,
        y=alt.Y("rate:Q"),
        text=alt.Text("rate:Q", format=".0%"),
    )
    return style_chart(bars + labels, height=height)


def revenue_chart(frame, dimension, order=None, height=260):
    """Monthly revenue sitting in at-risk accounts, by segment."""
    grouped = _grouped(frame, dimension)
    if grouped.empty:
        return None

    sort = order if order else "-y"
    x = alt.X("category:N", title=None, sort=sort,
              axis=alt.Axis(labelAngle=0, labelLimit=140))

    base = alt.Chart(grouped)
    bars = base.mark_bar(cornerRadiusEnd=4, size=38,
                         color=PALETTE["series2"]).encode(
        x=x,
        y=alt.Y("expected:Q", title="Expected monthly loss ($)",
                scale=alt.Scale(domainMin=0)),
        tooltip=[
            alt.Tooltip("category:N", title="Segment"),
            alt.Tooltip("expected:Q", title="Expected monthly loss", format="$,.0f"),
            alt.Tooltip("revenue:Q", title="Gross monthly revenue", format="$,.0f"),
            alt.Tooltip("at_risk:Q", title="At-risk customers", format=","),
        ],
    )
    labels = base.mark_text(dy=-8, fontSize=11, fontWeight=600,
                            color=PALETTE["text"]).encode(
        x=x, y=alt.Y("expected:Q"),
        text=alt.Text("expected:Q", format="$,.0f"),
    )
    return style_chart(bars + labels, height=height)


def risk_distribution_chart(frame, height=210):
    counts = (
        frame.groupby("risk_level")
        .agg(customers=("customerID", "count"),
             revenue=("monthly_value", "sum"))
        .reindex(RISK_ORDER)
        .fillna(0)
        .reset_index()
    )
    counts["customers"] = counts["customers"].astype(int)
    total = counts["customers"].sum() or 1
    counts["share"] = counts["customers"] / total
    counts["label"] = counts.apply(
        lambda r: f"{r['customers']:,.0f}  ({r['share']:.0%})", axis=1
    )

    y = alt.Y("risk_level:N", title=None, sort=RISK_ORDER,
              axis=alt.Axis(labelFontWeight=600, labelFontSize=12))
    base = alt.Chart(counts)
    bars = base.mark_bar(cornerRadiusEnd=4, size=30).encode(
        y=y,
        x=alt.X("customers:Q", title="Customers", scale=alt.Scale(domainMin=0)),
        # Status palette, fixed in both themes.
        color=alt.Color(
            "risk_level:N",
            scale=alt.Scale(domain=RISK_ORDER,
                            range=[RISK_COLORS[r] for r in RISK_ORDER]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("risk_level:N", title="Risk band"),
            alt.Tooltip("customers:Q", title="Customers", format=","),
            alt.Tooltip("share:Q", title="Share", format=".1%"),
            alt.Tooltip("revenue:Q", title="Monthly revenue", format="$,.0f"),
        ],
    )
    # Direct labels: the relief rule for the amber step on a light surface.
    labels = base.mark_text(align="left", dx=6, fontSize=11, fontWeight=600,
                            color=PALETTE["text"]).encode(
        y=y, x=alt.X("customers:Q"), text=alt.Text("label:N"),
    )
    return style_chart(bars + labels, height=height)


def drivers_chart(drivers_frame, height=260, top=8):
    if drivers_frame is None or drivers_frame.empty:
        return None
    frame = drivers_frame.head(top).copy()

    y = alt.Y("Driver:N", title=None, sort="-x",
              axis=alt.Axis(labelLimit=180))
    base = alt.Chart(frame)
    bars = base.mark_bar(cornerRadiusEnd=4, size=18,
                         color=PALETTE["series1"]).encode(
        y=y,
        x=alt.X("Importance:Q", title="Share of model importance",
                axis=alt.Axis(format="%"), scale=alt.Scale(domainMin=0)),
        tooltip=[
            alt.Tooltip("Driver:N"),
            alt.Tooltip("Importance:Q", title="Model importance", format=".1%"),
        ],
    )
    labels = base.mark_text(align="left", dx=6, fontSize=11,
                            color=PALETTE["text"]).encode(
        y=y, x=alt.X("Importance:Q"), text=alt.Text("Importance:Q", format=".1%"),
    )
    return style_chart(bars + labels, height=height)


def lifecycle_chart(frame, height=260):
    """Where in the customer lifecycle risk concentrates."""
    grouped = _grouped(frame, "tenure_band")
    if grouped.empty:
        return None
    grouped["category"] = pd.Categorical(
        grouped["category"], categories=TENURE_ORDER, ordered=True
    )
    grouped = grouped.sort_values("category")
    grouped["category"] = grouped["category"].astype(str)

    x = alt.X("category:N", title="Tenure", sort=TENURE_ORDER,
              axis=alt.Axis(labelAngle=0))
    base = alt.Chart(grouped)
    area = base.mark_area(
        line={"color": PALETTE["series1"], "strokeWidth": 2},
        color=alt.Gradient(
            gradient="linear",
            stops=[alt.GradientStop(color=PALETTE["series1"], offset=0),
                   alt.GradientStop(color="rgba(0,0,0,0)", offset=1)],
            x1=1, x2=1, y1=1, y2=0,
        ),
        opacity=0.35,
    ).encode(
        x=x,
        y=alt.Y("rate:Q", title="At-risk share", axis=alt.Axis(format="%"),
                scale=alt.Scale(domainMin=0)),
    )
    points = base.mark_circle(size=90, color=PALETTE["series1"]).encode(
        x=x, y=alt.Y("rate:Q"),
        tooltip=[
            alt.Tooltip("category:N", title="Tenure"),
            alt.Tooltip("rate:Q", title="At-risk share", format=".1%"),
            alt.Tooltip("customers:Q", title="Customers", format=","),
            alt.Tooltip("expected:Q", title="Expected monthly loss", format="$,.0f"),
        ],
    )
    labels = base.mark_text(dy=-14, fontSize=11, fontWeight=600,
                            color=PALETTE["text"]).encode(
        x=x, y=alt.Y("rate:Q"), text=alt.Text("rate:Q", format=".0%"),
    )
    return style_chart(area + points + labels, height=height)


def sample_note(frame):
    n = len(frame)
    if n >= 2000:
        return ""
    return (
        f"Computed over {n:,} customers in the current registry. "
        "Segment cells are small at this sample size - directional, not definitive."
    )


# ==========================================================================
# Navigation and roles
#
# This is a demonstration of role-based views, NOT authentication. Anyone can
# switch role from the sidebar; it shapes what the product shows, so a CEO is
# not asked to navigate system logs. Real deployment would put this behind an
# identity provider.
# ==========================================================================

ROLE_PAGES = {
    "Executive": ["Executive Dashboard", "Analytics"],
    "Retention Manager": [
        "Executive Dashboard", "Customer Registry", "Retention Cases", "Analytics",
    ],
    "Retention Agent": ["Customer Registry", "Customer 360", "Retention Cases"],
    "Data Analyst": [
        "Executive Dashboard", "Analytics", "Customer Registry", "Customer 360",
    ],
    "Administrator": [
        "Executive Dashboard", "Customer Registry", "Customer 360",
        "Retention Cases", "Analytics", "System Logs",
    ],
}

ROLE_BLURB = {
    "Executive": "Portfolio health and financial exposure.",
    "Retention Manager": "Portfolio, prioritisation and campaign outcomes.",
    "Retention Agent": "Your worklist and individual customer cases.",
    "Data Analyst": "Segment analysis and model behaviour.",
    "Administrator": "Full access including system diagnostics.",
}

with st.sidebar:
    st.markdown(f"### 📡 {PRODUCT_NAME}")
    st.caption(PRODUCT_TAGLINE)
    st.markdown("---")

    role = st.selectbox("Signed in as", list(ROLE_PAGES.keys()), index=0)
    st.caption(ROLE_BLURB[role])

    st.markdown("---")
    page = st.radio("Navigate", ROLE_PAGES[role], label_visibility="collapsed")

    st.markdown("---")
    st.caption(
        f"Decision threshold {CHURN_THRESHOLD:.0%} · High-risk band "
        f"{HIGH_RISK_THRESHOLD:.0%}"
    )
    st.caption(BUILT_BY)


st.markdown(
    f"""
    <div class="masthead">
        <span class="brand">{PRODUCT_NAME}</span>
        <span class="tagline">{PRODUCT_TAGLINE}</span>
    </div>
    <div class="rule"></div>
    """,
    unsafe_allow_html=True,
)

if not artifacts_loaded:
    st.error("Model artifacts could not be loaded, so scoring is unavailable.")
    st.code(artifact_error)
    st.caption(f"Expected preprocessor at: {PREPROCESSOR_FILE}")
    st.caption(f"Expected model at: {MODEL_FILE}")
    st.stop()

VERSION = data_version()
FRAME = portfolio(VERSION)

if FRAME.empty:
    st.warning("The customer registry is empty. Check that the seed file exists.")
    st.caption(f"Expected seed data at: {SEED_FILE}")
    st.stop()

KPIS = portfolio_kpis(FRAME)


def page_header(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def risk_pill(level, probability):
    colour = RISK_COLORS.get(level, PALETTE["series1"])
    icon = RISK_ICONS.get(level, "●")
    return (
        f'<span style="color:{colour};font-weight:800;">{icon} {level.upper()} '
        f'RISK</span> <span class="muted">·</span> '
        f'<b>{probability:.0%}</b>'
    )


# ==========================================================================
# Executive dashboard
# ==========================================================================

if page == "Executive Dashboard":
    page_header(
        "Executive Dashboard",
        "Where the customer base is leaking revenue, and what it is worth.",
    )

    row = st.columns(3)
    with row[0]:
        kpi_tile("Total customers", f"{KPIS['total']:,}",
                 "Scored customer registry", "neutral")
    with row[1]:
        kpi_tile("Customers at risk", f"{KPIS['at_risk']:,}",
                 f"At or above the {CHURN_THRESHOLD:.0%} action threshold",
                 "critical")
    with row[2]:
        kpi_tile("Churn risk rate", f"{KPIS['risk_rate']:.1%}",
                 "Share of the base flagged for action", "warning")

    row = st.columns(3)
    with row[0]:
        kpi_tile("Monthly revenue at risk", money(KPIS["gross_exposure"]),
                 "Gross billings of every flagged account", "critical")
    with row[1]:
        kpi_tile("Risk-adjusted exposure", money(KPIS["expected_loss"]),
                 "Revenue × churn probability across the base", "warning")
    with row[2]:
        kpi_tile("Revenue protected", money(KPIS["revenue_protected"]),
                 f"{KPIS['retained']} customer"
                 f"{'' if KPIS['retained'] == 1 else 's'} retained · "
                 f"{KPIS['win_rate']:.0%} win rate", "good")

    st.markdown(
        '<div class="fineprint" style="margin-top:10px;">'
        "Gross revenue at risk is the full monthly billing of every flagged "
        "account and assumes all of them leave. Risk-adjusted exposure weights "
        "each account by its churn probability and is the figure to plan "
        "against. Both are monthly, not annual."
        "</div>",
        unsafe_allow_html=True,
    )

    # A deliberately balanced prototype sample would otherwise read as a
    # catastrophic churn rate. Say so before anyone in the room has to ask.
    if KPIS.get("base_rate") is not None and KPIS["base_rate"] > 0.40:
        st.warning(
            
        )

    left, right = st.columns([1, 1])

    with left:
        section("Churn risk distribution", "How is the base spread across risk bands?")
        chart = risk_distribution_chart(FRAME)
        if chart is not None:
            st.altair_chart(chart, width="stretch")

    with right:
        section("Churn by contract type", "Which commercial terms are we losing on?")
        chart = risk_rate_chart(FRAME, "Contract")
        if chart is not None:
            st.altair_chart(chart, width="stretch")

    left, right = st.columns([1, 1])

    with left:
        section("Top churn drivers", "Why are customers leaving?")
        chart = drivers_chart(DRIVERS_FRAME)
        if chart is not None:
            st.altair_chart(chart, width="stretch")
            st.caption(
                "Ranked by the trained random forest's own feature importances, "
                "aggregated back to business attributes."
            )
        else:
            st.info("Driver importances are unavailable for this model.")

    with right:
        section("Revenue at risk by contract",
                "Where is the money, not just the headcount?")
        chart = revenue_chart(FRAME, "Contract")
        if chart is not None:
            st.altair_chart(chart, width="stretch")

    section("Act today", "Which customers should we work first?")
    priority = (
        FRAME[FRAME["at_risk"]]
        .sort_values("expected_loss", ascending=False)
        .head(10)
        .copy()
    )
    if priority.empty:
        st.info("No customers are currently above the action threshold.")
    else:
        display = pd.DataFrame({
            "Customer": priority["customerID"],
            "Risk": priority["risk_level"],
            "Churn probability": priority["churn_probability"],
            "Monthly value": priority["monthly_value"],
            "Expected loss": priority["expected_loss"],
            "Main driver": priority["main_driver"],
            "Case status": priority["status"],
        })
        st.dataframe(
            display, width="stretch", hide_index=True,
            column_config={
                "Churn probability": st.column_config.ProgressColumn(
                    "Churn probability", format="%.0f%%", min_value=0, max_value=1
                ),
                "Monthly value": st.column_config.NumberColumn(format="$%.2f"),
                "Expected loss": st.column_config.NumberColumn(format="$%.2f"),
            },
        )
        st.caption(
            "Ranked by expected monthly loss, not by probability - a 90% risk on "
            "a $20 account is worth less than a 75% risk on a $300 account."
        )

    note = sample_note(FRAME)
    if note:
        st.caption(note)


# ==========================================================================
# Customer Registry - a prioritised worklist, not a table dump
# ==========================================================================

elif page == "Customer Registry":
    page_header(
        "Customer Registry",
        "The retention worklist, ordered by what each account is worth.",
    )

    controls = st.columns([2, 2, 2, 3])
    with controls[0]:
        bands = st.multiselect("Risk band", RISK_ORDER, default=["High", "Medium"])
    with controls[1]:
        statuses = st.multiselect(
            "Case status", ["No case"] + CASE_STATUSES, default=[]
        )
    with controls[2]:
        sort_by = st.selectbox(
            "Sort by",
            ["Expected loss", "Monthly value", "Churn probability", "Tenure"],
        )
    with controls[3]:
        query = st.text_input("Find customer ID", placeholder="e.g. 6302-JGYRJ")

    view = FRAME.copy()
    if bands:
        view = view[view["risk_level"].isin(bands)]
    if statuses:
        view = view[view["status"].isin(statuses)]
    if query.strip():
        view = view[
            view["customerID"].str.contains(query.strip(), case=False, na=False)
        ]

    sort_column = {
        "Expected loss": "expected_loss",
        "Monthly value": "monthly_value",
        "Churn probability": "churn_probability",
        "Tenure": "tenure",
    }[sort_by]
    view = view.sort_values(sort_column, ascending=False)

    summary = st.columns(4)
    with summary[0]:
        kpi_tile("In this view", f"{len(view):,}", "Customers matching filters",
                 "neutral")
    with summary[1]:
        kpi_tile("Monthly value", money(view["monthly_value"].sum()),
                 "Gross billings in view", "neutral")
    with summary[2]:
        kpi_tile("Expected loss", money(view["expected_loss"].sum()),
                 "Risk-adjusted, per month", "warning")
    with summary[3]:
        kpi_tile("Without a case", f"{int((view['status'] == 'No case').sum()):,}",
                 "No intervention opened yet", "critical")

    section("Worklist", "Who should we save first?")
    display = pd.DataFrame({
        "Customer": view["customerID"],
        "Churn probability": view["churn_probability"],
        "Risk level": view["risk_level"],
        "Monthly value": view["monthly_value"],
        "Expected loss": view["expected_loss"],
        "Main driver": view["main_driver"],
        "Intervention status": view["status"],
        "Owner": view["assigned_to"],
    })
    st.dataframe(
        display, width="stretch", hide_index=True, height=460,
        column_config={
            "Churn probability": st.column_config.ProgressColumn(
                "Churn probability", format="%.0f%%", min_value=0, max_value=1
            ),
            "Monthly value": st.column_config.NumberColumn(format="$%.2f"),
            "Expected loss": st.column_config.NumberColumn(format="$%.2f"),
        },
    )
    st.caption(
        "Expected loss is monthly revenue × churn probability. Sorting by it "
        "rather than by probability is what turns a model output into a "
        "commercial priority list."
    )

    if not view.empty:
        section("Open a case", "Move a customer into the retention workflow.")
        picker = st.columns([3, 2, 2, 2])
        with picker[0]:
            chosen = st.selectbox(
                "Customer", view["customerID"].tolist(), key="registry_pick"
            )
        with picker[1]:
            owner = st.selectbox("Assign to", RETENTION_OFFICERS, index=1)
        with picker[2]:
            offer = st.selectbox("Opening offer", OFFER_TYPES)
        with picker[3]:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("Open retention case", type="primary",
                         width="stretch"):
                record = view[view["customerID"] == chosen].iloc[0]
                _case_id, created = open_case(
                    chosen, record["risk_level"], record["churn_probability"],
                    record["monthly_value"], record["main_driver"],
                    assigned_to=owner, status="Open", offer_type=offer,
                )
                if created:
                    st.success(f"Case opened for {chosen} and assigned to {owner}.")
                else:
                    st.info(f"{chosen} already has an active case.")
                st.rerun()


# ==========================================================================
# Customer 360 - score a customer and act on the result
# ==========================================================================

elif page == "Customer 360":
    page_header(
        "Customer 360",
        "Score a customer, understand the drivers, and start an intervention.",
    )

    lookup, scorer = st.tabs(["Look up a scored customer", "Score a new profile"])

    def render_result(customer_id, probability, level, customer_row,
                      signals_input, key_prefix):
        colour = RISK_COLORS.get(level, PALETTE["series1"])
        icon = RISK_ICONS.get(level, "●")
        monthly = float(customer_row.get("MonthlyCharges") or 0.0)

        st.markdown(
            f"""
            <div class="risk-banner" style="
                background:{colour}1f;
                border:1px solid {colour}66;
                border-left:6px solid {colour};">
                <div class="rb-left">
                    <div class="rb-icon" style="color:{colour};">{icon}</div>
                    <div>
                        <div class="rb-band" style="color:{colour};">
                            {level.upper()} RISK
                        </div>
                        <div class="rb-cust">Customer {customer_id}</div>
                    </div>
                </div>
                <div class="rb-prob" style="color:{colour};">{probability:.0%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        playbook = retention_playbook(customer_row, probability)
        programme_cost = sum(cost for _a, _r, cost in playbook)
        protected = monthly * RETENTION_HORIZON_MONTHS if probability >= CHURN_THRESHOLD else 0.0

        figures = st.columns(4)
        with figures[0]:
            kpi_tile("Monthly value", money(monthly), "Current billing", "neutral")
        with figures[1]:
            kpi_tile("Expected monthly loss", money(monthly * probability),
                     "Billing × churn probability", "warning")
        with figures[2]:
            kpi_tile("Revenue protected if retained", money(protected),
                     f"Over {RETENTION_HORIZON_MONTHS} months", "good")
        with figures[3]:
            kpi_tile("Estimated offer cost", money(programme_cost),
                     "Per month, planning assumption", "neutral")

        section("Recommended intervention", "What should we do about it?")
        for action, rationale, cost in playbook:
            cost_text = (
                f"<span class='muted'> · est. {money(cost)}/mo</span>"
                if cost else "<span class='muted'> · no direct cost</span>"
            )
            st.markdown(
                f"<div class='action'><b>{action}</b>{cost_text}<br>"
                f"<span class='muted'>{rationale}</span></div>",
                unsafe_allow_html=True,
            )
        st.caption(
            "Offer costs are planning assumptions expressed as a share of the "
            "customer's monthly bill, not quoted commercial terms."
        )

        section("Why this customer", "Which factors are moving the score?")
        signals = get_customer_signals(signals_input, preprocessor, model)
        if signals.empty:
            st.info("No individual factor moves this customer's score materially.")
        else:
            st.dataframe(
                signals, width="stretch", hide_index=True,
                column_config={
                    "Risk reduction": st.column_config.NumberColumn(
                        "Probability change if changed", format="%+.1f%%"
                    )
                },
            )
            st.caption(
                "Each row re-scores this customer with one attribute changed. "
                "A positive value means the change would lower churn risk."
            )

        # ---- Workflow actions ----
        section("Retention workflow", "Move this customer through the process.")
        existing = get_open_case(customer_id)

        if existing:
            st.markdown(
                f"<div class='card'><b>Case #{existing['case_id']}</b> · "
                f"status <b>{existing['status']}</b> · owner "
                f"<b>{existing['assigned_to']}</b><br>"
                f"<span class='muted'>Offer: {existing['offer_type'] or 'none yet'}"
                f"</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        buttons = st.columns(4)
        with buttons[0]:
            if st.button("Assign to Retention", width="stretch",
                         type="primary", key=f"{key_prefix}_act_assign"):
                open_case(customer_id, level, probability, monthly,
                          primary_driver(customer_row, DRIVER_LADDER),
                          assigned_to=RETENTION_OFFICERS[1], status="Open")
                st.rerun()
        with buttons[1]:
            if st.button("Create Case", width="stretch", key=f"{key_prefix}_act_case"):
                open_case(customer_id, level, probability, monthly,
                          primary_driver(customer_row, DRIVER_LADDER))
                st.rerun()
        with buttons[2]:
            if st.button("Contact Customer", width="stretch",
                         key=f"{key_prefix}_act_contact"):
                case = existing or {"case_id": open_case(
                    customer_id, level, probability, monthly,
                    primary_driver(customer_row, DRIVER_LADDER))[0]}
                update_case(case["case_id"], contact_attempted=1,
                            status="Contacted")
                st.rerun()
        with buttons[3]:
            if st.button("Mark Intervention", width="stretch",
                         key=f"{key_prefix}_act_offer"):
                case = existing or {"case_id": open_case(
                    customer_id, level, probability, monthly,
                    primary_driver(customer_row, DRIVER_LADDER))[0]}
                update_case(case["case_id"], offer_made=1, status="Offer Made",
                            offer_type=playbook[0][0])
                st.rerun()

        st.caption(
            "Cases are stored in the application database. On Streamlit "
            "Community Cloud that database resets when the container restarts."
        )

    with lookup:
        options = FRAME.sort_values("expected_loss", ascending=False)["customerID"]
        chosen = st.selectbox(
            "Customer", options.tolist(),
            help="Ordered by expected monthly loss.",
        )
        record = FRAME[FRAME["customerID"] == chosen].iloc[0].to_dict()
        render_result(
            chosen, float(record["churn_probability"]), record["risk_level"],
            record, load_customer_for_prediction(record), "lookup",
        )

    with scorer:
        with st.form("score_form"):
            top = st.columns([3, 2])
            with top[0]:
                new_id = st.text_input("Customer ID", placeholder="e.g. 7590-VHVEG")
            with top[1]:
                st.markdown("<div style='height:28px'></div>",
                            unsafe_allow_html=True)
                submitted = st.form_submit_button(
                    "Score customer", type="primary", width="stretch"
                )

            st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Account**")
                tenure = st.slider("Tenure (months)", 0, 72, 12)
                monthly_charges = st.number_input(
                    "Monthly charges ($)", 0.0, 200.0, 65.0, 0.5
                )
                total_charges = st.number_input(
                    "Total charges ($)", 0.0, 10000.0,
                    float(tenure * 65.0), 1.0
                )
                contract = st.selectbox(
                    "Contract type", ["Month-to-month", "One year", "Two year"]
                )
                payment_method = st.selectbox(
                    "Payment method",
                    ["Electronic check", "Mailed check",
                     "Bank transfer (automatic)", "Credit card (automatic)"],
                )
                paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            with c2:
                st.markdown("**Services**")
                internet = st.selectbox("Internet service",
                                        ["Fiber optic", "DSL", "No"])
                phone_service = st.selectbox("Phone service", ["Yes", "No"])
                multiple_lines = st.selectbox(
                    "Multiple lines", ["No", "Yes", "No phone service"]
                )
                online_security = st.selectbox(
                    "Online security", ["No", "Yes", "No internet service"]
                )
                online_backup = st.selectbox(
                    "Online backup", ["No", "Yes", "No internet service"]
                )
                device_protect = st.selectbox(
                    "Device protection", ["No", "Yes", "No internet service"]
                )
            with c3:
                st.markdown("**Household & support**")
                tech_support = st.selectbox(
                    "Tech support", ["No", "Yes", "No internet service"]
                )
                streaming_tv = st.selectbox(
                    "Streaming TV", ["No", "Yes", "No internet service"]
                )
                streaming_movies = st.selectbox(
                    "Streaming movies", ["No", "Yes", "No internet service"]
                )
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior_citizen = st.selectbox("Senior citizen",
                                              ["No (0)", "Yes (1)"])
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["No", "Yes"])

        if submitted:
            if not new_id.strip():
                st.warning("Enter a customer ID before scoring.")
            else:
                values = {
                    "tenure": tenure, "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges, "gender": gender,
                    "SeniorCitizen": senior_citizen, "Partner": partner,
                    "Dependents": dependents, "PhoneService": phone_service,
                    "MultipleLines": multiple_lines, "InternetService": internet,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protect,
                    "TechSupport": tech_support, "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies, "Contract": contract,
                    "PaperlessBilling": paperless,
                    "PaymentMethod": payment_method,
                }
                input_data = build_model_input(values)
                prediction, probability, level = predict_customer(
                    input_data, preprocessor, model
                )
                record = input_data.iloc[0].to_dict()
                record["customerID"] = new_id.strip()
                insert_customer(record, "Unknown", "New application scan")
                save_scan(new_id.strip(), probability, prediction, level)
                log_event(
                    "INFO", "Customer scored", customer_id=new_id.strip(),
                    details=f"probability={probability:.4f}; risk={level}",
                )
                st.session_state["scored"] = {
                    "id": new_id.strip(), "probability": probability,
                    "level": level, "record": record,
                }

        if "scored" in st.session_state:
            saved = st.session_state["scored"]
            render_result(
                saved["id"], saved["probability"], saved["level"],
                saved["record"],
                pd.DataFrame([{f: saved["record"][f] for f in MODEL_FEATURES}]),
                "scored",
            )


# ==========================================================================
# Retention Cases - prediction to intervention to outcome
# ==========================================================================

elif page == "Retention Cases":
    page_header(
        "Retention Cases",
        "Prediction → Intervention → Outcome. The loop that proves the product works.",
    )

    cases = case_log(VERSION)

    row = st.columns(4)
    with row[0]:
        kpi_tile("Cases opened", f"{len(cases):,}", "All time", "neutral")
    with row[1]:
        kpi_tile("In progress", f"{KPIS['open_cases']:,}",
                 "Not yet resolved", "warning")
    with row[2]:
        kpi_tile("Retained", f"{KPIS['retained']:,}",
                 f"{KPIS['win_rate']:.0%} of resolved cases", "good")
    with row[3]:
        kpi_tile("Monthly revenue saved", money(KPIS["revenue_protected"]),
                 "Billing of retained customers", "good")

    if cases.empty:
        st.info(
            "No retention cases yet. Open one from the Customer Registry or "
            "from a scored customer on Customer 360."
        )
    else:
        section("Case pipeline", "Where does the work currently sit?")
        pipeline = (
            cases.groupby("status")
            .agg(cases=("case_id", "count"),
                 value=("monthly_value", "sum"))
            .reindex(CASE_STATUSES).fillna(0).reset_index()
        )
        pipeline["cases"] = pipeline["cases"].astype(int)
        y = alt.Y("status:N", title=None, sort=CASE_STATUSES)
        base = alt.Chart(pipeline)
        bars = base.mark_bar(cornerRadiusEnd=4, size=26,
                             color=PALETTE["series1"]).encode(
            y=y,
            x=alt.X("cases:Q", title="Cases", scale=alt.Scale(domainMin=0)),
            tooltip=[
                alt.Tooltip("status:N", title="Status"),
                alt.Tooltip("cases:Q", title="Cases"),
                alt.Tooltip("value:Q", title="Monthly value", format="$,.0f"),
            ],
        )
        labels = base.mark_text(align="left", dx=6, fontSize=11, fontWeight=600,
                                color=PALETTE["text"]).encode(
            y=y, x=alt.X("cases:Q"), text=alt.Text("cases:Q"),
        )
        st.altair_chart(style_chart(bars + labels, height=220),
                        width="stretch")

        section("Case book", "Every intervention and where it ended up.")
        book = cases.copy()
        book["Accepted"] = book["customer_accepted"].map({1: "Yes", 0: "-"})
        display = pd.DataFrame({
            "Case": book["case_id"],
            "Customer": book["customerID"],
            "Risk at open": book["risk_level"],
            "Probability": book["churn_probability"],
            "Monthly value": book["monthly_value"],
            "Main driver": book["main_driver"],
            "Owner": book["assigned_to"],
            "Offer": book["offer_type"],
            "Status": book["status"],
            "Accepted": book["Accepted"],
            "Opened": book["opened_at"],
        })
        st.dataframe(
            display, width="stretch", hide_index=True,
            column_config={
                "Probability": st.column_config.ProgressColumn(
                    "Probability", format="%.0f%%", min_value=0, max_value=1
                ),
                "Monthly value": st.column_config.NumberColumn(format="$%.2f"),
            },
        )

        section("Update a case", "Record what happened.")
        edit = st.columns([2, 2, 2, 2, 2])
        with edit[0]:
            case_id = st.selectbox("Case", cases["case_id"].tolist())
        current = cases[cases["case_id"] == case_id].iloc[0]
        with edit[1]:
            new_status = st.selectbox(
                "Status", CASE_STATUSES,
                index=CASE_STATUSES.index(current["status"])
                if current["status"] in CASE_STATUSES else 0,
            )
        with edit[2]:
            new_owner = st.selectbox(
                "Owner", RETENTION_OFFICERS,
                index=RETENTION_OFFICERS.index(current["assigned_to"])
                if current["assigned_to"] in RETENTION_OFFICERS else 0,
            )
        with edit[3]:
            new_offer = st.selectbox(
                "Offer", OFFER_TYPES,
                index=OFFER_TYPES.index(current["offer_type"])
                if current["offer_type"] in OFFER_TYPES else 0,
            )
        with edit[4]:
            accepted = st.selectbox("Customer accepted", ["Not yet", "Yes", "No"])

        if st.button("Save case update", type="primary"):
            update_case(
                int(case_id),
                status=new_status,
                assigned_to=new_owner,
                offer_type=new_offer,
                contact_attempted=1 if new_status != "Open" else 0,
                offer_made=1 if new_status in ("Offer Made", "Retained", "Lost") else 0,
                customer_accepted=1 if accepted == "Yes" else 0,
            )
            st.success(f"Case #{case_id} updated.")
            st.rerun()


# ==========================================================================
# Analytics - every chart answers a stated business question
# ==========================================================================

elif page == "Analytics":
    page_header(
        "Analytics",
        "Where we lose customers, why, and whether our interventions work.",
    )

    tabs = st.tabs(["Where", "Why", "Who", "Did it work"])

    with tabs[0]:
        left, right = st.columns(2)
        with left:
            section("Churn risk by contract type",
                    "Which commercial terms leak customers?")
            chart = risk_rate_chart(FRAME, "Contract")
            if chart is not None:
                st.altair_chart(chart, width="stretch")
        with right:
            section("Churn risk by payment method",
                    "Does how they pay predict whether they stay?")
            chart = risk_rate_chart(FRAME, "PaymentMethod")
            if chart is not None:
                st.altair_chart(chart, width="stretch")

        left, right = st.columns(2)
        with left:
            section("Churn risk by internet service",
                    "Is a specific product line underperforming?")
            chart = risk_rate_chart(FRAME, "InternetService")
            if chart is not None:
                st.altair_chart(chart, width="stretch")
        with right:
            section("Churn risk by monthly charge band",
                    "At what price point do customers start leaving?")
            chart = risk_rate_chart(FRAME, "charge_band", order=CHARGE_ORDER)
            if chart is not None:
                st.altair_chart(chart, width="stretch")

        section("Churn risk across the customer lifecycle",
                "When in the relationship do we lose people?")
        chart = lifecycle_chart(FRAME)
        if chart is not None:
            st.altair_chart(chart, width="stretch")
            st.caption(
                "Risk concentrates in the first year. Retention spend aimed at "
                "early-lifecycle customers reaches the most winnable accounts."
            )

    with tabs[1]:
        section("Top churn drivers", "Why are customers leaving?")
        chart = drivers_chart(DRIVERS_FRAME, height=320, top=10)
        if chart is not None:
            st.altair_chart(chart, width="stretch")
            st.caption(
                "The trained random forest's own feature importances, summed "
                "over one-hot columns back to business attributes. This is what "
                "the model actually keys on, not a hand-written opinion."
            )

        section("Most common primary driver",
                "Which single factor flags the most at-risk customers?")
        at_risk = FRAME[FRAME["at_risk"]]
        if at_risk.empty:
            st.info("No customers currently above the action threshold.")
        else:
            driver_counts = (
                at_risk.groupby("main_driver")
                .agg(customers=("customerID", "count"),
                     expected=("expected_loss", "sum"))
                .reset_index()
                .sort_values("customers", ascending=False)
            )
            y = alt.Y("main_driver:N", title=None, sort="-x",
                      axis=alt.Axis(labelLimit=200))
            base = alt.Chart(driver_counts)
            bars = base.mark_bar(cornerRadiusEnd=4, size=20,
                                 color=PALETTE["series2"]).encode(
                y=y,
                x=alt.X("customers:Q", title="At-risk customers",
                        scale=alt.Scale(domainMin=0)),
                tooltip=[
                    alt.Tooltip("main_driver:N", title="Primary driver"),
                    alt.Tooltip("customers:Q", title="At-risk customers"),
                    alt.Tooltip("expected:Q", title="Expected monthly loss",
                                format="$,.0f"),
                ],
            )
            labels = base.mark_text(align="left", dx=6, fontSize=11,
                                    fontWeight=600,
                                    color=PALETTE["text"]).encode(
                y=y, x=alt.X("customers:Q"), text=alt.Text("customers:Q"),
            )
            st.altair_chart(style_chart(bars + labels, height=300),
                            width="stretch")
            st.caption(
                "Each at-risk customer is attributed to the highest-importance "
                "risk condition they actually meet."
            )

    with tabs[2]:
        left, right = st.columns(2)
        with left:
            section("Churn risk by customer segment",
                    "Which segments are we losing?")
            chart = risk_rate_chart(FRAME, "segment")
            if chart is not None:
                st.altair_chart(chart, width="stretch")
        with right:
            section("Expected loss by segment",
                    "Which segment costs us the most to lose?")
            chart = revenue_chart(FRAME, "segment")
            if chart is not None:
                st.altair_chart(chart, width="stretch")

        section("Expected loss across the lifecycle",
                "Where should the retention budget go?")
        chart = revenue_chart(FRAME, "tenure_band", order=TENURE_ORDER)
        if chart is not None:
            st.altair_chart(chart, width="stretch")

        section("Highest-value customers at risk",
                "Which individual accounts justify a personal call?")
        top_accounts = (
            FRAME[FRAME["at_risk"]]
            .sort_values("expected_loss", ascending=False).head(15)
        )
        if top_accounts.empty:
            st.info("No customers currently above the action threshold.")
        else:
            st.dataframe(
                pd.DataFrame({
                    "Customer": top_accounts["customerID"],
                    "Risk": top_accounts["risk_level"],
                    "Probability": top_accounts["churn_probability"],
                    "Monthly value": top_accounts["monthly_value"],
                    "Expected loss": top_accounts["expected_loss"],
                    "Main driver": top_accounts["main_driver"],
                }),
                width="stretch", hide_index=True,
                column_config={
                    "Probability": st.column_config.ProgressColumn(
                        "Probability", format="%.0f%%", min_value=0, max_value=1
                    ),
                    "Monthly value": st.column_config.NumberColumn(format="$%.2f"),
                    "Expected loss": st.column_config.NumberColumn(format="$%.2f"),
                },
            )

    with tabs[3]:
        section("Retention campaign performance",
                "Are our interventions actually saving customers?")
        cases = case_log(VERSION)
        resolved = cases[cases["status"].isin(["Retained", "Lost"])] if not cases.empty else cases

        row = st.columns(4)
        with row[0]:
            kpi_tile("Cases opened", f"{len(cases):,}", "All time", "neutral")
        with row[1]:
            kpi_tile("Resolved", f"{len(resolved):,}", "Retained or lost", "neutral")
        with row[2]:
            kpi_tile("Win rate", f"{KPIS['win_rate']:.0%}",
                     "Retained ÷ resolved", "good")
        with row[3]:
            kpi_tile("Monthly revenue saved", money(KPIS["revenue_protected"]),
                     "Billing of retained customers", "good")

        if cases.empty:
            st.info(
                "No interventions recorded yet, so campaign performance cannot "
                "be measured. Open cases from the Customer Registry to start "
                "building the outcome record."
            )
        else:
            outcome = (
                cases.groupby(["offer_type", "status"])
                .size().reset_index(name="cases")
            )
            chart = alt.Chart(outcome).mark_bar(cornerRadiusEnd=4, size=18).encode(
                y=alt.Y("offer_type:N", title=None, sort="-x",
                        axis=alt.Axis(labelLimit=220)),
                x=alt.X("cases:Q", title="Cases", stack="zero",
                        scale=alt.Scale(domainMin=0)),
                color=alt.Color(
                    "status:N", title="Status",
                    scale=alt.Scale(
                        domain=CASE_STATUSES,
                        range=["#93a4bf", PALETTE["series1"], "#fab219",
                               "#0ca30c", "#d03b3b"],
                    ),
                ),
                tooltip=["offer_type:N", "status:N", "cases:Q"],
            )
            st.altair_chart(style_chart(chart, height=280),
                            width="stretch")
            st.caption("Which offers convert, and which ones lose the customer anyway.")

        section("Model accuracy on known outcomes",
                "How much should we trust the score?")
        known = FRAME[FRAME["actual_churn"].isin(["Yes", "No"])].copy()
        if known.empty:
            st.info("No customers with a known historical outcome.")
        else:
            known["actual_binary"] = (known["actual_churn"] == "Yes").astype(int)
            known["predicted"] = (known["churn_probability"] >= CHURN_THRESHOLD).astype(int)
            accuracy = float((known["predicted"] == known["actual_binary"]).mean())
            caught = known[known["actual_binary"] == 1]
            recall = float((caught["predicted"] == 1).mean()) if len(caught) else 0.0

            row = st.columns(3)
            with row[0]:
                kpi_tile("Accuracy", f"{accuracy:.1%}",
                         f"On {len(known):,} known outcomes", "neutral")
            with row[1]:
                kpi_tile("Churners caught", f"{recall:.1%}",
                         "Recall on customers who did leave", "neutral")
            with row[2]:
                kpi_tile("Decision threshold", f"{CHURN_THRESHOLD:.0%}",
                         "Set below 50% to favour catching churners", "neutral")
            st.caption(
                "The threshold is deliberately below 50%: missing a churner "
                "costs more than an unnecessary retention call. This applies "
                "only to historical customers with a known outcome."
            )

    note = sample_note(FRAME)
    if note:
        st.caption(note)


# ==========================================================================
# System Logs - administrator only
# ==========================================================================

elif page == "System Logs":
    page_header("System Logs", "Application diagnostics and audit trail.")

    connection = get_connection()
    logs_df = pd.read_sql_query(
        """
        SELECT timestamp, level, event, customerID, details
        FROM system_logs ORDER BY log_id DESC LIMIT 500
        """,
        connection,
    )
    connection.close()

    row = st.columns(3)
    with row[0]:
        kpi_tile("Recorded events", f"{len(logs_df):,}", "Most recent 500",
                 "neutral")
    with row[1]:
        kpi_tile("Errors", f"{int((logs_df['level'] == 'ERROR').sum()):,}",
                 "Model or data failures", "critical")
    with row[2]:
        kpi_tile("Seed source", os.path.basename(SEED_FILE),
                 "Registry origin", "neutral")

    st.dataframe(logs_df, width="stretch", hide_index=True, height=460)
    st.caption(
        "Visible to the Administrator role only. Business users never see this "
        "page - it is diagnostics, not product."
    )
