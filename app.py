
import os
import warnings
import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DARK GRADIENT UI + CIRCULAR METERS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp{
    background:
      radial-gradient(circle at 10% 5%,rgba(56,189,248,.12),transparent 28%),
      radial-gradient(circle at 90% 10%,rgba(124,58,237,.15),transparent 30%),
      linear-gradient(135deg,#07111f 0%,#0b1730 50%,#111b3d 100%);
    color:#f8fafc;
}
.block-container{max-width:1400px;padding-top:2rem}
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#08101f,#0d1730 55%,#111b3d);
    border-right:1px solid rgba(148,163,184,.16);
}
[data-testid="stSidebar"] *{color:#f8fafc!important}
h1,h2,h3,h4,p,label{color:#f8fafc!important}
.muted{color:#94a3b8!important}
.eyebrow{color:#67e8f9;font-size:.78rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase}
.glass{
    background:rgba(15,23,42,.72);border:1px solid rgba(148,163,184,.14);
    border-radius:18px;padding:22px;box-shadow:0 14px 40px rgba(0,0,0,.22);
}
.section-title{
    font-size:1.05rem;font-weight:800;color:#f8fafc;
    border-left:4px solid #38bdf8;padding-left:10px;margin:24px 0 12px;
}
.result-churn,.result-retain{
    color:white;padding:22px;border-radius:18px;text-align:center;
    font-size:1.35rem;font-weight:800;margin:12px 0 22px;
}
.result-churn{background:linear-gradient(135deg,#ef4444,#7f1d1d)}
.result-retain{background:linear-gradient(135deg,#10b981,#064e3b)}
.meter-card{
    background:rgba(15,23,42,.72);border:1px solid rgba(148,163,184,.14);
    border-radius:18px;padding:18px 12px 16px;min-height:260px;
    display:flex;flex-direction:column;align-items:center;justify-content:center;
}
.circle{
    --v:50%;--c:#38bdf8;width:170px;height:170px;border-radius:50%;
    background:conic-gradient(var(--c) 0 var(--v),rgba(148,163,184,.14) var(--v) 100%);
    display:grid;place-items:center;position:relative;
}
.circle:before{
    content:"";width:134px;height:134px;border-radius:50%;
    position:absolute;background:#0c1730;border:1px solid rgba(148,163,184,.13);
}
.circle-inner{position:relative;z-index:2;text-align:center}
.circle-value{font-size:2rem;font-weight:900;color:#f8fafc}
.circle-label{font-size:.72rem;color:#94a3b8}
.meter-title{margin-top:14px;color:#f8fafc;font-size:.95rem;font-weight:800;text-align:center}
.meter-subtitle{color:#94a3b8;font-size:.75rem;text-align:center;margin-top:4px}
.risk-chip{display:inline-block;padding:7px 14px;border-radius:999px;font-weight:800;font-size:.82rem;margin-top:8px}
.high{background:rgba(239,68,68,.16);color:#fca5a5}
.medium{background:rgba(245,158,11,.16);color:#fcd34d}
.low{background:rgba(16,185,129,.16);color:#6ee7b7}
.stButton>button{
    background:linear-gradient(135deg,#38bdf8,#2563eb)!important;color:white!important;
    border:0;border-radius:11px;min-height:52px;font-weight:850;width:100%;
}
button[data-baseweb="tab"]{color:#94a3b8}
button[data-baseweb="tab"][aria-selected="true"]{color:#67e8f9}
.footer{text-align:center;color:#64748b;font-size:.78rem;padding:20px 0}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.abspath(__file__))
    preprocessor = joblib.load(os.path.join(base, "models", "preprocessor.joblib"))
    model = joblib.load(os.path.join(base, "models", "best_random_forest_tuned.pkl"))
    return preprocessor, model


def meter(value, title, subtitle, color):
    value = max(0.0, min(float(value), 100.0))
    return f"""
    <div class="meter-card">
      <div class="circle" style="--v:{value}%;--c:{color}">
        <div class="circle-inner">
          <div class="circle-value">{value:.1f}%</div>
          <div class="circle-label">of 100%</div>
        </div>
      </div>
      <div class="meter-title">{title}</div>
      <div class="meter-subtitle">{subtitle}</div>
    </div>
    """


def risk_band(prob):
    if prob >= 60:
        return "High", "high", "#ef4444"
    if prob >= 35:
        return "Medium", "medium", "#f59e0b"
    return "Low", "low", "#22c55e"


try:
    preprocessor, model = load_artifacts()
except Exception as exc:
    st.error("Model artifacts could not be loaded.")
    st.code(str(exc))
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOMER INPUT
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Customer Profile")
    st.caption("Complete the profile, then scan the customer.")

    st.markdown("**Account Information**")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, .5)
    total_charges = st.number_input(
        "Total Charges ($)", 0.0, 10000.0,
        float(tenure * monthly_charges), 1.0
    )
    contract = st.selectbox("Contract Type", ["Month-to-month","One year","Two year"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check","Mailed check",
         "Bank transfer (automatic)","Credit card (automatic)"]
    )
    paperless = st.selectbox("Paperless Billing", ["Yes","No"])

    st.markdown("**Demographics**")
    gender = st.selectbox("Gender", ["Male","Female"])
    senior = st.selectbox("Senior Citizen", ["No (0)","Yes (1)"])
    senior_val = "1" if senior.startswith("Yes") else "0"
    partner = st.selectbox("Partner", ["Yes","No"])
    dependents = st.selectbox("Dependents", ["No","Yes"])

    st.markdown("**Services Subscribed**")
    phone = st.selectbox("Phone Service", ["Yes","No"])
    lines = st.selectbox("Multiple Lines", ["No","Yes","No phone service"])
    internet = st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
    no_net = "No internet service"
    security = st.selectbox("Online Security", ["No","Yes",no_net])
    backup = st.selectbox("Online Backup", ["No","Yes",no_net])
    device = st.selectbox("Device Protection", ["No","Yes",no_net])
    support = st.selectbox("Tech Support", ["No","Yes",no_net])
    tv = st.selectbox("Streaming TV", ["No","Yes",no_net])
    movies = st.selectbox("Streaming Movies", ["No","Yes",no_net])

    st.markdown("---")
    scan = st.button("🔍  SCAN CUSTOMER", use_container_width=True)


st.markdown('<div class="eyebrow">GROUP 2 • AI CUSTOMER RETENTION</div>', unsafe_allow_html=True)
st.title("📡 Telecom Churn Predictor")
st.markdown(
    '<p class="muted">Scan a customer profile to estimate churn risk and support retention prioritisation.</p>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["🎯 Prediction","📊 Model Performance","ℹ️ How It Works"])


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    if scan:
        input_data = pd.DataFrame([{
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "gender": gender,
            "SeniorCitizen": senior_val,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone,
            "MultipleLines": lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
        }])

        with st.spinner("Scanning customer profile..."):
            X = preprocessor.transform(input_data)
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]

        churn = float(probabilities[1] * 100)
        retain = float(probabilities[0] * 100)
        risk, risk_css, risk_color = risk_band(churn)

        if prediction == 1:
            st.markdown(
                f'<div class="result-churn">⚠️ HIGH CHURN RISK'
                f'<div style="font-size:.9rem;font-weight:500;margin-top:6px;">'
                f'Model classification: Churn • Probability: {churn:.1f}%</div></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-retain">✅ LOW CHURN RISK'
                f'<div style="font-size:.9rem;font-weight:500;margin-top:6px;">'
                f'Model classification: No Churn • Retention probability: {retain:.1f}%</div></div>',
                unsafe_allow_html=True
            )

        a, b, c = st.columns(3)
        with a:
            st.markdown(meter(churn,"Churn Probability","Model-estimated probability","#ef4444"),
                        unsafe_allow_html=True)
        with b:
            st.markdown(meter(retain,"Retention Probability","Model-estimated probability","#10b981"),
                        unsafe_allow_html=True)
        with c:
            st.markdown(meter(churn,"Risk Score",f"{risk} risk band",risk_color),
                        unsafe_allow_html=True)
            st.markdown(
                f'<div style="text-align:center"><span class="risk-chip {risk_css}">{risk} Risk</span></div>',
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="glass"><b>Decision note</b><br>'
            '<span class="muted">The model classification uses the saved model threshold. '
            'The Low/Medium/High bands are a prototype presentation layer and are not '
            'being claimed as a separately validated production threshold.</span></div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-title">💡 Suggested Retention Actions</div>',
                    unsafe_allow_html=True)

        if prediction == 1:
            tips = []
            if contract == "Month-to-month":
                tips.append("Consider a discounted One-Year or Two-Year contract.")
            if internet == "Fiber optic" and security == "No":
                tips.append("Consider an Online Security retention bundle.")
            if support == "No":
                tips.append("Consider a Tech Support trial.")
            if payment_method == "Electronic check":
                tips.append("Consider encouraging an automatic payment method.")
            if monthly_charges > 70:
                tips.append("Review whether a loyalty/value offer is appropriate.")
            if tenure < 12:
                tips.append("Prioritise early-life-cycle engagement.")
            if not tips:
                tips.append("Assign a retention specialist to understand the customer's concerns.")
            for tip in tips:
                st.markdown(f"- {tip}")
        else:
            st.info("Current classification is No Churn. Continue standard engagement and monitor changes.")

        st.markdown('<div class="section-title">📋 Submitted Customer Profile</div>',
                    unsafe_allow_html=True)
        st.dataframe(input_data.T.rename(columns={0:"Value"}), use_container_width=True)

    else:
        st.markdown(
            '<div class="glass"><div class="eyebrow">READY TO SCAN</div>'
            '<h2>Find the customers most likely to leave.</h2>'
            '<p class="muted">Complete the profile in the sidebar and click '
            '<b style="color:#67e8f9">SCAN CUSTOMER</b>.</p></div>',
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Model Evaluation")
    results = pd.DataFrame({
        "Model":["Baseline (Dummy)","Logistic Regression","Random Forest",
                 "Gradient Boosting","Random Forest (Tuned) ✅"],
        "Accuracy":[.7346,.7381,.7438,.8027,.7559],
        "Precision":[0,.5043,.5114,.6633,.5272],
        "Recall":[0,.7834,.7807,.5214,.7781],
        "F1 Score":[0,.6136,.6180,.5838,.6285],
        "ROC-AUC":[.5000,.8415,.8411,.8448,.8415]
    })
    st.dataframe(
        results.style.format({
            "Accuracy":"{:.2%}","Precision":"{:.2%}","Recall":"{:.2%}",
            "F1 Score":"{:.4f}","ROC-AUC":"{:.4f}"
        }),
        use_container_width=True, hide_index=True
    )

    st.markdown('<div class="section-title">Recommended Model — Tuned Random Forest</div>',
                unsafe_allow_html=True)

    cols = st.columns(4)
    metrics = [
        (75.59,"Accuracy","Overall correctness","#38bdf8"),
        (52.72,"Precision","Flag reliability","#f59e0b"),
        (77.81,"Recall","Churners caught","#10b981"),
        (84.15,"ROC-AUC","Discrimination","#a78bfa")
    ]
    for col, item in zip(cols, metrics):
        with col:
            st.markdown(meter(*item), unsafe_allow_html=True)

    st.markdown(
        '<div class="glass" style="margin-top:20px"><b>Evaluation takeaway</b><br>'
        '<span class="muted">Gradient Boosting had higher accuracy, but substantially lower Recall. '
        'The Tuned Random Forest was therefore recommended under the project’s F1-first approach.</span></div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### How the Prototype Works")
    st.markdown(
        '<div class="glass"><div class="eyebrow">END-TO-END FLOW</div>'
        '<h3>Customer profile → preprocessing → trained model → risk signal</h3>'
        '<p class="muted">The application uses the saved preprocessing object and '
        'Tuned Random Forest produced by the project pipeline.</p></div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="section-title">1. Enter the customer profile</div>', unsafe_allow_html=True)
    st.write("Enter account, demographic and service information in the sidebar.")
    st.markdown('<div class="section-title">2. Run the scan</div>', unsafe_allow_html=True)
    st.write("The app transforms the inputs using the saved preprocessing object.")
    st.markdown('<div class="section-title">3. Receive the prediction</div>', unsafe_allow_html=True)
    st.write("The model returns churn probability and its classification.")
    st.markdown('<div class="section-title">4. Use the result as decision support</div>', unsafe_allow_html=True)
    st.write("The output supports — but does not replace — human retention judgement.")
    st.markdown(
        '<div class="glass"><b>Project limitation</b><br>'
        '<span class="muted">This is a prototype using the IBM Telco Customer Churn classroom dataset. '
        'It is not a production model trained on real Nigerian telecom operator data.</span></div>',
        unsafe_allow_html=True
    )

st.markdown('<div class="footer">Telecom Churn Predictor • Tuned Random Forest • Group 2 AI & ML Project</div>',
            unsafe_allow_html=True)
