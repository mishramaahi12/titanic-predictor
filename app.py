import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE ----------------
st.set_page_config(
    page_title="AI Titanic Oracle",
    page_icon="🚢",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: Arial;
}
section.main > div {
    max-width: 100% !important;
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
[data-testid="stAppViewContainer"]{
    background: linear-gradient(135deg,#0f172a,#1d4ed8,#0f172a);
    color:white;
}
[data-testid="stHeader"]{
    background: rgba(0,0,0,0);
}
.card{
    background: rgba(255,255,255,0.08);
    padding:20px;
    border-radius:20px;
    border:1px solid rgba(255,255,255,0.15);
    box-shadow:0 8px 20px rgba(0,0,0,0.25);
}
.title{
    text-align:center;
    font-size:48px;
    font-weight:800;
    color:white;
}
.sub{
    text-align:center;
    color:#dbeafe;
    margin-bottom:20px;
    font-size:18px;
}
.stButton > button{
    width:100%;
    height:3.2em;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

df = load_data()

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encode
le_sex = LabelEncoder()
le_emb = LabelEncoder()

df["Sex"] = le_sex.fit_transform(df["Sex"])
df["Embarked"] = le_emb.fit_transform(df["Embarked"])

# Features
X = df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
y = df["Survived"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🚢 AI Titanic Oracle</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Premium Full Width Titanic Survival Predictor</div>", unsafe_allow_html=True)

# ---------------- MAIN LAYOUT ----------------
left, right = st.columns([1,1], gap="large")

# ---------- LEFT SIDE ----------
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧍 Passenger Details")

    pclass = st.selectbox("Passenger Class", [1,2,3])
    sex = st.selectbox("Gender", ["Male","Female"])
    age = st.slider("Age", 0, 100, 25)
    sibsp = st.slider("Siblings / Spouse", 0, 8, 0)
    parch = st.slider("Parents / Children", 0, 6, 0)
    fare = st.slider("Fare", 0.0, 600.0, 30.0)
    embarked = st.selectbox("Embarked Port", ["C","Q","S"])

    predict = st.button("🔮 Predict Now")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RIGHT SIDE ----------
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Live Dataset Stats")

    col1, col2 = st.columns(2)
    col1.metric("Passengers", len(df))
    col2.metric("Survival Rate", f"{df['Survived'].mean():.1%}")

    col3, col4 = st.columns(2)
    col3.metric("Avg Fare", f"{df['Fare'].mean():.2f}")
    col4.metric("Model", "Random Forest")

    st.markdown("</div>", unsafe_allow_html=True)

    # Chart
    chart_df = df.copy()
    chart_df["Sex"] = chart_df["Sex"].map({1:"Male",0:"Female"})
    fig = px.histogram(
        chart_df,
        x="Sex",
        color=chart_df["Survived"].map({0:"No",1:"Yes"}),
        barmode="group",
        title="Survival by Gender"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- PREDICTION ----------------
if predict:
    sex_val = 1 if sex == "Male" else 0
    emb_map = {"C":0,"Q":1,"S":2}

    user_data = pd.DataFrame([[
        pclass, sex_val, age, sibsp, parch, fare, emb_map[embarked]
    ]], columns=X.columns)

    prob = model.predict_proba(user_data)[0][1]
    pred = model.predict(user_data)[0]

    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    a,b,c = st.columns(3)
    a.metric("Survival Chance", f"{prob:.1%}")
    b.metric("Risk Chance", f"{1-prob:.1%}")
    c.metric("Verdict", "SURVIVE" if pred==1 else "DANGER")

    st.progress(float(prob))

    if pred == 1:
        st.success("✅ Passenger is likely to survive!")
        st.balloons()
    else:
        st.error("❌ Passenger has lower survival chances.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<center>Made with Streamlit + Machine Learning 🚀</center>", unsafe_allow_html=True)
