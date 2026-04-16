import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Titanic Oracle", page_icon="🚢", layout="centered")

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
    background: linear-gradient(135deg,#0f172a,#1d4ed8,#0f172a);
}
.block-container{
    max-width: 800px;
    padding-top: 1rem;
}
.card{
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 20px;
    border:1px solid rgba(255,255,255,0.15);
}
.title{
    text-align:center;
    font-size:40px;
    font-weight:800;
    color:white;
}
.sub{
    text-align:center;
    color:#dbeafe;
    margin-bottom:20px;
}
.stButton>button{
    width:100%;
    border-radius:14px;
    height:3em;
    font-size:18px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

df = load_data()

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

X = df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=200,max_depth=6,random_state=42)
model.fit(X_train,y_train)

st.markdown("<div class='title'>🚢 AI Titanic Oracle</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Perfect Full Width Mobile/Desktop Layout</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)

pclass = st.selectbox("Passenger Class",[1,2,3])
sex = st.selectbox("Gender",["Male","Female"])
age = st.slider("Age",0,100,25)
sibsp = st.slider("Siblings/Spouse",0,8,0)
parch = st.slider("Parents/Children",0,6,0)
fare = st.slider("Fare",0.0,600.0,30.0)
embarked = st.selectbox("Embarked",["C","Q","S"])

predict = st.button("🔮 Predict Survival")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📊 Dataset Stats")
st.write(f"Total Passengers: {len(df)}")
st.write(f"Survival Rate: {df['Survived'].mean():.1%}")
st.write(f"Average Fare: {df['Fare'].mean():.2f}")
st.write("Model: Random Forest")
st.markdown("</div>", unsafe_allow_html=True)

if predict:
    sex_val = 1 if sex=="Male" else 0
    emb_map = {"C":0,"Q":1,"S":2}

    row = pd.DataFrame([[pclass,sex_val,age,sibsp,parch,fare,emb_map[embarked]]], columns=X.columns)
    row = scaler.transform(row)

    prob = model.predict_proba(row)[0][1]
    pred = 1 if prob >= 0.5 else 0

    st.markdown("---")
    st.metric("Survival Chance", f"{prob:.1%}")
    st.progress(float(prob))

    if pred == 1:
        st.success("✅ Likely to survive!")
        st.balloons()
    else:
        st.error("❌ Low survival chances!")

st.markdown("<hr><center>Made with Streamlit + ML</center>", unsafe_allow_html=True)