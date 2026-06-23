import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📩",
    layout="wide"
)

# ---------------- COLOR STYLING ----------------

st.markdown("""
<style>

.stApp{
background: linear-gradient(
135deg,
#090909,
#2D033B,
#810CA8
);
}

.title{
text-align:center;
font-size:45px;
font-weight:bold;
color:#00FFF5;
padding:10px;
}

.subtitle{
text-align:center;
font-size:18px;
color:#F5F5F5;
margin-bottom:30px;
}

.box{
padding:25px;
border-radius:20px;
background-color:rgba(255,255,255,0.08);
box-shadow:0px 0px 25px rgba(255,0,255,0.3);
}

.metric{
padding:15px;
border-radius:15px;
background:linear-gradient(
90deg,
#4C0033,
#790252
);

text-align:center;
font-size:18px;
color:white;
font-weight:bold;
}

.spam{
padding:15px;
border-radius:15px;
background:#FF1744;
font-size:25px;
font-weight:bold;
text-align:center;
color:white;
}

.safe{
padding:15px;
border-radius:15px;
background:#00E676;
font-size:25px;
font-weight:bold;
text-align:center;
color:black;
}

.stButton>button{

width:100%;
height:50px;

background:linear-gradient(
90deg,
#00FFF5,
#FF00E5
);

color:black;

font-size:18px;
font-weight:bold;

border-radius:15px;
border:none;
}

</style>
""", unsafe_allow_html=True)

# ---------------- DATASET ----------------

data = pd.read_csv(
    "spam.csv",
    encoding='latin-1'
)

data = data[['v1','v2']]
data.columns=['label','message']

data['label']=data['label'].map({
    'ham':0,
    'spam':1
})

# ---------------- MODEL ----------------

x=data['message']
y=data['label']

cv=CountVectorizer()

x=cv.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(
x,
y,
test_size=0.2,
random_state=42
)

model=MultinomialNB()

model.fit(
x_train,
y_train
)

pred=model.predict(
x_test
)

accuracy=accuracy_score(
y_test,
pred
)

# ---------------- HEADER ----------------

st.markdown(
"<div class='title'>📩 AI Spam Detection System</div>",
unsafe_allow_html=True
)

st.markdown(
"<div class='subtitle'>Smart Machine Learning Message Classifier</div>",
unsafe_allow_html=True
)

# ---------------- METRICS ----------------

col1,col2,col3=st.columns(3)

with col1:
    st.markdown(
    f"<div class='metric'>Total Messages<br>{len(data)}</div>",
    unsafe_allow_html=True)

with col2:
    st.markdown(
    f"<div class='metric'>Spam Messages<br>{sum(data['label']==1)}</div>",
    unsafe_allow_html=True)

with col3:
    st.markdown(
    f"<div class='metric'>Accuracy<br>{accuracy*100:.2f}%</div>",
    unsafe_allow_html=True)

st.write("")

# ---------------- INPUT ----------------

st.markdown(
"<div class='box'>",
unsafe_allow_html=True
)

message=st.text_area(
"Enter Message",
height=150,
placeholder="Type your message here..."
)

if st.button("🔍 Analyze Message"):

    if message.strip()!="":

        transformed=cv.transform(
        [message]
        )

        result=model.predict(
        transformed
        )

        if result[0]==1:

            st.markdown(
            "<div class='spam'>🚫 SPAM MESSAGE DETECTED</div>",
            unsafe_allow_html=True
            )

        else:

            st.markdown(
            "<div class='safe'>✅ SAFE MESSAGE</div>",
            unsafe_allow_html=True
            )

    else:
        st.warning(
        "Please enter a message"
        )

st.markdown(
"</div>",
unsafe_allow_html=True
)
