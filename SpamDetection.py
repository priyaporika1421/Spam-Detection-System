import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ----------------- Page Settings -----------------
st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📩",
    layout="centered"
)

# ----------------- Custom CSS -----------------
st.markdown("""
<style>

.stApp{
background: linear-gradient(to right,#141e30,#243b55);
color:white;
}

.main-title{
text-align:center;
font-size:40px;
font-weight:bold;
color:#00e6e6;
padding:10px;
}

.sub-title{
text-align:center;
font-size:18px;
color:white;
margin-bottom:30px;
}

.box{
padding:20px;
border-radius:15px;
background-color:rgba(255,255,255,0.1);
box-shadow:0px 0px 15px rgba(0,0,0,0.5);
}

.result-spam{
padding:15px;
border-radius:10px;
background-color:#ff4b4b;
color:white;
font-size:22px;
text-align:center;
font-weight:bold;
}

.result-safe{
padding:15px;
border-radius:10px;
background-color:#00cc66;
color:white;
font-size:22px;
text-align:center;
font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ----------------- Dataset -----------------
data = pd.read_csv("spam.csv")

data = data[['v1', 'v2']]
data.columns = ['label', 'message']

data['label'] = data['label'].map({
    'ham':0,
    'spam':1
})

# ----------------- Model Training -----------------
x = data['message']
y = data['label']

cv = CountVectorizer()
x = cv.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2,random_state=42
)

model = MultinomialNB()
model.fit(x_train,y_train)

# ----------------- UI -----------------
st.markdown(
    "<div class='main-title'>📩 Spam Detection System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>AI Powered Message Classification</div>",
    unsafe_allow_html=True
)

st.markdown("<div class='box'>", unsafe_allow_html=True)

message = st.text_area(
    "Enter your message:",
    height=150,
    placeholder="Type your message here..."
)

if st.button("Predict Message"):

    if message.strip() != "":

        transformed = cv.transform([message])
        prediction = model.predict(transformed)

        st.write("")

        if prediction[0] == 1:
            st.markdown(
                "<div class='result-spam'>🚫 SPAM MESSAGE DETECTED</div>",
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                "<div class='result-safe'>✅ SAFE MESSAGE</div>",
                unsafe_allow_html=True
            )

    else:
        st.warning("Please enter a message")

st.markdown("</div>", unsafe_allow_html=True)