import streamlit as st
import pickle
import re

st.set_page_config(page_title="AI Fake News Detector", page_icon="🕵️‍♂️", layout="centered")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #1f4068 0%, #162447 100%); color: #ffffff; }
    @keyframes float { 0% { transform: translateY(0px); } 50% { transform: translateY(-10px); } 100% { transform: translateY(0px); } }
    .animated-title { text-align: center; font-family: 'Poppins', sans-serif; background: linear-gradient(45deg, #00fff0, #0088ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem; font-weight: 800; animation: float 4s ease-in-out infinite; margin-bottom: 5px; }
    .glass-card { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border-radius: 15px; padding: 25px; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); margin-bottom: 20px; }
    .stButton>button { background: linear-gradient(45deg, #0088ff, #00fff0) !important; color: #111625 !important; font-weight: bold !important; font-size: 18px !important; border-radius: 10px !important; border: none !important; box-shadow: 0 4px 15px rgba(0, 255, 240, 0.4) !important; transition: all 0.3s ease !important; }
    .stButton>button:hover { transform: scale(1.02) !important; box-shadow: 0 6px 20px rgba(0, 255, 240, 0.6) !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_components():
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    return model, vectorizer

try:
    model, vectorizer = load_model_components()
except FileNotFoundError:
    st.error("🚨 Model files missing!")

# Updated Matching Text Cleaning
def clean_text_v2(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\n', ' ', text)
    return text

with st.sidebar:
    st.markdown("<div style='text-align: center;'><img src='https://flaticon.com' width='90' style='animation: float 3s ease-in-out infinite;'></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #00fff0;'>🕵️‍♂️ Control Panel</h2>", unsafe_allow_html=True)
    st.info("**Developer:** Jason Ezra.A\n\n**Roll No:** 26AI036\n\n**Department:** B.Tech AI & DS")
    st.markdown("---")
    st.success("System: **Online**")

st.markdown('<h1 class="animated-title">AI Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a2a8d3; font-size: 16px;'>Advanced Machine Learning Framework for News Verification</p>", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
user_input = st.text_area("🔮 Paste the news article text below for deep analysis:", height=150, placeholder="Enter your English news content here...")
st.markdown('</div>', unsafe_allow_html=True)

if st.button("✨ Verify News Article", use_container_width=True):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze!")
    else:
        with st.spinner("🧠 AI Neural Network is analyzing..."):
            cleaned_input = clean_text_v2(user_input)
            vectorized_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(vectorized_input)[0] # Extract absolute prediction value
            
            st.markdown("### 📊 Final Classification Report:")
            
            if int(prediction) == 1:
                st.balloons()
                st.success("### ✅ VERDICT: REAL NEWS")
                st.metric(label="Authenticity Status", value="Verified Real", delta="Safe to share")
            else:
                st.error("### 🚨 VERDICT: FAKE NEWS DETECTED")
                st.metric(label="Authenticity Status", value="Fabricated / False", delta="- High Risk", delta_color="inverse")
