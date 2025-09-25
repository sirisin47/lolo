import streamlit as st

st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
            font-family: 'Arial', sans-serif;
        }
        h1 {
            text-align: center;
            color: #2E86C1;
        }
        .card {
            background-color: white;
            padding: 20px;
            margin: 15px 0;
            border-radius: 15px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# ===== Title + Description =====
st.title("🔐 ระบบวิเคราะห์การโจมตีแบบ Phishing")
st.write("เลือกเทคนิค Machine Learning ที่คุณต้องการทดสอบด้านล่าง 👇")

# ===== Card + ลิงก์ =====
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.page_link("pages/KnnwithHeart.py", 
                 label="การวิเคราะห์การโจมตี Phishing ด้วยเทคนิค *KNN*", 
                 icon="🧮")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.page_link("pages/NaiveBaye.py", 
                 label="การวิเคราะห์การโจมตี Phishing ด้วยเทคนิค *Naive Bayes*", 
                 icon="📊")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.page_link("pages/DTree.py", 
                 label="การวิเคราะห์การโจมตี Phishing ด้วยเทคนิค *Decision Tree*", 
                 icon="🌳")
    st.markdown('</div>', unsafe_allow_html=True)