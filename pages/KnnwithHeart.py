from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title('การทำนายคุณภาพกล้วยด้วยเทคนิค K-Nearest Neighbor')

# โหลดข้อมูล
df = pd.read_csv("./data/banana_quality.csv")

# แปลงค่า target จาก b'xxx' → string ปกติ
df['Quality'] = df['Quality'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# แสดงข้อมูล
st.subheader("ข้อมูลส่วนแรก 10 แถว")
st.write(df.head(10))
st.subheader("ข้อมูลส่วนสุดท้าย 10 แถว")
st.write(df.tail(10))

# สถิติพื้นฐาน
st.subheader("📈 สถิติพื้นฐานของข้อมูล")
st.write(df.describe())

# ฟีเจอร์และ target
features = ['Size', 'Weight', 'Sweetness', 'Softness', 
            'HarvestTime', 'Ripeness', 'Acidity']
target = 'Quality'

# Visualization
st.subheader("📌 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล")
feature = st.selectbox("เลือกฟีเจอร์", features)

# Boxplot
st.write(f"### 🎯 Boxplot: {feature} แยกตามคุณภาพกล้วย")
fig, ax = plt.subplots()
sns.boxplot(data=df, x=target, y=feature, ax=ax)
st.pyplot(fig)

# Pairplot
if st.checkbox("แสดง Pairplot (ใช้เวลาประมวลผลเล็กน้อย)"):
    fig2 = sns.pairplot(df, hue=target)
    st.pyplot(fig2.fig)

# ส่วนทำนาย
st.subheader("🔮 ทำนายคุณภาพกล้วย")

# Input fields อัตโนมัติ
user_input = []
for f in features:
    val = st.number_input(f"กรุณาใส่ค่า {f}", value=0.0)
    user_input.append(val)

if st.button("ทำนายผล"):
    X = df[features]
    y = df[target]

    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)

    x_input = np.array([user_input])
    out = Knn_model.predict(x_input)

    st.write("ผลการทำนายคุณภาพกล้วยคือ:", out[0])

    if out[0] == "Good":
        st.success("🍌 กล้วยคุณภาพดี")
    else:
        st.error("🍌 กล้วยคุณภาพไม่ดี")
else:
    st.write("ไม่ทำนาย")
