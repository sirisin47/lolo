from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import streamlit as st

st.title("🍌 การพยากรณ์คุณภาพกล้วยด้วย Naive Bayes")

# โหลดข้อมูล Banana
df = pd.read_csv("./data/banana_quality.csv")

# แก้ค่าที่เป็น b'xxx' → string ปกติ
df['Quality'] = df['Quality'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Features และ Target
X = df.drop('Quality', axis=1)
y = df['Quality']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างและฝึกโมเดล Naive Bayes
clf = GaussianNB()
clf.fit(X_train, y_train)

# แสดงตัวอย่างข้อมูล
st.subheader("🔍 ตัวอย่างข้อมูล")
st.write(df.head())

# Input ฟีเจอร์
st.subheader("✍️ กรอกค่าคุณสมบัติของกล้วยเพื่อพยากรณ์")
size = st.number_input('Size', value=0.0)
weight = st.number_input('Weight', value=0.0)
sweetness = st.number_input('Sweetness', value=0.0)
softness = st.number_input('Softness', value=0.0)
harvest = st.number_input('HarvestTime', value=0.0)
ripeness = st.number_input('Ripeness', value=0.0)
acidity = st.number_input('Acidity', value=0.0)

# ปุ่มทำนาย
if st.button("พยากรณ์"):
    x_input = [[size, weight, sweetness, softness, harvest, ripeness, acidity]]
    y_pred = clf.predict(x_input)
    st.write("🎯 ผลการพยากรณ์คุณภาพกล้วยคือ:", y_pred[0])

    if y_pred[0] == "Good":
        st.success("✅ กล้วยคุณภาพดี")
    else:
        st.error("❌ กล้วยคุณภาพไม่ดี")
else:
    st.info("กดปุ่มด้านบนเพื่อพยากรณ์")

# แสดง Accuracy ของโมเดล
st.subheader("📊 ความแม่นยำของโมเดล")
acc = clf.score(X_test, y_test)
st.write(f"Accuracy: {acc:.2f}")
