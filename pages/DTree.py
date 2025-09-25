import pandas as pd
import streamlit as st
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.header("Decision Tree for Banana Quality Classification")

# โหลดไฟล์ที่อัปโหลดมา
df = pd.read_csv("./data/banana_quality.csv")
st.write("ตัวอย่างข้อมูล", df.head(10))

# แสดงชื่อคอลัมน์เพื่อความชัวร์

# กำหนด features และ target
# (สมมติว่า column ชื่อ 'Class' เป็น target)
features = [col for col in df.columns if col != 'Quality']
X = df[features]
y = df['Quality']

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=200
)

# Train model
ModelDtree = DecisionTreeClassifier()
dtree = ModelDtree.fit(x_train, y_train)

st.subheader("กรุณาป้อนข้อมูลเพื่อวิเคราะห์คุณภาพกล้วย")

# สร้าง input fields ตาม features จริงใน dataset
x_input = []
for f in features:
    val = st.number_input(f"ใส่ค่า {f}", value=0.0)
    x_input.append(val)

if st.button("วิเคราะห์ข้อมูล"):
    y_pred = dtree.predict([x_input])[0]
    st.write(f"ผลการพยากรณ์คุณภาพกล้วยคือ: **{y_pred}**")

# Accuracy
y_predict = dtree.predict(x_test)
score = accuracy_score(y_test, y_predict)
st.write(f'ความแม่นยำในการพยากรณ์: {(score * 100):.2f}%')

# Decision tree visualization
fig, ax = plt.subplots(figsize=(20, 15))
tree.plot_tree(
    dtree, feature_names=features,
    class_names=[str(c) for c in dtree.classes_],
    filled=True, fontsize=10, ax=ax
)
st.pyplot(fig)
