import pandas as pd
import streamlit as st
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.header("Decision Tree for classification (Banana Quality)")

# อ่านไฟล์ CSV ที่คุณอัปโหลด
df = pd.read_csv("./data/banana_quality (1).csv")
st.write("ตัวอย่างข้อมูล", df.head(10))

# ตรวจสอบชื่อคอลัมน์ก่อน
st.write("Columns:", df.columns.tolist())

# กำหนด features และ target
# แก้ตรงนี้ให้ตรงกับคอลัมน์ของ banana_quality
features = [col for col in df.columns if col != 'Class']  # สมมติว่าชื่อ target คือ Class
X = df[features]
y = df['Class']

# แบ่งข้อมูล train/test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=200)

# สร้างโมเดล Decision Tree
ModelDtree = DecisionTreeClassifier()
dtree = ModelDtree.fit(x_train, y_train)

st.subheader("กรุณาป้อนข้อมูลเพื่อพยากรณ์")

# สร้างช่องรับค่าตาม features
x_input = []
for f in features:
    val = st.number_input(f'ใส่ค่า {f}', value=0.0)
    x_input.append(val)

if st.button("พยากรณ์"):
    y_predict2 = dtree.predict([x_input])
    st.write("ผลการพยากรณ์:", y_predict2[0])

# ทดสอบความแม่นยำ
y_predict = dtree.predict(x_test)
score = accuracy_score(y_test, y_predict)
st.write(f'ความแม่นยำในการพยากรณ์: {(score*100):.2f} %')

# แสดง Tree Visualization
fig, ax = plt.subplots(figsize=(12, 8))
tree.plot_tree(dtree, feature_names=features, class_names=dtree.classes_, filled=True, ax=ax)
st.pyplot(fig)
