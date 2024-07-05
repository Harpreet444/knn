import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

st.set_page_config(page_title="KNN",page_icon='🪸',layout='centered')

data_set = load_digits()
x_train, x_test, y_train, y_test = train_test_split(data_set.data,data_set.target,test_size=0.2,random_state=10)
model = joblib.load("D:\\Machine_learning\\K_nearest_neabure\\model.job")

st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #ff8c1a'>KNN classifier</h1>", unsafe_allow_html=True)

st.write(''' 
Exercise: From sklearn.datasets load digits dataset and do following

1. Classify digits (0 to 9) using KNN classifier. You can use different values for k neighbors and need to figure out a value of K that gives you a maximum score. You can manually try different values of K or use gridsearchcv
2. Plot confusion matrix
3. Plot classification report
''')

st.markdown("<h3 style='text-align: center; color: #ff8c1a'>Best value of K</h1>", unsafe_allow_html=True)
st.write('By using RandomizedSearchCV hyper-parameter tuning technique i findout that model give best score for K=1')
st.code('Accuracy = '+str(model.score(x_test,y_test)))

st.markdown("<h3 style='text-align: center; color: #ff8c1a'>Confusion Matrix</h1>", unsafe_allow_html=True)

cn = confusion_matrix(y_test,model.predict(x_test))

fig, ax = plt.subplots()
sns.heatmap(cn,annot=True,cmap='Oranges')
ax.set_title("Confusion matrix")
ax.set_ylabel("Actual values")
ax.set_xlabel("Predicted values")
st.pyplot(fig)

st.markdown("<h3 style='text-align: center; color: #ff8c1a'>Classification Report</h1>", unsafe_allow_html=True)

st.text(classification_report(y_test,model.predict(x_test)))


# theam attributes:
# [theme]
# base="light"
# primaryColor="#ff8c1a"
# backgroundColor="#f5dbc2"
# secondaryBackgroundColor="#ffefe2"
