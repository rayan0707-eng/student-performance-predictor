import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split

#Streamlit Configurations
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("Student Performance Predictor")
st.write("This web app uses a simple machine learning model to predict a student's average score based on input subject marks.")

#Input Sliders
maths = st.slider('Math score',0,100,50)
science = st.slider('Science score',0,100,50)
english = st.slider('English score',0,100,50)

training_data = pd.DataFrame({
    'maths':np.random.randint(30,100,100),
    'science':np.random.randint(30,100,100),
    'english':np.random.randint(30,100,100)
})

# Average of each student
training_data['average'] = training_data.mean(axis = 1)

X = training_data[['maths','science','english']]
y = training_data['average']

#Split Data
train_X, test_X, train_y, test_y = train_test_split(X,y, test_size=0.2)

#Train Model
model = LinearRegression()
model.fit(X,y)

input_data=[[maths,science,english]]
prediction = model.predict(input_data)[0]

st.success(f"Predicted score is : {round(prediction,2)}")

# Visulization

# Plotting training data and predictions

st.subheader("Visual Comparison")

fig,ax = plt.subplots()

ax.scatter(training_data['maths'],training_data['average'],color = 'blue', label = 'Training Data')
ax.scatter(maths, prediction, color = 'red',label = 'Input', s= 100)
ax.set_xlabel("Maths")
ax.set_ylabel('Predicted Average')
ax.set_title('Math vs Average Prediction')
ax.legend()

# Using Streamlit
st.pyplot(fig)