import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("Online Course Completion Prediction")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("online_course_engagement_data.csv")
    return df

data = load_data()

st.subheader("Dataset Preview")
st.write(data.head())

# Encode Categorical Data
encoder = LabelEncoder()
data["CourseCategory"] = encoder.fit_transform(data["CourseCategory"])

# Features and Target
X = data.drop(["UserID", "CourseCompletion"], axis=1)
y = data["CourseCompletion"]

# Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

st.subheader("Enter User Details")

category = st.number_input("Course Category (encoded value)", 0)
time_spent = st.number_input("Time Spent On Course")
videos = st.number_input("Videos Watched")
quizzes = st.number_input("Quizzes Taken")
score = st.number_input("Quiz Score")
completion_rate = st.number_input("Completion Rate")
device = st.number_input("Device Type")

if st.button("Predict Completion"):

    input_data = np.array([[category,time_spent,videos,quizzes,score,completion_rate,device]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("User will COMPLETE the course")
    else:
        st.error("User will NOT complete the course")


