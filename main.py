from ollama import chat
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
 
model = joblib.load('Mental_Health_Model.pkl')
top_countries = ['Other','India','USA','Canada','Australia','UK','Germany','Mexico','Turkey','France']
 
app = FastAPI()
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
#A first Pydantic Model
class StudentData(BaseModel):
    age                     : int = Field(..., ge=10, le=100)
    gender                  : Literal['Male', 'Female']
    country                 : str
    academic_level          : Literal['Undergraduate', 'Graduate', 'High School']
    most_used_platform      : Literal['Facebook', 'LinkedIn', 'Instagram', 'Snapchat','Twitter','YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp','WeChat']
    purpose_of_use          : Literal['Networking', 'Education', 'Entertainment', 'News']
    avg_daily_usage_hours   : float = Field(..., ge=0, le=24)
    daily_unlocks           : int   = Field(..., ge=0)
    study_hours             : float = Field(..., ge=0, le=24)
    physical_activity_hours : float = Field(..., ge=0, le=24)
    sleep_hours_per_night   : float = Field(..., ge=0, le=24)
    stress_level            : Literal['Medium', 'Low', 'Very High', 'High']
 
 
 
 
# Describe what we send back
class PredictionResponse(BaseModel):
    predicted_mental_health_score: float
    #6.777777 -> float
    recommendations: list[str]
 
 
def generate_recommendations(data: StudentData, score: float) -> list[str]:
    """
    Rule-based recommendations. Combines the overall predicted score band
    with specific red flags in the individual inputs, so two people with
    the same score can get different, more relevant tips.
    """
    tips: list[str] = []
 
    # --- Overall score band -------------------------------------------
    if score < 4:
        tips.append(
            "Your overall signal is on the strained side — consider talking to a "
            "counselor, mentor, or someone you trust about how you've been feeling."
        )
    elif score < 7:
        tips.append(
            "Your overall signal looks fairly balanced, but there's room to build "
            "in more recovery time — small consistent changes tend to help most."
        )
    else:
        tips.append(
            "Your overall signal looks strong — keep up the habits that are working for you."
        )
 
    # --- Sleep -----------------------------------------------------------
    if data.sleep_hours_per_night < 6:
        tips.append(
            f"You're averaging {data.sleep_hours_per_night:.1f}h of sleep — try to work "
            "toward 7-8 hours a night; poor sleep is one of the strongest drivers of low mood and stress."
        )
    elif data.sleep_hours_per_night > 9.5:
        tips.append(
            "You're sleeping quite a lot — if you still feel tired afterward, it may be "
            "worth checking your sleep quality rather than just quantity."
        )
 
    # --- Screen time / phone unlocks --------------------------------------
    if data.avg_daily_usage_hours > 6:
        tips.append(
            f"Your screen time ({data.avg_daily_usage_hours:.1f}h/day) is on the higher side — "
            "try setting app time limits or scheduling screen-free blocks, especially before bed."
        )
    if data.daily_unlocks > 100:
        tips.append(
            f"You're unlocking your phone about {data.daily_unlocks} times a day — frequent "
            "checking often tracks with anxiety; try turning off non-essential notifications."
        )
 
    # --- Stress level --------------------------------------------------
    if data.stress_level in ("High", "Very High"):
        tips.append(
            "Your stress level is high — short daily practices like breathing exercises, "
            "journaling, or a walk between tasks can help lower baseline stress."
        )
 
    # --- Physical activity -----------------------------------------------
    if data.physical_activity_hours < 1:
        tips.append(
            "You're getting under an hour of physical activity a day — even a 20-30 minute "
            "walk most days is linked with meaningfully better mood and stress regulation."
        )
 
    # --- Study load --------------------------------------------------------
    if data.study_hours > 10:
        tips.append(
            f"You're studying about {data.study_hours:.1f}h/day — long unbroken study blocks "
            "increase burnout risk; try the Pomodoro technique (25-30 min focus + 5 min break)."
        )
 
    # --- Fallback if everything looks fine ---------------------------------
    if len(tips) == 1:  # only the score-band tip got added
        tips.append(
            "No major red flags in your habits — maintaining this routine should keep your signal healthy."
        )
 
    return tips
 
 
def generate_llm_recommendations(data: StudentData, score: float) -> list[str]:

    prompt = f"""
You are a supportive student wellness assistant.

The student's machine learning predicted mental health score is {score}.

Student information:

Age: {data.age}
Gender: {data.gender}
Country: {data.country}
Academic Level: {data.academic_level}
Most Used Platform: {data.most_used_platform}
Purpose of Use: {data.purpose_of_use}
Average Daily Social Media Usage: {data.avg_daily_usage_hours} hours
Daily Phone Unlocks: {data.daily_unlocks}
Study Hours: {data.study_hours} hours/day
Physical Activity: {data.physical_activity_hours} hours/day
Sleep: {data.sleep_hours_per_night} hours/night
Stress Level: {data.stress_level}

Based on this information, generate 4 practical and personalized recommendations.

Focus on:
- sleep
- study habits
- stress management
- physical activity
- healthy technology usage

Rules:
- Be supportive and non-judgmental.
- Do not diagnose any mental health condition.
- Do not claim the prediction is a medical diagnosis.
- Give practical and personalized advice.
- Generate exactly 4 recommendations.
- Each recommendation must be ONE short sentence.
- Maximum 15 words per recommendation.
- Do not provide explanations, examples, reasons, or extra details.
- Do not use headings or sub-points.
- Return ONLY a numbered list.
"""

    response = chat(
        model="mistral",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    text = response["message"]["content"]
    print("========== MISTRAL RESPONSE ==========")
    print(text)
    print("======================================")

    recommendations = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    return recommendations
 



@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/style.css")
def css():
    return FileResponse("style.css", media_type="text/css")


@app.get("/script.js")
def javascript():
    return FileResponse("script.js", media_type="application/javascript")
 
 
@app.post('/predict', response_model=PredictionResponse) #6.77777
def predict(data: StudentData):
   
   country_group = data.country if data.country in top_countries else "Other"
 
   input_row = pd.DataFrame([{
        'Age'                       :data.age,
        'Gender'                    :data.gender,
        'Country'                   :data.country,
        'Academic_Level'            :data.academic_level,
        'Most_Used_Platform'        :data.most_used_platform,
        'Purpose_Of_Use'            :data.purpose_of_use,
        'Avg_Daily_Usage_Hours'     :data.avg_daily_usage_hours,
        'Daily_Unlocks'             :data.daily_unlocks,
        'Study_Hours'               :data.study_hours,
        'Physical_Activity_Hours'   :data.physical_activity_hours,
        'Sleep_Hours_Per_Night'     :data.sleep_hours_per_night,
        'Stress_Level'              :data.stress_level,
        'Grouped_country'           :country_group
   }])
 
   prediction = model.predict(input_row)[0] #6.77
   score = round(float(prediction), 2)
   recommendations = generate_llm_recommendations(data, score)
 
   return PredictionResponse(
        predicted_mental_health_score=score,
        recommendations=recommendations
   )