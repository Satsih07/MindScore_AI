```python
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


# ============================================================
# Load ML Model
# ============================================================

model = joblib.load("Mental_Health_Model.pkl")

top_countries = [
    "Other",
    "India",
    "USA",
    "Canada",
    "Australia",
    "UK",
    "Germany",
    "Mexico",
    "Turkey",
    "France"
]


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI()


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Input Pydantic Model
# ============================================================

class StudentData(BaseModel):

    age: int = Field(..., ge=10, le=100)

    gender: Literal[
        "Male",
        "Female"
    ]

    country: str

    academic_level: Literal[
        "Undergraduate",
        "Graduate",
        "High School"
    ]

    most_used_platform: Literal[
        "Facebook",
        "LinkedIn",
        "Instagram",
        "Snapchat",
        "Twitter",
        "YouTube",
        "TikTok",
        "LINE",
        "KakaoTalk",
        "VKontakte",
        "WhatsApp",
        "WeChat"
    ]

    purpose_of_use: Literal[
        "Networking",
        "Education",
        "Entertainment",
        "News"
    ]

    avg_daily_usage_hours: float = Field(
        ...,
        ge=0,
        le=24
    )

    daily_unlocks: int = Field(
        ...,
        ge=0
    )

    study_hours: float = Field(
        ...,
        ge=0,
        le=24
    )

    physical_activity_hours: float = Field(
        ...,
        ge=0,
        le=24
    )

    sleep_hours_per_night: float = Field(
        ...,
        ge=0,
        le=24
    )

    stress_level: Literal[
        "Medium",
        "Low",
        "Very High",
        "High"
    ]


# ============================================================
# Prediction Response Model
# ============================================================

class PredictionResponse(BaseModel):

    predicted_mental_health_score: float

    recommendations: list[str]


# ============================================================
# Rule-Based Recommendations
# ============================================================

def generate_recommendations(
    data: StudentData,
    score: float
) -> list[str]:

    tips: list[str] = []

    # --------------------------------------------------------
    # Overall Score
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Sleep
    # --------------------------------------------------------

    if data.sleep_hours_per_night < 6:

        tips.append(
            f"You're averaging {data.sleep_hours_per_night:.1f}h of sleep — "
            "try to work toward 7-8 hours a night."
        )

    elif data.sleep_hours_per_night > 9.5:

        tips.append(
            "You're sleeping quite a lot — if you still feel tired afterward, "
            "it may be worth checking your sleep quality rather than just quantity."
        )


    # --------------------------------------------------------
    # Screen Time
    # --------------------------------------------------------

    if data.avg_daily_usage_hours > 6:

        tips.append(
            f"Your screen time ({data.avg_daily_usage_hours:.1f}h/day) is on the "
            "higher side — try setting app time limits or scheduling screen-free blocks."
        )


    # --------------------------------------------------------
    # Phone Unlocks
    # --------------------------------------------------------

    if data.daily_unlocks > 100:

        tips.append(
            f"You're unlocking your phone about {data.daily_unlocks} times a day — "
            "try turning off non-essential notifications."
        )


    # --------------------------------------------------------
    # Stress
    # --------------------------------------------------------

    if data.stress_level in ("High", "Very High"):

        tips.append(
            "Your stress level is high — short daily practices like breathing exercises, "
            "journaling, or a walk between tasks can help manage stress."
        )


    # --------------------------------------------------------
    # Physical Activity
    # --------------------------------------------------------

    if data.physical_activity_hours < 1:

        tips.append(
            "You're getting under an hour of physical activity a day — "
            "even a 20-30 minute walk most days can support mood and stress management."
        )


    # --------------------------------------------------------
    # Study Hours
    # --------------------------------------------------------

    if data.study_hours > 10:

        tips.append(
            f"You're studying about {data.study_hours:.1f}h/day — "
            "try taking regular breaks to reduce burnout risk."
        )


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if len(tips) == 1:

        tips.append(
            "No major red flags in your habits — maintaining this routine "
            "can help support your overall wellbeing."
        )


    return tips


# ============================================================
# Home Page
# ============================================================

@app.get("/")
def home():

    return FileResponse("index.html")


# ============================================================
# CSS
# ============================================================

@app.get("/style.css")
def css():

    return FileResponse(
        "style.css",
        media_type="text/css"
    )


# ============================================================
# JavaScript
# ============================================================

@app.get("/script.js")
def javascript():

    return FileResponse(
        "script.js",
        media_type="application/javascript"
    )


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(data: StudentData):

    # --------------------------------------------------------
    # Group Country
    # --------------------------------------------------------

    country_group = (
        data.country
        if data.country in top_countries
        else "Other"
    )


    # --------------------------------------------------------
    # Create Input DataFrame
    # --------------------------------------------------------

    input_row = pd.DataFrame([
        {
            "Age": data.age,

            "Gender": data.gender,

            "Country": data.country,

            "Academic_Level": data.academic_level,

            "Most_Used_Platform": data.most_used_platform,

            "Purpose_Of_Use": data.purpose_of_use,

            "Avg_Daily_Usage_Hours": data.avg_daily_usage_hours,

            "Daily_Unlocks": data.daily_unlocks,

            "Study_Hours": data.study_hours,

            "Physical_Activity_Hours": data.physical_activity_hours,

            "Sleep_Hours_Per_Night": data.sleep_hours_per_night,

            "Stress_Level": data.stress_level,

            "Grouped_country": country_group
        }
    ])


    # --------------------------------------------------------
    # ML Prediction
    # --------------------------------------------------------

    prediction = model.predict(input_row)[0]

    score = round(
        float(prediction),
        2
    )


    # --------------------------------------------------------
    # Generate Recommendations
    #
    # IMPORTANT:
    # We use the local rule-based recommendation system
    # instead of Ollama because Render does not have a
    # local Ollama server running.
    # --------------------------------------------------------

    recommendations = generate_recommendations(
        data,
        score
    )


    # --------------------------------------------------------
    # Return Response
    # --------------------------------------------------------

    return PredictionResponse(

        predicted_mental_health_score=score,

        recommendations=recommendations
    )

