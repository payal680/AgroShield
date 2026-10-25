from PIL import Image 
import pymysql

from fastapi.responses import FileResponse
from io import BytesIO 
from fastapi import FastAPI,File, UploadFile,HTTPException,Form
app = FastAPI()
import numpy as np 
import tensorflow as tf
from pathlib import Path
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
print("Key prefix:", os.getenv("GROQ_API_KEY", "")[:5])
load_dotenv(override=True)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
def get_db_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="Payal@2005",
        database="agroshield",
        cursorclass=pymysql.cursors.DictCursor
    )

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "plant_disease_output" / "plant_disease_cnn.keras"

DATASET_DIR = Path(
    r"C:\Users\HP\Downloads\PlantVillage-Dataset-master\raw\color"
)

class_names = sorted(
    folder.name
    for folder in DATASET_DIR.iterdir()
    if folder.is_dir()
)

model = tf.keras.models.load_model(MODEL_PATH)
print("model loaded successfully!")


@app.get("/")
def home():
    return FileResponse("templates/index.html")



@app.post("/predict")
async def predict(file: UploadFile = File(...), username: str = Form("Guest")):
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        image = image.resize((128, 128))

        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        print("Output shape:", predictions.shape)
        print("Top predictions:")
        for i in np.argsort(predictions[0])[-5:][::-1]:
            print(class_names[i], float(predictions[0][i]))

        disease = class_names[predicted_index]
        plant_name, detected_disease = disease.split("___", 1)

        confidence = round(
            float(predictions[0][predicted_index]) * 100, 2
        )
        print("Received username:", username)

        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO diagnostic_history
                    (user_name, plant_name, detected_disease, confidence_score)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (username, plant_name, detected_disease, confidence)
                )
            conn.commit()
        finally:
            conn.close()

        return {
            "predicted_disease": disease,
            "confidence": confidence,
            "symptoms": (
                "Symptoms information for "
                + disease.replace("___", " ").replace("_", " ")
                + " will be added."
            )
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics")
def get_metrics():
    conn = get_db_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT
                    COUNT(*) AS total_scans,
                    ROUND(AVG(confidence_score), 2) AS avg_confidence
                FROM diagnostic_history
            """)
            result = cursor.fetchone()

        return {
            "total_scans": result["total_scans"],
            "avg_confidence": float(result["avg_confidence"] or 0)
        }

    finally:
        conn.close()


@app.post("/recommendations")
async def get_recommendations(
    plant_name: str = Form(...),
    disease_name: str = Form(...)
):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a plant disease guidance assistant. "
                        "Give 3 short, practical recommendations. "
                        "Do not invent pesticide dosages or guarantee cures. "
                        "Advise consulting a local agricultural expert "
                        "before using chemical treatments."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Plant: {plant_name}\n"
                        f"Predicted disease: {disease_name}\n"
                        "Suggest safe next steps."
                    )
                }
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content

        recommendations = [
            line.strip().lstrip("•-0123456789. ")
            for line in answer.splitlines()
            if line.strip()
        ]

        return {
            "plant_name": plant_name,
            "disease_name": disease_name,
            "recommendations": recommendations,
            "source": "Groq AI"
        }

    except Exception as e:
        print("Groq API error:", repr(e))
    raise HTTPException(
        status_code=503,
        detail="AI recommendations are temporarily unavailable."
    )