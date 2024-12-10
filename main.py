from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware

# Define app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Load the pre-trained model
model = load_model("model_fish_classifier.h5")

# Define fish names based on model output
class_mapping = {
    0: 'Fin_FinbackWhale',
    1: 'NorthernRightWhale',
    2: 'AtlanticSpottedDolphin',
    3: 'BottlenoseDolphin',
    4: 'Fraser',
    5: 'sDolphin',
    6: 'KillerWhale',
    7: 'SpinnerDolphin',
    8: 'White_sided.d',
    9: 'WeddellSeal',
    10: 'RossSeal'
}

@app.get("/")
def home():
    return {"message": "Fish Sound Classifier API is running. Use the /predict/ endpoint to classify fish species."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Predict the fish species from the uploaded sound file.
    """
    try:
        # Validate file type
        if file.content_type not in ["audio/wav", "audio/mpeg"]:
            raise HTTPException(status_code=400, detail="Invalid audio format. Use WAV or MP3.")

        # Save the uploaded file
        audio_path = f"temp_audio.{file.filename.split('.')[-1]}"
        with open(audio_path, "wb") as audio_file:
            audio_file.write(await file.read())

        # Load and preprocess the audio file
        y, sr = librosa.load(audio_path, sr=None)

        # If the audio is too short, pad it with zeros to a minimum length
        if len(y) < sr * 2:  # Ensure at least 2 seconds of audio
            y = np.pad(y, (0, sr * 2 - len(y)), mode='constant')

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # Ensure enough frames for reshaping (take first 16 frames)
        if mfccs.shape[1] < 16:
            # Pad MFCCs to ensure we have at least 16 frames
            mfccs = np.pad(mfccs, ((0, 0), (0, 16 - mfccs.shape[1])), mode='constant')

        # Reshape to match model input (1, 640)
        mfccs = mfccs.T[:16].flatten()  # Take the first 16 frames and flatten to 640 features
        mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension

        # Predict using the model
        predictions = model.predict(mfccs)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))

        # Map predicted class to fish names
        fish_name = class_mapping.get(predicted_class, "Unknown")

        return JSONResponse({
            "predicted_fish": fish_name,
            "confidence": confidence
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
