import os
import json
import uvicorn
import time

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from typing import List, Dict
from utils import chat_completions
import whisper
model_speech_to_text = whisper.load_model("base")
app = FastAPI()


from transformers import AutoProcessor, BarkModel
import scipy

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small")
model_text_to_speech = whisper.load_model("base")


@app.post("/speak-to-text/", tags=["speaking"])
async def speak_to_text(audio: UploadFile = File(...)):
    if not audio.filename.lower().endswith(('.mp3',)):
        raise HTTPException(status_code=400, detail="Only .mp3 files are allowed")

    file_path = os.path.join("uploads", audio.filename)
    with open(file_path, "wb") as f:
        f.write(await audio.read())
    
    result = model_speech_to_text.transcribe(file_path)
    return result["text"]
    

    
    # return result["text"]
@app.get("/text-to-speech/", tags=["speaking"])
async def text_to_speech(text: str):
    start_time = time.time()
    voice_preset = "v2/en_speaker_6"
    inputs = processor(text, voice_preset=voice_preset, return_tensors="pt")
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate
    path_file = "bark_out.wav"
    scipy.io.wavfile.write(path_file, rate=sample_rate, data=audio_array)
    end_time = time.time()
    print(end_time - start_time)
    return FileResponse(path_file)





class Grammar(BaseModel):
    user_text: str
    user_level: str
    user_history : List[Dict[str, str]] = []


@app.post("/gramar-question/", tags=["grammar"])
async def grammer_question(params: Grammar):
    user_input = params.user_text
    user_history = params.user_history
    user_level = params.user_level

    user_input = "User question: "+ user_input + "\nHistory: " + json.dumps(user_history)
    sys_prompt = f"Youre a grammar teacher and you're teaching a {user_level} student. You cant response Vietnamese or English"
    messages = [
        {
            "role": "system",
            "content": f"{sys_prompt}"
        },
        {
            "role": "user",
            "content": f"{user_input}"
        }
    ]

    response = chat_completions(messages)
    return response




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4500)