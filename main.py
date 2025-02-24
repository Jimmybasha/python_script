from fastapi import UploadFile, File,Form

from Api import generate_api
from services import process_audio, get_corrected_audio, assess_pronunciation

app = generate_api()


@app.post("/analyze-audio/")
async def analyze_audio(file: UploadFile = File(...)):
    """Receives an MP3 file, converts it to text, corrects grammar, and returns corrected text and audio."""
    return await process_audio(file)

@app.get("/corrected-audio/{file_id}.mp3")
async def corrected_audio(file_id: str):
    """Returns the corrected audio file."""
    return get_corrected_audio(file_id)
@app.post("/assess-pronunciation/")
async def assess_pronunciation_endpoint(
    file: UploadFile = File(...),
    expected_text: str = Form(...)
):
    """
    Receives an audio file and expected text, assesses pronunciation, and returns feedback.
    """
    return await assess_pronunciation(file, expected_text)

@app.get("/")
async def test():
    return {"message": "FastAPI is working!"}