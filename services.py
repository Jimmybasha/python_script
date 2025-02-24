import os
import io
import traceback
import uuid
from functools import lru_cache
from fastapi import FastAPI, UploadFile , Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from fastapi.responses import FileResponse
from murf import Murf
from pydub import AudioSegment
from pydub.utils import which
import whisper
from starlette.responses import StreamingResponse
from transformers import pipeline
import language_tool_python
import requests
import cmudict
import difflib
import subprocess

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_methods=["*"],
    allow_headers=["*"],
)
murf_client = Murf(api_key="ap2_576212f2-12a9-46de-bddd-7caddaa3570d")
# Set ffmpeg path
AudioSegment.converter = which("ffmpeg")
cmu_dict = cmudict.dict()


@lru_cache()
def load_models():
    """Load Whisper, Vennify T5, and LanguageTool once to optimize performance."""
    print("Loading AI Models...")
    whisper_model = whisper.load_model("small")
    grammar_corrector = pipeline(
        "text2text-generation",
        model="vennify/t5-base-grammar-correction",
        tokenizer="vennify/t5-base-grammar-correction"

    )
    grammar_tool = language_tool_python.LanguageTool('en-US')
    print("Models Loaded Successfully!")
    return whisper_model, grammar_corrector, grammar_tool

@app.post("/analyze-audio/")
async def process_audio(file: UploadFile):
    """Processes the uploaded MP3 file: transcribes, corrects grammar, and converts back to speech."""
    unique_id = str(uuid.uuid4())  # Generate a unique ID
    temp_audio_path = "temp_audio.wav"
    corrected_audio_path = f"corrected_audio_{unique_id}.mp3"

    try:
        whisper_model, grammar_corrector, grammar_tool = load_models()

        print("Step 1: Reading audio file")
        audio_bytes = await file.read()

        print("Step 2: Converting to AudioSegment")
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        except Exception as e:
            return {"error": f"Audio conversion failed: {str(e)}"}

        print("Step 3: Saving as WAV file (16kHz, mono)")
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(temp_audio_path, format="wav")

        print("Step 4: Transcribing speech with Whisper")
        result = whisper_model.transcribe(
            temp_audio_path,
            language="en",
            temperature=0,  # 🔹 Makes Whisper deterministic (less guessing)
            word_timestamps=True,
            fp16=False,
            task="transcribe",
            logprob_threshold=0.8,  # 🔹 Only accept words with high confidence
            no_speech_threshold=0.3,  # 🔹 Prevents Whisper from adding extra words
            initial_prompt="Speak concisely. No extra words.",  # 🔹 Hints Whisper to expect short phrases
            compression_ratio_threshold=2.0,  # 🔹 Stops over-prediction
            suppress_tokens=[1, 2, 3, 10, 11, 23]  # 🔹 Prevents auto-corrections
        )
        transcribed_text = result["text"].strip()
        print(f"Transcribed Text: {transcribed_text}")

        if not transcribed_text:
            return {"error": "Whisper failed to transcribe any text."}

        print("Step 5: Correcting grammar using Vennify T5")
        generation_params = {
            "max_length": 150,  # Adjust as needed
            "num_beams": 5,  # Beam search for better results
            "early_stopping": True
        }
        corrected_text = grammar_corrector(transcribed_text,**generation_params)[0]['generated_text']
        print(f"Vennify Corrected Text: {corrected_text}")

        print("Step 6: Running additional LanguageTool check")
        matches = grammar_tool.check(corrected_text)
        final_corrected_text = language_tool_python.utils.correct(corrected_text, matches)
        print(f"Final Corrected Text: {final_corrected_text}")

        print("Step 7: Converting corrected text to speech using Murf AI")
        response = murf_client.text_to_speech.generate(
            text=final_corrected_text,
            voice_id="en-US-natalie",  # Replace with desired voice ID
            format="MP3",
            sample_rate=44100.0
        )

        # Retrieve the audio file URL from the response
        audio_url = response.audio_file

        # Download the audio content
        audio_response = requests.get(audio_url)
        if audio_response.status_code == 200:
            with open(corrected_audio_path, "wb") as audio_file:
                audio_file.write(audio_response.content)
            print("Audio file downloaded successfully.")
        else:
            print(f"Failed to download audio file. Status code: {audio_response.status_code}")
            return {"error": "Failed to download audio file from Murf AI."}

        # Step 8: Cleaning up temp files
        print("Step 8: Cleaning up temp files")
        os.remove(temp_audio_path)

        print("Step 9: Returning response")
        return {
            "original_text": transcribed_text,
            "corrected_text": final_corrected_text,
            "audio_feedback_url": f"http://10.0.2.2:8000/corrected-audio/{unique_id}.mp3"

        }

    except Exception as e:
        print(f"❌ Error Occurred: {traceback.format_exc()}")
        return {"error": str(e)}


@app.get("/corrected-audio/{file_id}.mp3")
def get_corrected_audio(file_id: str):
    """Returns a dynamically stored corrected audio file."""
    corrected_audio_path = f"corrected_audio_{file_id}.mp3"  # ✅ Corrected File Naming

    if os.path.exists(corrected_audio_path):
        audio_file = open(corrected_audio_path, "rb")
        return StreamingResponse(audio_file, media_type="audio/mpeg")

    return {"error": "Audio file not found"}

def extract_phonemes(text: str) -> str:
    """Use eSpeak to extract phonemes from the given text."""
    try:
        result = subprocess.run(['espeak', '-q', '--ipa', text], capture_output=True, text=True)
        phonemes = result.stdout.strip()
        return phonemes
    except Exception as e:
        print(f"Error extracting phonemes: {e}")
        return ""
@app.post("/assess-pronunciation/")
async def assess_pronunciation(file: UploadFile, expected_text: str = Form(...)):
    try:
        audio_bytes = await file.read()
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_frame_rate(16000).set_channels(1)

        model = whisper.load_model("small")
        audio_data = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

        result = model.transcribe(
            audio_data,
            language="en",
            temperature=0,
            word_timestamps=True,
            fp16=False,
            task="transcribe",
            logprob_threshold=0.8,
            no_speech_threshold=0.3,
            initial_prompt="Speak concisely. No extra words.",
            compression_ratio_threshold=2.0,
            suppress_tokens=[1, 2, 3, 10, 11, 23]
        )
        transcribed_text = result["text"].strip()
        print(f"Transcribed Text: {transcribed_text}")

        if not transcribed_text:
            return {"error": "Whisper failed to transcribe any text."}

        expected_words = expected_text.lower().split()
        transcribed_words = transcribed_text.lower().split()

        word_level_results = []
        bad_pronunciations = []

        for i, expected_word in enumerate(expected_words):
            if i < len(transcribed_words):
                transcribed_word = transcribed_words[i]
                expected_phonemes = []
                transcribed_phonemes = []

                try:
                    pronunciations = cmu_dict[expected_word]
                    if pronunciations:
                        expected_phonemes = pronunciations[0]
                    else:
                        print(f"No pronunciations found for '{expected_word}' in CMU dictionary.")
                        word_level_results.append({"expected_word": expected_word, "transcribed_word": transcribed_word, "similarity": 0})
                        bad_pronunciations.append({"expected_word": expected_word, "transcribed_word": transcribed_word, "similarity": 0})
                        continue

                except KeyError:
                    print(f"Word '{expected_word}' not found in CMU dictionary.")
                    word_level_results.append({"expected_word": expected_word, "transcribed_word": transcribed_word, "similarity": 0})
                    bad_pronunciations.append({"expected_word": expected_word, "transcribed_word": transcribed_word, "similarity": 0})
                    continue

                try:
                    pronunciations = cmu_dict[transcribed_word]
                    if pronunciations:
                        transcribed_phonemes = pronunciations[0]
                    else:
                        print(f"No pronunciations found for '{transcribed_word}' in CMU dictionary.")
                        word_level_results.append({"expected_word": expected_word, "transcribed_word": transcribed_word, "similarity": 0})
                        bad_pronunciations.append({"expected_word": expected_word, "transcribed_word": transcribed_word, "similarity": 0})
                        continue

                except KeyError:
                    print(f"Word '{transcribed_word}' not found in CMU dictionary.")
                    word_level_results.append({"expected_word": expected_word, "transcribed_word": transcribed_word, "similarity": 0})
                    bad_pronunciations.append({"expected_word": expected_word, "transcribed_word": transcribed_word, "similarity": 0})
                    continue

                similarity = difflib.SequenceMatcher(None, expected_phonemes, transcribed_phonemes).ratio() * 100

                word_level_results.append({"expected_word": expected_word, "transcribed_word": transcribed_word, "similarity": similarity})

                if similarity < 70:
                    bad_pronunciations.append({"expected_word": expected_word, "transcribed_word": transcribed_word, "similarity": similarity})
            else:
                word_level_results.append({
                    "expected_word": expected_word,
                    "transcribed_word": "",
                    "similarity": 0
                })

        overall_similarity = np.mean([
            result["similarity"]
            for result in word_level_results
            if result["similarity"] > 0 and result["transcribed_word"] != "" and result["expected_word"] in cmu_dict and result["transcribed_word"] in cmu_dict
        ]) if word_level_results else 0
        feedback = f"Overall pronunciation accuracy: {overall_similarity:.2f}%\n"

        if bad_pronunciations:
            feedback += "Words with pronunciation issues:\n"
            for bad_word in bad_pronunciations:
                feedback += f"- Expected: {bad_word['expected_word']}, Transcribed: {bad_word['transcribed_word']}, Similarity: {bad_word['similarity']:.2f}%\n"
        else:
            feedback += "No pronunciation issues detected."

        return {
            "expected_text": expected_text,
            "transcribed_text": transcribed_text,
            "word_level_results": word_level_results,
            "bad_pronunciations": bad_pronunciations,
            "feedback": feedback
        }

    except Exception as e:
        print(f"Error Occurred: {str(e)}")
        return {"error": str(e)}