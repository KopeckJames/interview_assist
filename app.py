# app.py
from flask import Flask, render_template, request, jsonify
from typing import Optional
import os
from dotenv import load_dotenv
import openai  # Import openai directly
from loguru import logger
import whisper  # Import Whisper for local audio transcription

# Load environment variables
load_dotenv()

# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Flask app configuration
app = Flask(__name__)

# Application settings
APPLICATION_WIDTH = 85
OUTPUT_FILE_NAME = "record.wav"
MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
DEFAULT_MODEL = MODELS[0]
DEFAULT_POSITION = "Python Developer"

# Prompt templates
SYS_PREFIX = "You are interviewing for a "
SYS_SUFFIX = """ position.\nYou will receive an audio transcription of the question. It may not be complete. You need to understand the question and write an answer to it.\n"""
SHORT_INSTRUCTION = "Concisely respond, limiting your answer to 50 words."
LONG_INSTRUCTION = "Before answering, take a deep breath and think one step at a time. Believe the answer in no more than 150 words."

# Set up logging
logger.add("app_log_{time}.log", rotation="1 day")

# Load the Whisper model for local transcription
whisper_model = whisper.load_model("base")  # Adjust model size as needed

# Audio transcription function
def transcribe_audio(path_to_file: str) -> str:
    logger.debug(f"Transcribing audio from: {path_to_file}...")
    try:
        result = whisper_model.transcribe(path_to_file)
        transcript = result['text']
        logger.debug("Audio transcription completed.")
        return transcript
    except Exception as error:
        logger.error(f"Transcription error: {error}")
        raise error

# Answer generation function
def generate_answer(
    transcript: str,
    short_answer: bool,
    model: str = DEFAULT_MODEL,
    position: str = DEFAULT_POSITION,
    job_posting: str = "",
    resume: str = ""
) -> str:
    # Set the prompt with either short or long instruction
    system_prompt = SYS_PREFIX + position + SYS_SUFFIX
    if job_posting:
        system_prompt += f"\n\nJob Posting:\n{job_posting}\n\n"
    if resume:
        system_prompt += f"\n\nResume:\n{resume}\n\n"
    system_prompt += SHORT_INSTRUCTION if short_answer else LONG_INSTRUCTION

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
            temperature=0.7,
        )
        answer = response['choices'][0]['message']['content']
        return answer
    except Exception as error:
        logger.error(f"Answer generation error: {error}")
        raise error

# Route for main interface
@app.route("/")
def home():
    return render_template("index.html")

# Endpoint for audio transcription
@app.route("/transcribe", methods=["POST"])
def transcribe_audio_endpoint():
    try:
        file = request.files["audio_file"]
        path = os.path.join("uploads", OUTPUT_FILE_NAME)
        os.makedirs("uploads", exist_ok=True)
        file.save(path)
        transcript = transcribe_audio(path)
        return jsonify({"transcript": transcript}), 200
    except Exception as error:
        logger.error(f"Error in transcription endpoint: {error}")
        return jsonify({"error": "Transcription failed."}), 500

# Endpoint for generating both short and long answers
@app.route("/generate_answer", methods=["POST"])
def generate_answer_endpoint():
    data = request.json
    transcript = data.get("transcript", "")
    job_posting = data.get("job_posting", "")
    resume = data.get("resume", "")
    position = data.get("position", DEFAULT_POSITION)
    model = data.get("model", DEFAULT_MODEL)

    try:
        # Generate both short and long answers
        short_answer = generate_answer(transcript, short_answer=True, model=model, position=position, job_posting=job_posting, resume=resume)
        long_answer = generate_answer(transcript, short_answer=False, model=model, position=position, job_posting=job_posting, resume=resume)
        
        return jsonify({
            "short_answer": short_answer,
            "long_answer": long_answer
        }), 200
    except Exception as error:
        logger.error(f"Error in answer generation endpoint: {error}")
        return jsonify({"error": "Answer generation failed."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
