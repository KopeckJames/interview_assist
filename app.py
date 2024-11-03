# app.py
import streamlit as st
from typing import Optional
import tempfile
import os
from dotenv import load_dotenv
import openai
from loguru import logger
import io

# Load environment variables
load_dotenv()

# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Application settings
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

def transcribe_audio(audio_file) -> str:
    """Transcribe audio using OpenAI Whisper API"""
    try:
        response = openai.Audio.transcribe("whisper-1", audio_file)
        return response["text"]
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise e

def generate_answer(
    transcript: str,
    short_answer: bool,
    model: str = DEFAULT_MODEL,
    position: str = DEFAULT_POSITION,
    job_posting: str = "",
    resume: str = ""
) -> str:
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

def main():
    st.set_page_config(page_title="Interview Assistant", layout="wide")
    
    # Initialize session state
    if 'transcript' not in st.session_state:
        st.session_state.transcript = ""
    if 'short_answer' not in st.session_state:
        st.session_state.short_answer = ""
    if 'long_answer' not in st.session_state:
        st.session_state.long_answer = ""

    # Title
    st.title("Interview Assistant")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Settings")
        
        model = st.selectbox("Select Model", MODELS, index=MODELS.index(DEFAULT_MODEL), key="model_select")
        position = st.text_input("Position", value=DEFAULT_POSITION, key="position_input")
        job_posting = st.text_area("Job Posting (Optional)", height=100, key="job_posting_input")
        resume = st.text_area("Resume (Optional)", height=100, key="resume_input")

        st.subheader("Audio Input")
        tabs = st.tabs(["Record Audio", "Upload Audio"])

        with tabs[0]:
            st.write("Click to record your voice:")
            # Using Streamlit's built-in audio recorder
            audio_bytes = st.audio_recorder(pause_threshold=3.0)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                
                try:
                    # Create a bytes buffer
                    audio_file = io.BytesIO(audio_bytes)
                    # Add .wav filename suffix
                    audio_file.name = "recorded_audio.wav"
                    
                    with st.spinner('Transcribing audio...'):
                        transcript = transcribe_audio(audio_file)
                        st.session_state.transcript = transcript
                        st.success("Transcription completed!")
                except Exception as e:
                    st.error(f"Error transcribing audio: {str(e)}")
                    logger.error(f"Error during transcription: {str(e)}")

        with tabs[1]:
            uploaded_file = st.file_uploader(
                "Upload Audio File",
                type=['wav', 'mp3', 'm4a'],
                key="file_uploader",
                help="Upload an audio file to transcribe"
            )

            if uploaded_file is not None:
                if st.button("Transcribe Uploaded File", key="transcribe_button"):
                    try:
                        with st.spinner('Transcribing uploaded audio...'):
                            transcript = transcribe_audio(uploaded_file)
                            st.session_state.transcript = transcript
                            st.success("File transcribed successfully!")
                    except Exception as e:
                        st.error(f"Error transcribing file: {str(e)}")

        if st.button("Generate Answers", key="generate_button"):
            if not st.session_state.transcript:
                st.warning("Please record or upload audio first")
            else:
                try:
                    with st.spinner("Generating answers..."):
                        st.session_state.short_answer = generate_answer(
                            st.session_state.transcript,
                            short_answer=True,
                            model=model,
                            position=position,
                            job_posting=job_posting,
                            resume=resume
                        )
                        st.session_state.long_answer = generate_answer(
                            st.session_state.transcript,
                            short_answer=False,
                            model=model,
                            position=position,
                            job_posting=job_posting,
                            resume=resume
                        )
                    st.success("Answers generated successfully!")
                except Exception as e:
                    st.error(f"Error generating answers: {str(e)}")

    with col2:
        st.subheader("Transcription")
        st.text_area(
            label="Transcription Output",
            value=st.session_state.transcript,
            height=100,
            disabled=True,
            key="transcript_display",
            label_visibility="collapsed"
        )

        st.subheader("Generated Answers")
        answer_col1, answer_col2 = st.columns(2)
        
        with answer_col1:
            st.markdown("**Short Answer**")
            st.text_area(
                label="Short Answer Output",
                value=st.session_state.short_answer,
                height=200,
                disabled=True,
                key="short_answer_display",
                label_visibility="collapsed"
            )
            
        with answer_col2:
            st.markdown("**Long Answer**")
            st.text_area(
                label="Long Answer Output",
                value=st.session_state.long_answer,
                height=200,
                disabled=True,
                key="long_answer_display",
                label_visibility="collapsed"
            )

if __name__ == "__main__":
    main()
