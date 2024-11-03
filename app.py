# app.py
import streamlit as st
from typing import Optional
import tempfile
import os
from dotenv import load_dotenv
import openai
from loguru import logger
import sounddevice as sd
import soundfile as sf
import numpy as np
import queue
import time
from pathlib import Path

# Load environment variables
load_dotenv()

# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Application settings
MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
DEFAULT_MODEL = MODELS[0]
DEFAULT_POSITION = "Python Developer"

# Audio settings
SAMPLE_RATE = 44100
CHANNELS = 1

# Prompt templates
SYS_PREFIX = "You are interviewing for a "
SYS_SUFFIX = """ position.\nYou will receive an audio transcription of the question. It may not be complete. You need to understand the question and write an answer to it.\n"""
SHORT_INSTRUCTION = "Concisely respond, limiting your answer to 50 words."
LONG_INSTRUCTION = "Before answering, take a deep breath and think one step at a time. Believe the answer in no more than 150 words."

# Set up logging
logger.add("app_log_{time}.log", rotation="1 day")

class AudioRecorder:
    def __init__(self):
        self.audio_data = []
        self.recording = False
        self.audio_queue = queue.Queue()
        self.device_info = sd.query_devices(None, 'input')
        self.sample_rate = int(self.device_info['default_samplerate'])

    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        if self.recording:
            self.audio_queue.put(indata.copy())

    def start_recording(self):
        self.recording = True
        self.audio_data = []
        self.stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=self.sample_rate,
            callback=self.callback
        )
        self.stream.start()

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.stream.stop()
            self.stream.close()
            
            # Get all remaining audio data from queue
            while not self.audio_queue.empty():
                self.audio_data.append(self.audio_queue.get())

            if self.audio_data:
                # Combine all audio chunks
                audio_data = np.concatenate(self.audio_data)
                
                # Create a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                sf.write(temp_file.name, audio_data, self.sample_rate)
                
                return temp_file.name
        return None

def transcribe_audio(audio_file_path) -> str:
    logger.debug("Transcribing audio using Whisper API...")
    try:
        with open(audio_file_path, 'rb') as audio_file:
            response = openai.Audio.transcribe("whisper-1", audio_file)
            transcript = response['text']
            logger.debug("Audio transcription completed.")
            return transcript
    except Exception as error:
        logger.error(f"Transcription error: {error}")
        raise error
    finally:
        # Clean up temporary file
        try:
            os.unlink(audio_file_path)
        except:
            pass

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
    if 'recorder' not in st.session_state:
        st.session_state.recorder = AudioRecorder()
    if 'recording' not in st.session_state:
        st.session_state.recording = False

    # Custom CSS
    st.markdown("""
        <style>
        .stButton button {
            width: 100%;
        }
        .recording {
            color: red;
            animation: blink 1s infinite;
        }
        @keyframes blink {
            50% { opacity: 0.5; }
        }
        </style>
    """, unsafe_allow_html=True)

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
            st.write("Click the button below to start/stop recording:")
            
            # Record button
            if st.session_state.recording:
                button_label = "⏹️ Stop Recording"
                button_type = "secondary"
            else:
                button_label = "🎙️ Start Recording"
                button_type = "primary"

            if st.button(button_label, key="record_button", type=button_type):
                if not st.session_state.recording:
                    # Start recording
                    st.session_state.recorder.start_recording()
                    st.session_state.recording = True
                    st.rerun()
                else:
                    # Stop recording
                    audio_file = st.session_state.recorder.stop_recording()
                    st.session_state.recording = False
                    if audio_file:
                        try:
                            with st.spinner('Transcribing recorded audio...'):
                                transcript = transcribe_audio(audio_file)
                                st.session_state.transcript = transcript
                                st.success("Recording transcribed successfully!")
                        except Exception as e:
                            st.error(f"Error transcribing recording: {str(e)}")
                    st.rerun()
            
            # Show recording status
            if st.session_state.recording:
                st.markdown("### 🔴 Recording in progress...")
                st.warning("Click 'Stop Recording' to finish recording")
                
                # Add a progress indicator
                progress_placeholder = st.empty()
                progress_bar = progress_placeholder.progress(0)
                for i in range(100):
                    if not st.session_state.recording:
                        break
                    progress_bar.progress((i + 1) % 100)
                    time.sleep(0.1)

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
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                                temp_file.write(uploaded_file.read())
                                temp_file_path = temp_file.name
                            
                            transcript = transcribe_audio(temp_file_path)
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