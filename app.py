# app.py
import streamlit as st
from typing import Optional
import tempfile
import os
from dotenv import load_dotenv
import openai
from loguru import logger
import base64

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

def get_audio_recorder_html():
    return """
        <style>
            .button-container { margin: 10px 0; }
            .record-button { 
                background-color: #ff4b4b;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                transition: background-color 0.3s;
            }
            .record-button:hover { background-color: #eb2020; }
            .recording { 
                animation: pulse 1.5s infinite;
                background-color: #eb2020;
            }
            .audio-level {
                width: 100%;
                height: 20px;
                background-color: #f0f2f6;
                margin-top: 10px;
                border-radius: 4px;
                overflow: hidden;
            }
            .level-bar {
                height: 100%;
                width: 0%;
                background-color: #ff4b4b;
                transition: width 0.1s;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
        </style>
        <div class="button-container">
            <button id="recordButton" class="record-button" onclick="toggleRecording()">
                🎙️ Start Recording
            </button>
            <div class="audio-level">
                <div id="levelBar" class="level-bar"></div>
            </div>
        </div>
        <script>
            let mediaRecorder;
            let audioChunks = [];
            let isRecording = false;
            let audioContext;
            let analyser;
            let dataArray;
            let animationId;

            async function setupAudio() {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                
                // Set up audio analysis
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                const source = audioContext.createMediaStreamSource(stream);
                source.connect(analyser);
                analyser.fftSize = 256;
                dataArray = new Uint8Array(analyser.frequencyBinCount);
                
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const reader = new FileReader();
                    reader.readAsDataURL(audioBlob);
                    reader.onloadend = () => {
                        const base64Audio = reader.result.split(',')[1];
                        window.parent.postMessage({
                            type: 'streamlit:setComponentValue',
                            data: base64Audio
                        }, '*');
                    };
                };
            }

            function updateAudioLevel() {
                if (!isRecording) {
                    cancelAnimationFrame(animationId);
                    return;
                }
                
                analyser.getByteFrequencyData(dataArray);
                const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
                const level = (average / 255) * 100;
                document.getElementById('levelBar').style.width = level + '%';
                
                animationId = requestAnimationFrame(updateAudioLevel);
            }

            async function toggleRecording() {
                const button = document.getElementById('recordButton');
                
                if (!isRecording) {
                    try {
                        if (!mediaRecorder) {
                            await setupAudio();
                        }
                        audioChunks = [];
                        mediaRecorder.start();
                        isRecording = true;
                        button.textContent = '⏹️ Stop Recording';
                        button.classList.add('recording');
                        updateAudioLevel();
                    } catch (error) {
                        console.error('Error:', error);
                    }
                } else {
                    mediaRecorder.stop();
                    isRecording = false;
                    button.textContent = '🎙️ Start Recording';
                    button.classList.remove('recording');
                    cancelAnimationFrame(animationId);
                    document.getElementById('levelBar').style.width = '0%';
                }
            }
        </script>
    """

def transcribe_audio(audio_bytes) -> str:
    logger.debug("Transcribing audio using Whisper API...")
    try:
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()
            
            # Open the file in binary read mode for the API
            with open(temp_file.name, 'rb') as audio_file:
                response = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio_file
                )
            
            # Clean up the temporary file
            os.unlink(temp_file.name)
            
            transcript = response['text']
            logger.debug("Audio transcription completed.")
            return transcript
    except Exception as error:
        logger.error(f"Transcription error: {error}")
        raise error

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
            st.write("Click the button below to start/stop recording:")
            
            # Audio recorder component
            st.components.v1.html(get_audio_recorder_html(), height=100)
            
            # Handle recorded audio
            if 'audio_data' in st.session_state:
                try:
                    with st.spinner('Transcribing recorded audio...'):
                        audio_bytes = base64.b64decode(st.session_state.audio_data)
                        transcript = transcribe_audio(audio_bytes)
                        st.session_state.transcript = transcript
                        st.success("Recording transcribed successfully!")
                        # Clear the audio data
                        del st.session_state.audio_data
                except Exception as e:
                    st.error(f"Error transcribing recording: {str(e)}")

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
                            audio_bytes = uploaded_file.read()
                            transcript = transcribe_audio(audio_bytes)
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
