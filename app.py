

import streamlit as st
import os
import tavily
from PIL import Image
from io import BytesIO
from phi.agent import Agent
from phi.model.google import Gemini
from transformers import pipeline  # For Whisper integration
from tempfile import NamedTemporaryFile
from phi.tools.tavily import TavilyTools
from constants import SYSTEM_PROMPT, INSTRUCTIONS
#from tavily import TavilyTools  # Add this line

# Configuration: Replace with your actual API keys
os.environ['GOOGLE_API_KEY'] = st.secrets.get('GEMINI_KEY')
os.environ['TAVILY_API_KEY'] = st.secrets.get('TAVILY_KEY')
MAX_IMAGE_WIDTH = 300

def resize_image_for_display(image_file):
    """Resize image for display only, returns bytes"""
    img = Image.open(image_file)
    aspect_ratio = img.height / img.width
    new_height = int(MAX_IMAGE_WIDTH * aspect_ratio)
    img = img.resize((MAX_IMAGE_WIDTH, new_height), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

@st.cache_resource
def get_agent():
    return Agent(
        model=Gemini(id="gemini-2.0-flash-exp"),  # Or another suitable large language model
        system_prompt=SYSTEM_PROMPT,
        instructions=INSTRUCTIONS,
        tools=[TavilyTools(api_key=os.getenv("TAVILY_API_KEY"))],  # Add Tavily
        markdown=True,
    )

def analyze_image(image_path):
    """Analyzes the image, extracts text with Gemini, and sends to the agent"""
    agent = get_agent()
    whisper_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")  # Load Whisper

    with st.spinner('Analyzing image and lyrics...'):
        try:
            # Extract text using Gemini
            response = agent.run(
                f"Analyze the lyrics in the image",
                images=[image_path]  # Pass the image to Gemini
            )
            extracted_text = response.content.strip()  # Extract text from response

            # Transcribe text using Whisper
            audio_bytes = st.audio(extracted_text)  # Simulate audio from text
            if audio_bytes:
                transcription = whisper_pipeline(audio_bytes)["text"]
                st.success(f"Whisper Transcription: {transcription}")

            st.markdown(response.content)  # Display Gemini analysis

        except Exception as e:
            st.error(f"An error occurred: {e}")

def save_uploaded_file(uploaded_file):
    """Saves uploaded file to a temp file, returns path"""
    with NamedTemporaryFile(dir='.', suffix='.jpg', delete=False) as f:
        f.write(uploaded_file.getbuffer())
        return f.name

def main():
    st.title("🎶 🎵🎼🎸🎹🎻Lyric Assistant & Music Production Guide🎙🎤🎸👨‍🎤🎶👩‍🎤")

    # Session state initialization
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = None
    if 'analyze_clicked' not in st.session_state:
        st.session_state.analyze_clicked = False

    tab_upload, tab_camera = st.tabs([
        "📤 Upload Image",
        "📸 Take Photo"
    ])

    # Upload Image Tab
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload image of lyrics",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of your song lyrics"
        )
        if uploaded_file:
            resized_image = resize_image_for_display(uploaded_file)
            st.image(resized_image, caption="Uploaded Image", use_container_width=False, width=MAX_IMAGE_WIDTH)
            if st.button("Analyze Lyrics", key="analyze_upload"):
                temp_path = save_uploaded_file(uploaded_file)
                analyze_image(temp_path)
                os.unlink(temp_path)  # Clean up temp file

    # Camera Input Tab
    with tab_camera:
        camera_photo = st.camera_input("Take a picture of your lyrics")
        if camera_photo:
            resized_image = resize_image_for_display(camera_photo)
            st.image(resized_image, caption="Captured Photo", use_container_width=False, width=MAX_IMAGE_WIDTH)
            if st.button("Analyze Lyrics", key="analyze_camera"):
                temp_path = save_uploaded_file(camera_photo)
                analyze_image(temp_path)
                os.unlink(temp_path)  # Clean up temp file

if __name__ == "__main__":
    st.set_page_config(
        page_title="Lyric Assistant",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    main()


