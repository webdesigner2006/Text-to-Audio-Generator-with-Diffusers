import streamlit as st
import torch
from diffusers import AudioLDM2Pipeline
import scipy.io.wavfile
import os
from typing import Optional

# --- Constants ---
# Using cvssp/audioldm2 as it's a solid text-to-audio model available on the Hub.
MODEL_ID = "cvssp/audioldm2"
OUTPUT_FILENAME = "generated_audio.wav"

# --- App Configuration ---
st.set_page_config(page_title="🎶 AI Audio Generator", layout="centered")
st.title("🎶 AI Text-to-Audio Generator")
st.write("Describe a sound, and I'll try my best to create it!")

# --- Model Loading ---
@st.cache_resource
def load_audio_pipeline():
    """
    Loads the AudioLDM2 model from Hugging Face.
    This is cached so it only runs once per session.
    """
    print("Loading audio pipeline for the first time...") # A simple debug print
    
    # Determine the best device and data type for performance
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if "cuda" in device else torch.float32

    try:
        audio_pipeline = AudioLDM2Pipeline.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
        print(f"Pipeline loaded successfully on {device}.")
        return audio_pipeline
    except Exception as e:
        st.error(f"Failed to load the model. It might be a network issue. Error: {e}")
        return None

# --- Audio Generation Logic ---
def generate_audio(pipeline, prompt: str, neg_prompt: str, duration_s: float, guidance: float) -> Optional[str]:
    """
    Generates audio using the loaded pipeline and returns the file path.
    Returns None if the generation process fails.
    """
    try:
        # NOTE: Higher inference steps lead to better quality but are slower.
        # 200 is a good balance for quality vs. speed with this model.
        audio_output = pipeline(
            prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=200,
            audio_length_in_s=duration_s,
            guidance_scale=guidance
        ).audios[0]

        # Save the audio file to disk
        sampling_rate = 16000 # This is the model's default sampling rate
        scipy.io.wavfile.write(OUTPUT_FILENAME, rate=sampling_rate, data=audio_output)
        
        return OUTPUT_FILENAME
        
    except Exception as e:
        st.error(f"An error occurred during audio generation: {e}")
        return None

# --- Streamlit UI ---
st.sidebar.title("👨‍💻 About This App")
st.sidebar.info(
    "This is a fun demo of the `AudioLDM 2` model from Hugging Face, wrapped in a simple Streamlit UI.\n\n"
    "**A few tips:**\n"
    "- Be descriptive in your prompt!\n"
    "- Use the negative prompt to avoid things like 'noise' or 'bad quality'.\n"
    "- Higher guidance makes the AI stick closer to your prompt."
)
st.sidebar.warning("**Note:** Audio generation can be slow, especially on a CPU. Please be patient!")

# Load the model
audio_pipeline = load_audio_pipeline()

# Only show the main interface if the model loaded correctly
if audio_pipeline:
    st.header("What sound do you want to create?")

    prompt_text = st.text_area(
        "Prompt:",
        "A cinematic orchestral piece with a sense of adventure",
        height=100,
        help="Describe the sound, music, or environment you want to hear."
    )
    negative_prompt_text = st.text_input(
        "Negative Prompt (what to avoid):",
        "low quality, harsh, noisy, distorted"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        duration_seconds = st.slider("Duration (seconds)", min_value=2.0, max_value=15.0, value=5.0)
    with col2:
        guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=15.0, value=7.5)

    if st.button("✨ Generate Audio"):
        if prompt_text:
            print(f"Generating audio for prompt: '{prompt_text}'") # Another debug print
            with st.spinner("Generating... this can take up to a minute... ⏳"):
                audio_file_path = generate_audio(
                    audio_pipeline,
                    prompt_text,
                    negative_prompt_text,
                    duration_seconds,
                    guidance_scale
                )

            if audio_file_path and os.path.exists(audio_file_path):
                st.subheader("🎵 Here's your sound:")
                st.audio(audio_file_path)
            else:
                st.error("Failed to generate audio. Please try a different prompt or check the console logs.")
        else:
            st.warning("Please enter a prompt to generate audio.")
