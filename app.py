import streamlit as st
import torch
from diffusers import AudioLDM2Pipeline
import scipy.io.wavfile

# --- App Configuration ---
st.set_page_config(page_title="🎶 AI Text-to-Audio Generator", layout="centered")
st.title("🎶 AI Text-to-Audio Generator")
st.write("Describe a sound, and let the AI create it for you!")

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the AudioLDM2 model from Hugging Face."""
    model_id = "cvssp/audioldm2"
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    return pipe, device

# Load the model
pipe, device = load_model()

# --- Audio Generation Function ---
def generate_audio(prompt, negative_prompt, duration_seconds, guidance):
    """Generates audio based on the text prompt."""
    try:
        # Generate the audio waveform
        # num_inference_steps can be increased for higher quality, but takes longer
        audio_output = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=200,
            audio_length_in_s=duration_seconds,
            guidance_scale=guidance
        ).audios[0]

        # Save the audio to a file in memory
        rate = 16000  # Sampling rate
        # Using a temporary file to be displayed by st.audio
        output_filename = "generated_audio.wav"
        scipy.io.wavfile.write(output_filename, rate, audio_output)
        
        return output_filename
    except Exception as e:
        st.error(f"An error occurred during audio generation: {e}")
        return None

# --- Streamlit UI ---
st.sidebar.header("How to Use")
st.sidebar.info(
    "1. **Enter a prompt:** Describe the sound you want (e.g., 'A cinematic orchestral piece with a sense of adventure').\n"
    "2. **(Optional) Enter a negative prompt:** Describe what you *don't* want (e.g., 'low quality, noisy, distorted').\n"
    "3. **Adjust settings:** Set the duration and guidance scale.\n"
    "4. **Click Generate!**"
)

st.header("Audio Generation Controls")

# Input fields
prompt = st.text_area(
    "Enter your audio description (Prompt):",
    "A calm, relaxing lo-fi hip hop beat with soft piano chords"
)
negative_prompt = st.text_input(
    "Enter what to avoid (Negative Prompt):",
    "low quality, harsh, noisy"
)
duration = st.slider("Duration (seconds)", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
guidance = st.slider("Guidance Scale", min_value=1.0, max_value=15.0, value=7.5, step=0.5)

if st.button("Generate Audio"):
    if prompt:
        with st.spinner("Generating audio... This might take a moment. ⏳"):
            audio_file = generate_audio(prompt, negative_prompt, duration, guidance)
            if audio_file:
                st.subheader("Your Generated Audio:")
                st.audio(audio_file)
    else:
        st.warning("Please enter a prompt to generate audio.")
