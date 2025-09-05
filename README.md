# Text-to-Audio-Generator-with-Diffusers
This project showcases generative audio AI. You provide a text prompt describing a sound or piece of music, and the app uses the Music Gen model via the Hugging Face Diffusers library to generate a corresponding audio clip.
# 🎶 AI Text-to-Audio Generator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.streamlit.app/) Turn your words into sound! This app uses generative AI to create audio clips from simple text descriptions. Whether you want to hear "a cinematic orchestral piece with a sense of adventure" or "the sound of rain falling on a tin roof," this tool brings your imagination to life.

It's a fun playground for musicians, sound designers, or anyone curious about the future of AI-powered creativity.



---

## ✨ Features

-   **Text-to-Audio**: Generate high-quality audio from any text prompt.
-   **Fine-Tuning Controls**:
    -   Use **Negative Prompts** to specify what you *don't* want to hear.
    -   Adjust the **Duration** of the audio clip.
    -   Control the **Guidance Scale** to influence how closely the AI follows your prompt.
-   **Interactive UI**: A simple and intuitive interface built with Streamlit.
-   **Powered by AudioLDM 2**: Uses a cutting-edge model for text-to-audio synthesis.

---

## 🛠️ Tech Stack

-   **Language**: Python
-   **Framework**: Streamlit
-   **Core AI**: Hugging Face Diffusers
-   **Deep Learning**: PyTorch
-   **Audio Processing**: SciPy

---

## 🚀 How to Run It Locally

Let's get you making some noise! Follow these steps to run the app locally.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/text_to_audio.git](https://github.com/your-username/text_to_audio.git)
    cd text_to_audio
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Heads up: The first time you run the app, it will download the AudioLDM 2 model from Hugging Face. It's a fairly large download, so please be patient!*

4.  **Launch the app:**
    ```bash
    streamlit run app.py
    ```

The app will open in your web browser. Start typing and see what sounds you can create!
