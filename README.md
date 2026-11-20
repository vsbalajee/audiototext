# Murmur · Whisper Studio

A personal audio-to-text studio built with Streamlit and OpenAI's open-source `openai-whisper` package. It performs CPU inference on the machine hosting the app. No OpenAI API key is required.

Upload audio or video, or record from the microphone. Select a spoken language or detect it automatically. Whisper can preserve the original language or translate speech into English. Export TXT, SRT, VTT, or JSON. JSON includes word timestamps when selected for original-language transcription.

## Run locally

On Linux with Python 3.12 and FFmpeg installed:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The requirements select an official CPU-only PyTorch wheel on Linux x86-64 with Python 3.12, 3.13, or 3.14. Other platforms use the standard PyPI PyTorch package.

## Streamlit Community Cloud

Deploy `vsbalajee/audiototext`, branch `main`, entry point `app.py`. Select Python 3.12 in advanced settings. Community Cloud installs FFmpeg from `packages.txt`. Set app access to private in the sharing settings before uploading personal recordings. If private sharing is unavailable, run the app locally for personal recordings.

Tiny is the default, with Base available for comparison. Each model downloads on first use. The app accepts up to 50 MB and 10 minutes of decoded audio. Processing time and usable model size depend on host resources. The app uses a temporary file while decoding and removes it when decoding finishes. Streamlit receives the audio on the host and the result is held in session memory.
