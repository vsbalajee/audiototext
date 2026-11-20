"""Murmur, a personal Whisper transcription studio."""
from hashlib import sha256
import logging
from pathlib import Path
from threading import Lock

import streamlit as st
import torch
import whisper
from whisper.tokenizer import LANGUAGES

from transcription import AudioError, MAX_MINUTES, decode_audio, subtitles, timestamp

LOGGER = logging.getLogger(__name__)
MODELS = {"Fast · Tiny": "tiny", "Balanced · Base": "base"}
LANGUAGE_OPTIONS = {name.title(): code for code, name in LANGUAGES.items()}
FORMATS = ["wav", "mp3", "m4a", "mp4", "aac", "flac", "ogg", "opus", "webm", "mov", "mpeg", "mpga", "wma"]

st.set_page_config(page_title="Murmur · Whisper Studio", page_icon="🎙️", layout="wide")


@st.cache_resource
def model_state():
    return {"lock": Lock(), "name": None, "model": None}


def transcribe(audio, model_name, options):
    state = model_state()
    with state["lock"]:
        if state["name"] != model_name:
            state.update(name=None, model=None)
            torch.set_num_threads(2)
            state["model"] = whisper.load_model(model_name, device="cpu")
            state["name"] = model_name
        return state["model"].transcribe(audio, fp16=False, verbose=None, **options)


st.html(Path(__file__).with_name("style.html"))
st.html("""
<header class="studio-hero">
  <div class="studio-content">
    <div class="studio-label">WHISPER STUDIO</div>
    <h1>Murmur</h1>
    <p>Turn recordings and conversations into text and subtitles.</p>
  </div>
  <div class="studio-wave" aria-hidden="true">
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
  </div>
</header>
""")

source, settings = st.columns([1.2, 1], gap="medium")
with source, st.container(border=True):
    st.subheader("Audio source")
    mode = st.radio("Audio source", ["Upload a file", "Record audio"], horizontal=True, label_visibility="collapsed")
    if mode == "Upload a file":
        media = st.file_uploader("Select an audio or video file", type=FORMATS)
    else:
        media = st.audio_input("Record with your microphone", sample_rate=16000)
    st.caption(f"Up to {MAX_MINUTES} minutes and 50 MB per recording")
    if media is not None and mode == "Upload a file":
        st.audio(media)

with settings, st.container(border=True):
    st.subheader("Transcription settings")
    task = st.radio("Output language", ["Original language", "Translate to English"], horizontal=True)
    language = st.selectbox("Spoken language", ["Auto-detect", *sorted(LANGUAGE_OPTIONS)])
    model = st.selectbox("Whisper model", list(MODELS), help="Tiny uses less memory. Base can improve accuracy at the cost of speed.")
    with st.expander("Advanced options"):
        prompt = st.text_area("Names or specialist terms", max_chars=500, placeholder="Project names, acronyms, or unusual spellings")
        word_timestamps = st.checkbox("Include word timestamps in JSON", disabled=task == "Translate to English")
        use_context = st.checkbox("Use preceding speech as context", value=True)
    run = st.button(
        "Translate audio" if task == "Translate to English" else "Transcribe audio",
        type="primary", width="stretch", disabled=media is None,
    )

data = media.getvalue() if media is not None else b""
signature = (sha256(data).digest(), model, task, language, prompt, word_timestamps, use_context)
if st.session_state.get("signature") != signature:
    st.session_state.pop("result", None)
    st.session_state["signature"] = signature

if run:
    st.session_state.pop("result", None)
    try:
        with st.spinner("Decoding audio…"):
            audio = decode_audio(data)
        with st.spinner("Processing speech with Whisper… The first run downloads the model."):
            st.session_state["result"] = transcribe(
                audio, MODELS[model],
                {
                    "task": "translate" if task == "Translate to English" else "transcribe",
                    "language": LANGUAGE_OPTIONS.get(language),
                    "initial_prompt": prompt.strip() or None,
                    "word_timestamps": word_timestamps and task == "Original language",
                    "condition_on_previous_text": use_context,
                },
            )
    except AudioError as error:
        st.warning(str(error))
    except (OSError, RuntimeError, ValueError):
        LOGGER.exception("Whisper processing failed")
        st.error("Whisper could not finish. Try again with Tiny or a shorter recording.")

result = st.session_state.get("result")
if result is not None:
    import json

    transcript = result["text"].strip()
    segments = [segment for segment in result["segments"] if segment["text"].strip()]
    if not transcript:
        st.info("Whisper did not find speech in this recording.")
    else:
        with st.container(border=True):
            st.subheader("Your transcript")
            detected = LANGUAGES.get(result["language"], result["language"]).title()
            st.caption(f"Spoken language · {detected}  |  {len(segments)} timestamped segments")
            text_tab, timestamps_tab = st.tabs(["Transcript", "Timestamps"])
            with text_tab:
                st.code(transcript, language=None, wrap_lines=True)
            with timestamps_tab:
                st.dataframe(
                    [
                        {"Start": timestamp(segment["start"]), "End": timestamp(segment["end"]), "Text": segment["text"].strip()}
                        for segment in segments
                    ],
                    hide_index=True, width="stretch",
                )
            stem = Path(media.name).stem or "transcript"
            exports = [
                ("TXT", transcript + "\n", "text/plain"),
                ("SRT", subtitles(segments, vtt=False), "application/x-subrip"),
                ("VTT", subtitles(segments, vtt=True), "text/vtt"),
                ("JSON", json.dumps(result, ensure_ascii=False, indent=2), "application/json"),
            ]
            for column, (extension, payload, mime) in zip(st.columns(4), exports):
                column.download_button(
                    f"Download {extension}", payload, f"{stem}.{extension.lower()}",
                    mime, width="stretch", on_click="ignore",
                )
