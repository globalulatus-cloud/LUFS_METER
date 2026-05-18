# streamlit_app.py

import os
import tempfile
import subprocess
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import pyloudnorm as pyln
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="LUFS & True Peak Analyzer",
    layout="wide"
)


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# AUDIO CONVERSION
# =========================================================

def convert_to_wav(input_file: str) -> str:

    temp_wav = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    temp_wav.close()

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_file,
        "-vn",
        "-acodec",
        "pcm_f32le",
        "-ar",
        "48000",
        temp_wav.name
    ]

    try:

        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        return temp_wav.name

    except subprocess.CalledProcessError as e:

        logger.error(e.stderr)

        raise RuntimeError(
            f"FFmpeg conversion failed for: {input_file}"
        )


# =========================================================
# TRUE PEAK ANALYSIS USING FFMPEG
# =========================================================

def get_true_peak(file_path: str) -> float:

    command = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        file_path,
        "-filter_complex",
        "ebur128=peak=true",
        "-f",
        "null",
        "-"
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    output = result.stderr

    true_peak = None

    for line in output.splitlines():

        # Example:
        # Peak: -1.0 dBFS

        if "Peak:" in line:

            try:

                peak_text = line.split("Peak:")[1].strip()

                peak_text = peak_text.replace("dBFS", "").strip()

                true_peak = float(peak_text)

            except Exception:
                continue

    if true_peak is None:
        raise RuntimeError("Unable to detect true peak")

    return true_peak


# =========================================================
# AUDIO ANALYSIS
# =========================================================

def analyze_audio(file_path: str):

    temp_wav = None

    try:

        temp_wav = convert_to_wav(file_path)

        data, rate = sf.read(temp_wav)

        if len(data) == 0:
            raise ValueError("Empty audio file")

        data = data.astype(np.float32)

        if data.ndim == 1:
            channels = 1
        else:
            channels = data.shape[1]

        duration_seconds = len(data) / rate

        # =====================================================
        # LUFS ANALYSIS
        # =====================================================

        meter = pyln.Meter(rate)

        integrated_lufs = meter.integrated_loudness(data)

        # =====================================================
        # TRUE PEAK ANALYSIS
        # =====================================================

        true_peak = get_true_peak(file_path)

        return {
            "LUFS": round(float(integrated_lufs), 2),
            "True Peak (dBTP)": round(float(true_peak), 2),
            "Sample Rate": rate,
            "Channels": channels,
            "Duration (sec)": round(duration_seconds, 2)
        }

    except Exception as e:

        raise RuntimeError(str(e))

    finally:

        if temp_wav and os.path.exists(temp_wav):

            try:
                os.remove(temp_wav)

            except Exception:
                pass


# =========================================================
# UI
# =========================================================

st.title("Audio Loudness Analyzer")

st.write(
    "Upload WAV / MP3 / M4A / FLAC files and analyze LUFS and True Peak."
)

uploaded_files = st.file_uploader(
    "Upload Audio Files",
    type=["wav", "mp3", "m4a", "flac", "aac"],
    accept_multiple_files=True
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if uploaded_files:

    st.info(f"{len(uploaded_files)} file(s) ready for analysis")

    if st.button("Analyze Files", type="primary"):

        results = []

        progress_bar = st.progress(0)

        total_files = len(uploaded_files)

        for idx, uploaded_file in enumerate(uploaded_files):

            temp_input = None

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(uploaded_file.name).suffix
                ) as tmp:

                    tmp.write(uploaded_file.read())

                    temp_input = tmp.name

                analysis = analyze_audio(temp_input)

                analysis["File Name"] = uploaded_file.name

                analysis["Error"] = ""

                results.append(analysis)

            except Exception as e:

                logger.exception("Analysis failed")

                results.append({
                    "File Name": uploaded_file.name,
                    "LUFS": "ERROR",
                    "True Peak (dBTP)": "ERROR",
                    "Sample Rate": "-",
                    "Channels": "-",
                    "Duration (sec)": "-",
                    "Error": str(e)
                })

            finally:

                if temp_input and os.path.exists(temp_input):

                    try:
                        os.remove(temp_input)

                    except Exception:
                        pass

            progress = (idx + 1) / total_files

            progress_bar.progress(progress)

        # =====================================================
        # RESULTS TABLE
        # =====================================================

        st.success("Analysis Complete")

        df = pd.DataFrame(results)

        column_order = [
            "File Name",
            "LUFS",
            "True Peak (dBTP)",
            "Sample Rate",
            "Channels",
            "Duration (sec)",
            "Error"
        ]

        existing_columns = [
            col for col in column_order if col in df.columns
        ]

        df = df[existing_columns]

        st.dataframe(
            df,
            width="stretch"
        )

        # =====================================================
        # CSV DOWNLOAD
        # =====================================================

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download CSV Report",
            data=csv,
            file_name="audio_analysis_report.csv",
            mime="text/csv"
        )
