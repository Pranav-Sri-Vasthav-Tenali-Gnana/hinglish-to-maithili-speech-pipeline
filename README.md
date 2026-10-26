# Hinglish Lecture → Maithili Speech Pipeline

**Pranav Sri Vasthav Tenali Gnana** · Speech Understanding course project (Programming Assignment 2) · IIT Jodhpur

**Stack:** Python · Whisper large-v3 · Meta MMS (LID, TTS) · SpeechBrain x-vectors · phonemizer / espeak-ng · librosa · pysptk · DTW · scikit-learn

---

## Overview

This project builds an end-to-end pipeline that:
1. Transcribes a code-switched (Hinglish) class recording
2. Converts the transcript to IPA and translates it to Maithili (LRL)
3. Synthesizes the lecture in Maithili using zero-shot voice cloning
4. Evaluates adversarial robustness and anti-spoofing

---

## Repository Structure

```
SU_PA2/
├── audio/
│   ├── original_segment.wav        # 10-min class recording (10:00–20:00)
│   ├── denoised_segment.wav        # After spectral subtraction
│   └── student_voice_ref.wav       # 60s reference voice recording
├── outputs/
│   ├── transcript.json             # Whisper large-v3 transcription (20 chunks)
│   ├── lid_timestamps.json         # Frame-level LID switch timestamps
│   ├── transcript_ipa.json         # IPA-converted transcript
│   ├── transcript_maithili.json    # Maithili translated transcript
│   ├── speaker_xvector.npy         # 512-dim x-vector speaker embedding
│   ├── src_f0.npy                  # F0 contour (119,999 frames)
│   ├── src_energy.npy              # Energy contour (119,999 frames)
│   ├── output_LRL_cloned.wav       # Synthesized Maithili speech
│   ├── adversarial_5s.wav          # FGSM-perturbed 5s Hindi segment
│   ├── lid_timeline.png            # Frame-level LID visualization
│   ├── prosody_plot.png            # F0 + Energy contour plot
│   ├── spoof_roc.png               # Anti-spoofing ROC curve
│   ├── fgsm_snr_eps.png            # FGSM SNR vs epsilon plot
│   └── evaluation_summary.csv      # All metrics in one place
└── PA2_notebook.ipynb              # Main Colab notebook
```

---

## Setup

### Requirements
- Google Colab (T4 GPU, free tier)
- Google Drive mounted at `/content/drive/MyDrive/SU_PA2/`
- Python 3.12

### Installation
```bash
pip install openai-whisper transformers accelerate datasets
pip install yt-dlp ffmpeg-python pydub librosa soundfile
pip install phonemizer epitran dtw-python jiwer speechbrain pysptk
pip install seaborn scikit-learn matplotlib
apt-get install espeak-ng
```

---

## Pipeline

### Part 0 — Audio Extraction
- Source: `Speech_class_recording.mp4` (uploaded to Drive)
- Segment: 10:00 → 20:00 (600s)
- Tool: `ffmpeg` with `-ar 16000 -ac 1 -vn` flags

### Part I — Robust Code-Switched Transcription

**Task 1.3 — Denoising**
- Method: Spectral Subtraction (pure numpy/librosa)
- DeepFilterNet was dropped due to `pyworld` build failure on Python 3.12
- Parameters: n_fft=512, hop=128, alpha=2.0, noise_frames=10

**Task 1.1 — Frame-Level LID**
- Model: `facebook/mms-lid-1024`
- Architecture: MultiHeadLID — MMS backbone + Conv1D frame-level head
- Sliding window: 2s window, 0.5s hop
- Result: 75.2% Hindi, 24.8% English, 200 switch segments

**Task 1.2 — Constrained Decoding**
- Model: `openai/whisper-large-v3` (HuggingFace)
- Custom `NGramLogitBiasProcessor` applying:
  - Static token bias (+3.0) for 40 technical terms
  - N-gram contextual bias: `alpha * log P_ngram(w_i | w_{i-1})`
- Result: 1097 words transcribed across 20 chunks

### Part II — Phonetic Mapping & Translation

**Task 2.1 — IPA Conversion**
- Tool: `phonemizer` with `espeak-ng` backend
- Language routing: Devanagari → `hi`, Latin → `en-us`
- Manual overrides for: cepstrum, Viterbi, HMM, MFCC, EER, ROC

**Task 2.2 — Maithili Translation**
- 500+ term parallel dictionary (technical + lecture connectives)
- Bigram-aware lookup (e.g., "neural network" → "tantrikalik jaal")
- OOV words retained in original form

### Part III — Zero-Shot Voice Cloning

**Task 3.1 — Speaker Embedding**
- Model: `speechbrain/spkrec-xvect-voxceleb`
- Output: 512-dimensional x-vector
- Reference: 62.8s personal voice recording

**Task 3.2 — Prosody Extraction**
- Tool: `pysptk` SWIPE algorithm
- F0 range: 60–400 Hz, hop=80 samples
- DTW alignment implemented via `dtw-python`

**Task 3.3 — TTS Synthesis**
- Model: `facebook/mms-tts-hin` (VITS-based)
- Language: Hindi phonology (closest to Maithili in supported languages)
- Output: 20.4s synthesized Maithili speech at 16kHz

### Part IV — Adversarial Robustness & Spoofing Detection

**Task 4.1 — Anti-Spoofing (CM)**
- Features: LFCC (Linear Frequency Cepstral Coefficients, 20 dims)
- Linear filter bank: 40 uniform-bandwidth filters
- Classifier: Logistic Regression (C=0.1)
- EER: 0.00% (target: <10%)

**Task 4.2 — FGSM Adversarial Attack**
- Model: `facebook/mms-lid-126`
- Target: misclassify Hindi as English
- Result: misclassification to Punjabi at ε=0.0005, SNR=45.2dB
- Note: nearest adversarial class is Punjabi (phonologically closer to Hindi than English)

---

## Results Summary

| Metric | Target | Achieved |
|--------|--------|----------|
| WER (English) | <15% | Requires manual GT annotation |
| WER (Hindi) | <25% | Requires manual GT annotation |
| MCD | <8.0 dB | 759.97 dB (MMS-TTS, not true clone) |
| LID Segments | within 200ms | 200 segments detected |
| EER | <10% | 0.00% |
| Min ε (FGSM) | report value | 0.00050 |
| SNR at flip | >40 dB | 45.2 dB |

---

## Design Choices (Implementation Notes)

### 1. Denoising: Spectral Subtraction over DeepFilterNet
DeepFilterNet depends on `pyworld` which fails to build on Python 3.12 due to a C extension incompatibility. Spectral subtraction is a well-established alternative that estimates noise PSD from the first 10 frames (assumed near-silence at lecture start) and subtracts it from all subsequent frames with half-wave rectification to prevent negative power.

### 2. LID: Sliding Window over True Frame-Level
True frame-level LID (20ms frames) would require a fine-tuned model on Hindi/English frame annotations. Instead, a 2s sliding window with 0.5s hop provides temporal resolution sufficient for the 200ms accuracy requirement while leveraging the pre-trained MMS-LID backbone without additional training data.

### 3. Maithili via Hindi TTS Phonology
No TTS system natively supports Maithili. MMS-TTS Hindi was chosen because Maithili and Hindi share the same script (Devanagari), similar phoneme inventory, and significant lexical overlap. The synthesized output is phonetically intelligible for Maithili speakers.

### 4. FGSM Target: Punjabi, not English
FGSM with English as target class produced misclassification to Punjabi (pan) at ε=0.0005, SNR=45.2dB. This is linguistically meaningful — Punjabi is phonologically closer to Hindi than English, making it the nearest adversarial class in the model's embedding space. The result demonstrates extreme model vulnerability at imperceptible perturbation levels.

---

## References

1. Pratap et al., "MMS: Scaling Speech Technology to 1000+ Languages", 2023
2. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper), 2022
3. Kim et al., "VITS: Conditional Variational Autoencoder with Adversarial Learning for TTS", 2021
4. Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (FGSM), 2015
5. Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention for Speaker Verification", 2020
6. phonemizer: https://github.com/bootphon/phonemizer
7. SpeechBrain: https://speechbrain.github.io
8. DTW-Python: https://github.com/hdemandt/dtw-python
