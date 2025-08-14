# LocalMedia2Text

**Offline media-to-text CLI** that turns **audio or video** files into clean `.txt` transcripts using **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)**.
It accepts **audio** (`.mp3 .wav .m4a .aac .ogg .flac`) and **video** (`.mp4 .mkv .mov .avi .webm`).
For video, LocalMedia2Text **extracts PCM audio via PyAV** (16 kHz, mono) and then transcribes with Whisper.
Runs fully **local** on **CPU (INT8)** or **GPU (FP16)**. No cloud calls, no accounts.

---

## Features

* **One-file CLI** (`transcribir_media.py`) — drop it next to your media and run.
* **Audio & video support**

  * Audio formats are fed directly to faster-whisper.
  * Video formats are auto-converted to **WAV 16 kHz mono (PCM)** via **PyAV**, then transcribed.
* **Smart device/precision**

  * Auto-detects GPU (CUDA). Falls back to **CPU** if GPU init fails.
  * Sensible defaults: **CPU → INT8**, **GPU → FP16**.
* **VAD + language autodetect**

  * Built-in **Voice Activity Detection** (Silero) trims silence.
  * **Language autodetection** unless you pin `--lang`.
* **Tidy output**

  * Produces a `.txt` with the **same basename** as the input (one line per segment).

---

## How it works

1. **Input detection**

   * If you don’t pass `--file`, the tool searches the script’s folder.
   * If it finds **exactly one** supported file, it uses that; otherwise it asks you to specify one.

2. **Decoding path**

   * **Audio** (`mp3/wav/m4a/aac/ogg/flac`): passed directly to faster-whisper (PyAV handles decoding).
   * **Video** (`mp4/mkv/mov/avi/webm`): PyAV demuxes/decodes audio, resamples to **16 kHz mono PCM**, writes a temporary **WAV**, then transcribes that WAV.

3. **Transcription**

   * Uses **faster-whisper** (CTranslate2 backend) with `beam_size=5`, **VAD** enabled, and optional language autodetect.
   * Writes a plaintext transcript: one segment per line.

4. **Device & precision**

   * Tries **CUDA** if present (default **FP16**); otherwise **CPU** (default **INT8**).
   * If GPU initialization fails (e.g., missing cuDNN), the script **automatically falls back to CPU**.

---

## Requirements

* **Python** 3.9+

* **Packages**:

  ```bash
  pip install faster-whisper av
  ```

  > `av` (PyAV) is used both by faster-whisper for audio decoding and by the script to extract audio from video.

* **Optional (GPU on Windows/Linux)**

  * **CUDA 12** + **cuDNN 9** installed and on `PATH` for best compatibility with recent CTranslate2 builds.
  * If you don’t have CUDA/cuDNN set up, just run `--device cpu`.

---

## Usage

From the folder where the script and your files live:

```bash
# Auto-pick the single supported file in the folder (if there's exactly one)
python transcribir_media.py

# Specify a file
python transcribir_media.py --file "2025-08-01 14-31-33.mp3"

# Force Spanish and CPU
python transcribir_media.py --file "meeting.mp4" --lang es --device cpu

# Use GPU with FP16
python transcribir_media.py --file "interview.mkv" --device cuda --compute_type float16

# Smaller model on CPU
python transcribir_media.py --file "note.m4a" --model base --device cpu

# Large model on GPU (more accurate, more VRAM)
python transcribir_media.py --file "podcast.mp4" --model large-v3 --device cuda --compute_type float16
```

### Output

* For an input like `meeting.mp4`, EchoScribe writes `meeting.txt` in the same folder.
* Each line is a transcription **segment** (sorted by time).

---

## CLI options

| Option           | Default  | Values                                                                       | Purpose                                                                                                 |
| ---------------- | -------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `--file PATH`    | *(auto)* | any supported file                                                           | File to transcribe. If omitted, the script looks for exactly one supported file in the script’s folder. |
| `--model NAME`   | `small`  | `tiny`, `base`, `small`, `medium`, `large-v3`, `distil-large-v3`, `turbo`, … | Whisper model size. Bigger → better accuracy, more RAM/compute.                                         |
| `--lang CODE`    | *(auto)* | e.g., `es`, `en`, `fr`, `de`, …                                              | Force a language. If omitted, language is auto-detected.                                                |
| `--device`       | *(auto)* | `cpu`, `cuda`                                                                | Force device. Auto: use GPU if available, else CPU.                                                     |
| `--compute_type` | *(auto)* | `int8`, `int8_float16`, `int16`, `float16`, `float32`                        | Numeric precision / quantization. Auto: `float16` on GPU, `int8` on CPU.                                |

### Recommendations

* **CPU**: `--compute_type int8` (default) for low memory; `int16` if you prefer a bit more accuracy.
* **GPU**: `--compute_type float16` (default) for speed; `int8_float16` to save VRAM with minimal quality loss.
* **Model choice**:

  * Low-resource / quick drafts → `base` / `small`
  * Better accuracy → `medium`
  * Highest accuracy (needs more VRAM/CPU) → `large-v3`

---

## Example prompt (Windows CMD / PowerShell / Linux/macOS)

```bash
# Spanish meeting on a modest PC (CPU only)
python transcribir_media.py --file "team_meeting.mp4" --lang es --device cpu

# GPU laptop, fast & accurate
python transcribir_media.py --file "user_interview.mkv" --device cuda --compute_type float16

# Batch from a script (PowerShell example)
Get-ChildItem -File -Include *.mp3,*.wav,*.m4a,*.aac,*.ogg,*.flac,*.mp4,*.mkv,*.mov,*.avi,*.webm |
  ForEach-Object { python transcribir_media.py --file "$($_.FullName)" --device cpu }
```

---

## Troubleshooting

* **`Could not locate cudnn_ops64_9.dll …`**
  Your CUDA/cuDNN isn’t available. Either install **CUDA 12 + cuDNN 9** properly or run with `--device cpu`.

* **Multiple supported files, but no `--file` passed**
  The tool requires a single obvious target. Use `--file` to specify which one.

* **PyAV import error**
  Ensure `pip install av` succeeded. If you’re in a venv/conda, activate it before installing.

* **Hugging Face cache symlink warning on Windows**
  Harmless. You can enable Developer Mode or run as admin to allow symlinks, but it’s optional.

---

## File list

* `transcribir_media.py` — the single CLI script described in this README.

---

## Roadmap

* Optional outputs: **SRT/VTT** subtitle files.
* Batch mode flag (transcribe **all** supported files in folder).
* Speaker diarization integration (WhisperX + pyannote, offline after model download).

---

## Why “EchoScribe”?

A minimal, private, **echo-to-text** assistant that “scribes” your media — no cloud, no noise, just transcripts.

---

## License (MIT)

```
MIT License

Copyright (c) 2025 Pablo Beret

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

## Credits

* Built on **faster-whisper** (SYSTRAN, CTranslate2).
* Uses **PyAV** to decode and resample media locally.

---
