# Prayer Video Studio

Turns a `script.md` into a finished long-form prayer video: an AI voice with natural pacing, centered captions, a themed
background, a soft music bed, and a thumbnail. Built for **Victory Prayers** (Christian prayer) and **TapoBhumi**.

```
videos/01-night-prayer-family/script.md  ->  video.mp4 + thumbnail.png
```

## Windows setup (NVIDIA GPU)

1. Install the tools. Open PowerShell and run:
   ```powershell
   winget install Python.Python.3.11 Git.Git Gyan.FFmpeg
   ```
   Keep your NVIDIA driver up to date. Close and reopen PowerShell afterwards.
2. Clone the repo and create a Python environment:
   ```powershell
   git clone https://github.com/Aayush-13-cpu/prayer-video-studio.git
   cd prayer-video-studio
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Switch PyTorch to its GPU build. Chatterbox installs the CPU build, so reinstall the **same version** with CUDA:
   ```powershell
   pip show torch            # note the version, e.g. 2.6.0
   pip install --force-reinstall torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   The last line must print `True` and your GPU's name.
4. Run the self-check:
   ```powershell
   python tools\test_make_video.py
   ```

The first video you render downloads the Chatterbox model, a few GB, one time only.

## Make a video

```powershell
python tools\make_video.py videos\01-night-prayer-family --thumb "HE NEVER SLEEPS" --nvenc
```

| Option | What it does |
|---|---|
| `--theme night\|dawn\|sanctuary\|ocean` | Background and accent color (default `night`) |
| `--label "MORNING PRAYER"` | Small line under the thumbnail text |
| `--ref narrator.wav` | Chatterbox matches this voice. Use a 10–20 s clean clip that **you have the rights to**. |
| `--music track.mp3` | Music bed, looped (use the YouTube Audio Library for monetized videos) |
| `--bg image.png` | Your own 1920×1080 background instead of a theme |
| `--nvenc` | Encode on the GPU, much faster |
| `--reuse-voice` | Re-render visuals without regenerating the voice |
| `--engine kokoro` | Fallback CPU voice (needs `kokoro-onnx` + model files, see below) |

The voice settings are at the top of `tools/make_video.py`:

```python
CB_PRAYER, CB_SCRIPTURE = (0.45, 0.30), (0.30, 0.25)   # (exaggeration, cfg_weight)
```

Raise `exaggeration` for more emotion, and lower `cfg_weight` for slower, more deliberate delivery.

## Writing a script

Copy `videos/01-night-prayer-family/script.md` and edit it. The rules:

| In script.md | Becomes |
|---|---|
| `## BEAT …` | Section break (2.2 s pause) |
| `[ON SCREEN: … Caption: "For my children."]` | Section label above the captions |
| `[PAUSE: 4 seconds … Caption: "Say their names."]` | 4 s of music only, with that text on screen |
| Blank line | Paragraph pause (1.3 s) |
| Any other line | Spoken. Lines quoting KJV scripture are read slower. |

## YouTube skills (scripts, titles, SEO)

Install the free yt-* skills into Claude Code on Windows:

```powershell
git clone https://github.com/Jakeschincariol/youtube-agent-skill.git $env:TEMP\yt
Copy-Item -Recurse $env:TEMP\yt\skills\yt-* $env:USERPROFILE\.claude\skills\
New-Item -ItemType Directory -Force $env:USERPROFILE\.claude\youtube | Out-Null
Copy-Item channel\voice.md $env:USERPROFILE\.claude\youtube\voice.md
```

## Kokoro fallback (optional, CPU)

```powershell
pip install kokoro-onnx
mkdir models; cd models
curl.exe -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl.exe -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
cd ..; $env:KOKORO_DIR = "$PWD\models"
```

## Licenses

- Chatterbox: MIT (commercial use allowed; the audio carries an inaudible watermark).
- Kokoro: Apache 2.0.
- Fonts in `fonts/`: SIL Open Font License.
- KJV scripture: public domain.
