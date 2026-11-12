"""script.md -> finished prayer video (voice + pad music + captions + background) and thumbnail.

Run from the repo root (Windows or Linux):
  python tools/make_video.py videos/01-night-prayer-family --thumb "HE NEVER SLEEPS"
      [--engine chatterbox|kokoro] [--ref narrator.wav] [--theme night] [--nvenc]
      [--music track.mp3] [--bg image.png] [--reuse-voice]

Script conventions (see videos/01-*/script.md):
  ## BEAT ...                      section break (long pause)
  [ON SCREEN: ... Caption: "X"]    X becomes the small label at the top
  [PAUSE: N seconds ... Caption: "X"]  N seconds of music only, X shown as the main caption
  blank line                       paragraph pause
  any other line                   spoken, one caption block per line
"""
import argparse, json, os, re, subprocess, sys
import numpy as np, soundfile as sf

GAP_LINE, GAP_SCRIPTURE, GAP_PARA, GAP_BEAT = 0.55, 0.85, 1.3, 2.2
SPEED_PRAYER, SPEED_SCRIPTURE = 0.86, 0.80  # kokoro
# chatterbox: (exaggeration, cfg_weight). Lower cfg = slower, more deliberate; lower exaggeration = calmer.
CB_PRAYER, CB_SCRIPTURE = (0.45, 0.30), (0.30, 0.25)
KJV = re.compile(r"\b(thee|thou|thy|thine|shalt|hath|keepeth|dwelleth|saith|unto|ye|liest|makest|careth)\b", re.I)
FONT, LABEL_FONT = "Cormorant Garamond", "Cinzel"  # bundled in fonts/ (Google Fonts, OFL)
FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts")

# theme: sky gradient top->bottom, glow (colour, centre y as fraction of height), stars, accent (RRGGBB)
THEMES = {
    "night":     dict(c0="0x03050f", c1="0x16204a", glow="0xd9c3a0", gy=1.10, stars=True,  accent="E3C78A"),
    "dawn":      dict(c0="0x1a1030", c1="0x7a3d2c", glow="0xffc27a", gy=1.05, stars=False, accent="F2D29B"),
    "sanctuary": dict(c0="0x0b0605", c1="0x3a1414", glow="0xffb35c", gy=0.95, stars=False, accent="E8C27A"),
    "ocean":     dict(c0="0x020a10", c1="0x0f3a4a", glow="0xbfd8e6", gy=-0.05, stars=True, accent="CFE3EE"),
}


def bgr(rgb):  # ASS colours are &HAABBGGRR
    return f"&H00{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}"


def parse(md):
    """-> list of events: ('say', text, top_label) | ('gap', seconds) | ('pause', seconds, caption)"""
    ev, label, buf, started, para = [], "", None, False, []

    def flush(gap):
        if para:  # a paragraph may wrap across lines: rejoin, then speak sentence by sentence
            for s in re.split(r'(?<=[.!?"”])\s+(?=[A-Z"“])', " ".join(para)):
                ev.append(("say", s, label))
            para.clear()
        if gap and started and ev:  # never delay the hook
            if ev and ev[-1][0] == "gap":
                ev[-1] = ("gap", max(ev[-1][1], gap))
            else:
                ev.append(("gap", gap))

    for raw in open(md, encoding="utf-8"):
        line = raw.strip()
        if line.startswith("## BEAT"):
            flush(GAP_BEAT)
            started = True
            continue
        if not started or line.startswith("#") or line == "---":
            continue
        if buf is not None or line.startswith("["):  # bracket block, may span lines
            if buf is None:
                flush(0)
            buf = (buf + " " if buf else "") + line
            if "]" not in line:
                continue
            block, buf = buf, None
            cap = re.search(r'Caption:\s*"([^"]+)"', block)
            pause = re.match(r"\[PAUSE:\s*(\d+(?:\.\d+)?)", block)
            if pause:
                ev.append(("pause", float(pause.group(1)), cap.group(1) if cap else ""))
            elif cap:
                label = cap.group(1)
            continue
        if not line:
            flush(GAP_PARA)
            continue
        para.append(line)
    flush(0)
    return ev


def speakable(t):
    t = t.replace("LORD", "Lord").replace("2 a.m.", "two a.m.")
    t = re.sub(r'[“”"]', "", t).replace("—", ", ").replace("–", ", ")
    return t


def chunks(text, n=7):
    """Split a line into caption chunks of <= n words, preferring punctuation breaks."""
    words, out, cur = text.split(), [], []
    for w in words:
        cur.append(w)
        if len(cur) >= n or (len(cur) >= 4 and w[-1] in ",;:.!?"):
            out.append(" ".join(cur)); cur = []
    if cur:
        if out and len(cur) <= 2:
            out[-1] += " " + " ".join(cur)
        else:
            out.append(" ".join(cur))
    return out


def ts(s):
    h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def tts(engine, voice, ref=None):
    """-> (say(text, scripture) -> float32 mono array, sample_rate)"""
    if engine == "chatterbox":
        import torch
        from chatterbox.tts import ChatterboxTTS
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cpu":
            print("  warning: no CUDA GPU found, chatterbox on CPU is very slow")
        m = ChatterboxTTS.from_pretrained(device=dev)

        def say(text, scripture):
            ex, cfg = CB_SCRIPTURE if scripture else CB_PRAYER
            wav = m.generate(text, audio_prompt_path=ref, exaggeration=ex, cfg_weight=cfg)
            return wav.squeeze(0).cpu().numpy().astype(np.float32)
        return say, m.sr
    from kokoro_onnx import Kokoro
    md = os.environ.get("KOKORO_DIR", os.path.expanduser("~/.local/share/kokoro-models"))
    k = Kokoro(os.path.join(md, "kokoro-v1.0.onnx"), os.path.join(md, "voices-v1.0.bin"))
    lang = "en-gb" if voice.startswith("b") else "en-us"

    def say(text, scripture):
        return k.create(text, voice=voice, lang=lang, speed=SPEED_SCRIPTURE if scripture else SPEED_PRAYER)[0]
    return say, 24000


def build_audio(ev, out_dir, say, SR):
    parts, subs, t = [np.zeros(int(0.6 * SR), np.float32)], [], 0.6
    for i, e in enumerate(ev):
        if e[0] in ("gap", "pause"):
            parts.append(np.zeros(int(e[1] * SR), np.float32))
            if e[0] == "pause" and e[2]:
                subs.append((t, t + e[1], "Main", e[2]))
            t += e[1]
            continue
        _, text, label = e
        scripture = text.startswith(('"', "“")) or bool(KJV.search(text))
        audio = say(speakable(text), scripture)
        dur = len(audio) / SR
        cs = chunks(re.sub(r'[“”"]', "", text))
        total = sum(len(c) for c in cs)
        c0 = t
        for c in cs:  # ponytail: caption timing proportional to characters; per-word alignment if it drifts
            d = dur * len(c) / total
            subs.append((c0, c0 + d, "Main", c)); c0 += d
        if label:
            subs.append((t, t + dur, "Top", label))
        gap = GAP_SCRIPTURE if scripture else GAP_LINE
        parts += [audio.astype(np.float32), np.zeros(int(gap * SR), np.float32)]
        t += dur + gap
        print(f"\r  voice {i + 1}/{len(ev)}  {t / 60:5.1f} min", end="", flush=True)
    print()
    sf.write(f"{out_dir}/voice.wav", np.concatenate(parts), SR)
    return t, subs


def write_ass(subs, path, accent="E3C78A"):
    # merge consecutive identical top labels into one event
    tops, mains = [], [s for s in subs if s[2] == "Main"]
    for s in sorted((s for s in subs if s[2] == "Top"), key=lambda s: s[0]):
        if tops and tops[-1][3] == s[3] and s[0] - tops[-1][1] < 6:
            tops[-1] = (tops[-1][0], s[1], "Top", s[3])
        else:
            tops.append(s)
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,{FONT},88,&H00F2F4F7,&H00F2F4F7,&H60000000,&H00000000,-1,0,0,0,100,100,0.5,0,1,3,0,5,240,240,0,1
Style: Top,{LABEL_FONT},34,{bgr(accent)},{bgr(accent)},&H70000000,&H00000000,0,0,0,0,100,100,6,0,1,1.5,0,8,200,200,330,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(head)
        for a, b, st, txt in mains + tops:
            if st == "Main":  # soft dark halo instead of a hard outline, centred on screen
                fx = r"{\fad(350,350)\blur6}"
            else:  # small-caps label with hairlines either side
                fx, txt = r"{\fad(900,900)\blur1}", f"—   {txt.rstrip('.').upper()}   —"
            f.write(f"Dialogue: 0,{ts(a)},{ts(b)},{st},,0,0,0,,{fx}{txt}\n")


def ff(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-stats", *args], check=True)


def fpath(p):
    """Path safe inside an ffmpeg filter option: forward slashes, escaped drive colon (C:\\x -> C\\:/x)."""
    try:
        p = os.path.relpath(p)
    except ValueError:  # other drive on Windows
        p = os.path.abspath(p)
    return p.replace("\\", "/").replace(":", r"\:")


def make_background(path, theme="night"):
    t = THEMES[theme]
    stars = ("geq=lum='if(lt(random(1),0.0007*(1-Y/H)),170+85*random(2),0)',gblur=sigma=0.9,"
             "curves=all='0/0 0.08/0.6 1/1'") if t["stars"] else "geq=lum=0"
    gy = int(1080 * t["gy"])
    ff("-f", "lavfi", "-i", f"gradients=s=1920x1080:c0={t['c0']}:c1={t['c1']}:x0=960:y0=0:x1=960:y1=1080:nb_colors=2",
       "-f", "lavfi", "-i", f"color=black:s=1920x1080,format=gray,{stars}",
       "-f", "lavfi", "-i", f"color={t['glow']}:s=1920x1080,format=rgba,"
                            f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='110*exp(-((X-960)^2/(2*620^2)+(Y-{gy})^2/(2*300^2)))'",
       "-f", "lavfi", "-i", "color=black:s=1920x1080,format=gray,noise=alls=7:allf=u",  # faint film grain, kills banding
       "-filter_complex", "[0]format=gbrp[g];[1]format=gbrp[st];[g][st]blend=all_mode=screen[s];[s][2]overlay[o];"
                          "[3]format=gbrp[n];[o]format=gbrp[o2];[o2][n]blend=all_mode=addition:all_opacity=0.35,"
                          "vignette=PI/4.5",
       "-frames:v", "1", path)


def make_music(dur, path, track=None):
    if track:  # licensed track supplied: loop it to length
        ff("-stream_loop", "-1", "-i", track, "-t", f"{dur}", "-af",
           f"afade=t=in:d=4,afade=t=out:st={dur - 6}:d=6", "-ar", "48000", "-ac", "2", path)
        return
    # ponytail: generated A-major pad, swap for a YouTube Audio Library track via --music for a richer sound
    expr = ("0.05*(sin(2*PI*110*t)+0.6*sin(2*PI*164.81*t)+0.45*sin(2*PI*220*t)+0.3*sin(2*PI*277.18*t)"
            "+0.2*sin(2*PI*329.63*t+sin(2*PI*0.07*t)))*(0.78+0.22*sin(2*PI*t/11))")
    ff("-f", "lavfi", "-i", f"aevalsrc='{expr}':s=48000:d={dur}", "-af",
       "lowpass=f=1200,aecho=0.8:0.7:120|260:0.35|0.25,"
       f"afade=t=in:d=5,afade=t=out:st={dur - 8}:d=8,pan=stereo|c0=c0|c1=c0", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video_dir")
    p.add_argument("--engine", default="chatterbox", choices=["chatterbox", "kokoro"])
    p.add_argument("--voice", default="am_michael", help="kokoro voice name")
    p.add_argument("--ref", help="chatterbox: 10-20 s clean clip of the narrator voice to match (you must own the rights)")
    p.add_argument("--nvenc", action="store_true", help="encode on an NVIDIA GPU")
    p.add_argument("--thumb", default="")
    p.add_argument("--music")
    p.add_argument("--bg")
    p.add_argument("--theme", default="night", choices=sorted(THEMES))
    p.add_argument("--label", default="NIGHT PRAYER", help="small line under the thumbnail text")
    p.add_argument("--reuse-voice", action="store_true", help="skip TTS if voice.wav + subs.json exist")
    p.add_argument("--outro", type=float, default=35.0, help="seconds of music after the last line")
    a = p.parse_args()
    d = a.video_dir.rstrip("/\\")
    ev = parse(f"{d}/script.md")
    if not any(e[0] == "say" for e in ev):
        sys.exit("no spoken lines found in script.md")
    accent = THEMES[a.theme]["accent"]

    if a.reuse_voice and os.path.exists(f"{d}/voice.wav") and os.path.exists(f"{d}/subs.json"):
        print("1/5 reusing voice.wav + subs.json")
        t = sf.info(f"{d}/voice.wav").duration
        subs = json.load(open(f"{d}/subs.json"))
    else:
        print("1/5 voice")
        t, subs = build_audio(ev, d, *tts(a.engine, a.voice, a.ref))
        json.dump(subs, open(f"{d}/subs.json", "w"))
    print("2/5 captions"); write_ass(subs, f"{d}/captions.ass", accent)
    dur = t + a.outro
    print("3/5 background")
    bg = a.bg or f"{d}/background.png"
    if not a.bg:
        make_background(bg, a.theme)
    print("4/5 music + mix")
    make_music(dur, f"{d}/music.wav", a.music)
    ff("-i", f"{d}/voice.wav", "-i", f"{d}/music.wav", "-filter_complex",
       "[0]aresample=48000,highpass=f=70,equalizer=f=220:t=q:w=1:g=2,equalizer=f=3500:t=q:w=1.5:g=1.5,"
       "aecho=0.85:0.6:35|60:0.12|0.08,apad[v];[1]volume=0.55[m];"
       "[v][m]amix=inputs=2:duration=longest:weights=1 1:normalize=0,loudnorm=I=-14:TP=-1.5:LRA=11",
       "-t", f"{dur}", "-ar", "48000", f"{d}/mix.wav")
    print(f"5/5 render {dur / 60:.1f} min video")
    enc = (["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "21"] if a.nvenc else
           ["-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage", "-crf", "20"])
    ff("-loop", "1", "-framerate", "24", "-i", bg, "-i", f"{d}/mix.wav",
       "-vf", f"ass=filename={fpath(d + '/captions.ass')}:fontsdir={fpath(FONTS_DIR)},format=yuv420p", *enc,
       "-c:a", "aac", "-b:a", "192k", "-t", f"{dur}", "-movflags", "+faststart", f"{d}/video.mp4")
    if a.thumb:
        font = fpath(os.path.join(FONTS_DIR, "Cinzel.ttf"))
        esc = lambda s: s.replace(":", r"\:").replace("'", "’")
        ff("-i", bg, "-vf", f"scale=1280:720,drawtext=fontfile={font}:text='{esc(a.thumb.upper())}':fontsize=104:"
                            "fontcolor=0xF6F2EA:shadowcolor=0x000000@0.6:shadowx=0:shadowy=4:"
                            "x=(w-tw)/2:y=(h-th)/2-30,"
                            f"drawbox=x=(iw/2)-160:y=(ih/2)+52:w=320:h=2:color=0x{accent}@0.8:t=fill,"
                            f"drawtext=fontfile={font}:text='{esc(a.label.upper())}':fontsize=38:fontcolor=0x{accent}:"
                            "x=(w-tw)/2:y=(h/2)+78", "-frames:v", "1", f"{d}/thumbnail.png")
    for f in ("music.wav", "mix.wav"):  # keep voice.wav so --reuse-voice can skip the slow TTS step
        os.remove(f"{d}/{f}")
    print(f"done: {d}/video.mp4 ({dur / 60:.1f} min)")


if __name__ == "__main__":
    main()
