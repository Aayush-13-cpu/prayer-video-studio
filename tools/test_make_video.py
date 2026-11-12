"""Run: ~/.local/share/kokoro-venv/bin/python tools/test_make_video.py"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
from make_video import parse, chunks, GAP_BEAT, GAP_PARA

md = """# title
Voice: ignored header

## BEAT 1: Hook
[ON SCREEN: glow. Caption: "Label A"]

Before you sleep, pray this over
your family. Then rest.

## BEAT 2
[ON SCREEN: multi
line. Caption: "Label B"]
[PAUSE: 4 seconds. Caption: "Say their names."]
"Thou shalt not be afraid."
"""
with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
    f.write(md)
ev = parse(f.name)
assert ev[0] == ("say", "Before you sleep, pray this over your family.", "Label A"), ev[0]  # wrapped line rejoined, no lead gap
assert ev[1] == ("say", "Then rest.", "Label A")
assert ev[2] == ("gap", GAP_BEAT)  # para gap + beat gap collapse to the larger
assert ev[3] == ("pause", 4.0, "Say their names.")
assert ev[4] == ("say", '"Thou shalt not be afraid."', "Label B")  # multi-line bracket parsed
assert GAP_PARA < GAP_BEAT

assert chunks("one two three four five six seven eight") == ["one two three four five six seven eight"]  # 1-word tail merged
assert chunks("Protect us, from fear, and from every danger tonight Lord") == ["Protect us, from fear,", "and from every danger tonight Lord"]
print("ok")
