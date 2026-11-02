# He Got Nuked TWICE ☢️ — YouTube Short

`nuked_twice.mp4` — 1080x1920, 30fps, ~54s, voiceover + captions + music/SFX. Ready to upload.

**Title:** He survived TWO atomic bombs 💀
**Description:** Tsutomu Yamaguchi was in Hiroshima on Aug 6, 1945… then took the train home to Nagasaki. #shorts #history #facts #ww2 #crazystories

## Regenerate / remix
```bash
pip install piper-tts pillow numpy   # ffmpeg also required
# download a piper voice, e.g. en_US-ryan-high -> voice.onnx + voice.onnx.json
python3 make_short.py --voice voice.onnx --out nuked_twice.mp4
python3 make_short.py --voice voice.onnx --preview 2,10,30 --out sheet.png  # quick contact sheet
```
Edit `SEGMENTS` at the top of `make_short.py` to change the script.
