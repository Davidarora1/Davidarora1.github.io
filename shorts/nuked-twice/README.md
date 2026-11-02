# He Got Nuked TWICE ☢️ (YouTube Short, v2)

`nuked_twice.mp4`: 1080x1920, 30fps, about 63s. Real 1945–46 archive footage, neural voiceover, word-synced captions, score and sound design.

**Title:** He survived TWO atomic bombs 💀
**Description:** Tsutomu Yamaguchi was in Hiroshima on Aug 6, 1945… then took the train home to Nagasaki. #shorts #history #ww2 #facts

## Footage (public domain)
| Clip | Source (archive.org) |
|---|---|
| Nagasaki fireball (colour) | 428Npc2586NagasakiBurst (US Navy film, NARA) |
| Hiroshima ruins 1946 | USAF-11075 (NARA) |
| Hiroshima street life 1946 | USAF-11079 (NARA) |
| B-29s 1945 | CS-3338 (NARA) |
| Nagasaki damage | NPC-3243 (USMC, NARA) |
| Train | passenger_train (Prelinger) |
| Hiroshima cloud photo | propix.493984294 (US gov photo) |

## Rebuild
```bash
pip install edge-tts pillow numpy   # + ffmpeg
./fetch_footage.sh
python3 make_short.py --out nuked_twice.mp4
python3 make_short.py --preview 2,10,30 --out sheet.png   # quick contact sheet
```
To change the script, shots or overlays, edit `SEG` in `make_short.py`.
