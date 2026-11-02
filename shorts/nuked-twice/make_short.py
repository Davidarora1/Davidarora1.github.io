#!/usr/bin/env python3
"""
"He Got Nuked TWICE" v2 — YouTube Short (1080x1920, 30fps) built from REAL 1945-46
public-domain archive footage (US National Archives, via archive.org) + a neural
narrator voice with exact word-timed captions.

True story: Tsutomu Yamaguchi, the only person officially recognized by Japan as
surviving both the Hiroshima and Nagasaki atomic bombings.

Pipeline:
  1. fetch_footage.sh downloads the public-domain clips into ./footage
  2. edge-tts voices each segment and gives word timestamps (captions sync exactly)
  3. ffmpeg decodes each shot; PIL composites (blur-fill layout, pans, flashes,
     shake, RGB split, callouts, map inset, captions, grain)
  4. numpy synthesizes a cinematic score + sound design; ffmpeg muxes the MP4

Usage:
  pip install edge-tts pillow numpy      # + ffmpeg
  ./fetch_footage.sh
  python3 make_short.py --out nuked_twice.mp4
  python3 make_short.py --preview 2,10,30 --out sheet.png   # contact sheet
"""
import argparse, asyncio, math, os, random, subprocess, wave, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

W, H, FPS, SR = 1080, 1920, 30, 44100
HERE = os.path.dirname(os.path.abspath(__file__))
FOOT = os.path.join(HERE, "footage")
INTER_BLACK = "/usr/share/fonts/opentype/inter/Inter-Black.otf"
INTER_BOLD = "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"
EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
VOICE = "en-US-AndrewMultilingualNeural"

# foreground video window
FW, FH = 1080, 1240
FY = 170                      # top of the footage window
PAD = 1.1                     # decode bigger than window so we can pan

# Sources: path, crop box in source pixels usable area (x0,y0,x1,y1)
SRC = {
    "burst": ("naga_burst.mp4", (0, 0, 1280, 620)),     # Nagasaki fireball, colour (US Navy)
    "hdmg":  ("hiro_damage.mp4", (0, 0, 640, 360)),     # Hiroshima ruins 1946 (USAF)
    "life":  ("hiro_life.mp4", (0, 0, 640, 360)),       # Hiroshima street life 1946 (USAF)
    "b29":   ("b29.mp4", (0, 0, 640, 360)),             # B-29s, 1945 (USAF)
    "ndmg":  ("naga_damage.ogv", (0, 0, 532, 300)),     # Nagasaki ruins 1945 (USMC)
    "train": ("train.mp4", (0, 0, 320, 240)),           # 1940s train
    "cloud": ("hiro_cloud.jpg", (110, 90, 537, 640)),   # Hiroshima cloud photo (US gov)
}

# ------------------------------------------------------------------ script --
# say: narration. cap: caption override words (same count as say words) optional.
# shots: (src, start_sec, focus_x 0..1, weight)
# ov: overlays (kind, arg, at_fraction_of_segment)
SEG = [
    dict(say="This man survived an atomic bomb.", flash=True,
         shots=[("burst", 2.0, .42, 1), ("cloud", 0, .5, 1.2)],
         ov=[("big", ("NUKED.", (255, 230, 0)), .45)]),
    dict(say="Then he took a train... straight into the second one.",
         shots=[("train", 498, .5, 1), ("train", 182, .5, 1), ("burst", 33, .5, 1.3)],
         ov=[("big", ("x2", (255, 50, 50)), .78)]),
    dict(say="August 6th, 1945. Tsutomu Yamaguchi, a 29 year old engineer, is in Hiroshima on a work trip.",
         shots=[("life", 495, .55, 1.2), ("life", 247, .5, 1), ("life", 326, .5, 1)],
         ov=[("date", "AUG 6, 1945", .0), ("loc", "HIROSHIMA, JAPAN", .5)]),
    dict(say="It's his last day. He's literally about to go home.",
         shots=[("life", 232, .5, 1), ("life", 300, .5, 1)], ov=[("emoji", "🧳", .5)]),
    dict(say="8:15 AM. He looks up... and sees a plane.",
         shots=[("b29", 356, .5, 1.2), ("b29", 431, .45, 1)],
         ov=[("date", "8:15 AM", .0)], tick=True),
    dict(say="Then the sky explodes.", flash=True, pause=0.35,
         shots=[("burst", 6.0, .45, 1), ("cloud", 0, .5, 1.3)], ov=[]),
    dict(say="He's under 2 miles away. His eardrums burst. He's burned... and he can't see.",
         shots=[("hdmg", 797, .45, 1), ("hdmg", 182, .5, 1), ("hdmg", 271, .45, 1.2)],
         ov=[("tag", ("📍", "1.9 MILES AWAY"), .0), ("tag", ("👂", "EARDRUMS BURST"), .3),
             ("tag", ("🔥", "BURNED"), .52), ("tag", ("👁️", "BLINDED"), .72)], blind=True),
    dict(say="But somehow... he's alive.", pause=0.25,
         shots=[("hdmg", 821, .4, 1)], ov=[("big", ("ALIVE.", (60, 255, 120)), .45)], heartbeat=True),
    dict(say="So the next morning, he does the most normal thing ever. He takes the train home.",
         shots=[("hdmg", 857, .4, 1), ("train", 178, .5, 1), ("train", 500, .5, 1)], ov=[("map", "route", .4)]),
    dict(say="Home... was Nagasaki.", pause=0.15,
         shots=[("train", 381, .5, 1)], ov=[("map", "naga", .0)], ominous=True),
    dict(say="August 9th. Covered in bandages, he shows up to work anyway.", pause=0.3,
         shots=[("life", 269, .5, 1), ("life", 356, .5, 1)],
         ov=[("date", "AUG 9, 1945", .0), ("loc", "NAGASAKI, JAPAN", .35)], tick=True),
    dict(say="He tells his boss one bomb destroyed an entire city... and his boss thinks he's lost it.",
         shots=[("life", 272, .4, 1), ("life", 309, .5, 1)], ov=[("emoji", "🤨", .66)]),
    dict(say="And right at that moment...", riser=True,
         shots=[("life", 498, .75, 1)], ov=[]),
    dict(say="the room goes white. Again.", flash=True, pause=0.4,
         shots=[("burst", 0.5, .45, 1), ("burst", 10, .42, 1), ("burst", 31, .5, 1)],
         ov=[("big", ("AGAIN.", (255, 50, 50)), .45), ("rec", "REAL FOOTAGE • NAGASAKI • AUG 9, 1945", .62)]),
    dict(say="Same flash. Same distance. Just 3 days apart.",
         shots=[("ndmg", 2, .5, 1), ("ndmg", 45, .5, 1), ("ndmg", 67, .5, 1)],
         ov=[("tag", ("☢️", "HIROSHIMA • 1.9 MI"), .05), ("tag", ("☢️", "NAGASAKI • 1.9 MI"), .3)]),
    dict(say="And he survived. Again.", pause=0.2,
         shots=[("burst", 116, .5, 1)], ov=[("big", ("SURVIVED x2", (60, 255, 120)), .5)], heartbeat=True),
    dict(say="He lived to 93. The only person Japan ever officially recognized as surviving both bombs.",
         shots=[("life", 541, .5, 1), ("life", 603, .5, 1), ("life", 630, .5, 1)],
         ov=[("count", 93, .0), ("badge", "ONLY OFFICIAL DOUBLE SURVIVOR", .3)]),
    dict(say="Some people have a bad week. This man survived the worst week in human history.", pause=0.3,
         shots=[("hdmg", 801, .5, 1)], ov=[("emoji", "💀", .75)]),
    dict(say="Follow for more.", shots=[("hdmg", 812, .5, 1)], ov=[("follow", None, .0)]),
]

# ----------------------------------------------------------------- helpers --
_fc = {}
def font(p, s):
    if (p, s) not in _fc: _fc[(p, s)] = ImageFont.truetype(p, s)
    return _fc[(p, s)]

_ec = {}
def emoji(ch, size):
    if (ch, size) not in _ec:
        im = Image.new("RGBA", (260, 160))
        ImageDraw.Draw(im).text((4, 4), ch, font=ImageFont.truetype(EMOJI, 109), embedded_color=True)
        im = im.crop(im.getbbox()); s = size / max(im.size)
        _ec[(ch, size)] = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    return _ec[(ch, size)]

def text_img(txt, size, fill=(255, 255, 255), stroke=10, path=INTER_BLACK, sf=(0, 0, 0)):
    f = font(path, size); l, t, r, b = f.getbbox(txt, stroke_width=stroke)
    im = Image.new("RGBA", (r - l + 8, b - t + 8))
    ImageDraw.Draw(im).text((4 - l, 4 - t), txt, font=f, fill=fill, stroke_width=stroke, stroke_fill=sf)
    return im

def shadowed(im, blur=14, alpha=170):
    pad = blur * 3
    out = Image.new("RGBA", (im.width + pad * 2, im.height + pad * 2))
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0)); sh.putalpha(im.getchannel("A").point(lambda v: v * alpha // 255))
    out.alpha_composite(sh, (pad, pad + 6)); out = out.filter(ImageFilter.GaussianBlur(blur))
    out.alpha_composite(im, (pad, pad)); return out

def paste_c(base, im, cx, cy, scale=1.0, alpha=1.0, rot=0):
    if scale <= 0.02 or alpha <= 0.02: return
    if abs(scale - 1) > 1e-3:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BILINEAR)
    if rot: im = im.rotate(rot, resample=Image.BICUBIC, expand=True)
    if alpha < 1:
        im = im.copy(); im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
    base.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))

def eob(x):
    x = min(max(x, 0), 1); return 1 + 2.70158 * (x - 1) ** 3 + 1.70158 * (x - 1) ** 2
def eo(x):
    x = min(max(x, 0), 1); return 1 - (1 - x) ** 3
def c01(x): return min(max(x, 0.0), 1.0)

def pill(w, h, fill, outline=None, r=None, width=6):
    im = Image.new("RGBA", (w, h))
    ImageDraw.Draw(im).rounded_rectangle((0, 0, w - 1, h - 1), r if r is not None else h // 2, fill=fill, outline=outline, width=width)
    return im

# --------------------------------------------------------- static layers ----
def _vignette():
    y, x = np.mgrid[0:H, 0:W]
    d = np.sqrt(((x - W / 2) / (W * .7)) ** 2 + ((y - H / 2) / (H * .62)) ** 2)
    a = (np.clip((d - .5) * 1.5, 0, 1) * 210).astype(np.uint8)
    im = np.zeros((H, W, 4), np.uint8); im[..., 3] = a; return Image.fromarray(im)
VIGNETTE = _vignette()

def _fgmask():
    m = np.full((FH, FW), 255, np.float32)
    ramp = 90
    r = np.linspace(0, 1, ramp) ** 1.5
    m[:ramp] *= r[:, None]; m[-ramp:] *= r[::-1][:, None]
    return Image.fromarray(m.astype(np.uint8))
FGMASK = _fgmask()

GRAIN = [Image.fromarray(np.dstack([np.random.default_rng(i).integers(0, 255, (H // 3, W // 3), dtype=np.uint8)] * 3 +
         [np.full((H // 3, W // 3), 22, np.uint8)])).resize((W, H)) for i in range(8)]

def _scratches(i):
    rnd = random.Random(i); im = Image.new("RGBA", (W, H)); d = ImageDraw.Draw(im)
    for _ in range(rnd.randint(0, 2)):
        x = rnd.randint(0, W); d.line((x, 0, x + rnd.randint(-20, 20), H), fill=(255, 255, 255, rnd.randint(25, 60)), width=2)
    for _ in range(rnd.randint(0, 4)):
        x, y = rnd.randint(0, W), rnd.randint(FY, FY + FH); r = rnd.randint(2, 6)
        d.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 0, 120) if rnd.random() < .6 else (255, 255, 255, 90))
    return im
SCRATCH = [_scratches(i) for i in range(12)]

# map inset
KYUSHU = [(129.7, 33.35), (130.05, 33.85), (130.55, 33.95), (130.95, 33.92), (131.05, 33.6), (131.7, 33.35), (131.6, 32.9),
          (131.9, 32.5), (131.45, 31.4), (131.1, 31.25), (130.7, 31.0), (130.2, 31.3), (130.15, 31.75), (130.35, 32.1),
          (130.6, 32.7), (130.2, 32.65), (129.95, 32.55), (129.7, 32.75), (129.85, 33.05), (129.55, 33.2)]
HONSHU = [(130.9, 33.95), (130.95, 34.35), (131.4, 34.45), (131.8, 34.65), (132.3, 35.1), (132.9, 35.5), (133.4, 35.55),
          (134.3, 35.6), (134.6, 35.6), (134.6, 34.7), (134.2, 34.6), (133.6, 34.45), (133.0, 34.35), (132.5, 34.3),
          (132.2, 33.95), (131.8, 34.0), (131.3, 33.95)]
SHIKOKU = [(132.05, 33.3), (132.4, 33.75), (132.75, 33.95), (133.2, 33.95), (133.6, 34.25), (134.2, 34.3), (134.65, 34.2),
           (134.75, 33.85), (134.3, 33.6), (133.9, 33.4), (133.4, 33.35), (133.0, 32.75), (132.5, 32.85)]
HIRO, NAGA = (132.46, 34.39), (129.87, 32.75)
MS = 560
def ll(lon, lat): return (30 + (lon - 129.4) * 98, 40 + (35.9 - lat) * 112)
def _map_base():
    im = Image.new("RGBA", (MS, MS)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, MS - 1, MS - 1), 36, fill=(8, 12, 28, 225), outline=(255, 255, 255, 90), width=3)
    land = Image.new("RGBA", (MS, MS)); ld = ImageDraw.Draw(land)
    for poly in (KYUSHU, HONSHU, SHIKOKU):
        pts = [ll(*p) for p in poly]
        ld.polygon(pts, fill=(45, 60, 105, 255)); ld.line(pts + [pts[0]], fill=(130, 180, 255, 255), width=4)
    im.alpha_composite(land.filter(ImageFilter.GaussianBlur(10))); im.alpha_composite(land)
    return im
MAPBASE = _map_base()
def map_panel(t, mode):
    im = MAPBASE.copy(); d = ImageDraw.Draw(im)
    (x0, y0), (x1, y1) = ll(*HIRO), ll(*NAGA)
    cx, cy = (x0 + x1) / 2 + 10, (y0 + y1) / 2 - 90
    p = c01(t / 1.6) if mode == "route" else 1
    pts = [((1 - u) ** 2 * x0 + 2 * (1 - u) * u * cx + u * u * x1, (1 - u) ** 2 * y0 + 2 * (1 - u) * u * cy + u * u * y1)
           for u in np.linspace(0, p, 60)]
    for i in range(0, len(pts) - 1, 2): d.line((pts[i], pts[i + 1]), fill=(255, 230, 0, 255), width=7)
    def dot(x, y, col, lab, below):
        if mode == "naga" and lab == "NAGASAKI":
            ph = (t * 1.3) % 1; r = 14 + ph * 60
            d.ellipse((x - r, y - r, x + r, y + r), outline=(255, 50, 50, int(255 * (1 - ph))), width=5)
        d.ellipse((x - 12, y - 12, x + 12, y + 12), fill=col, outline=(255, 255, 255, 255), width=4)
        tx = text_img(lab, 30, stroke=6); im.alpha_composite(tx, (int(x - tx.width / 2), int(y + 18 if below else y - 52)))
    dot(x0, y0, (255, 140, 0, 255), "HIROSHIMA", False)
    if p > .98: dot(x1, y1, (255, 50, 50, 255), "NAGASAKI", True)
    if p < 1: paste_c(im, emoji("🚆", 64), pts[-1][0], pts[-1][1] - 26)
    return im

# ----------------------------------------------------------- footage I/O ----
def shot_frames(src, start, fx, n):
    """Yield n RGB PIL frames (FW*PAD x FH*PAD) for a shot."""
    path, (cx0, cy0, cx1, cy1) = SRC[src]
    path = os.path.join(FOOT, path)
    OW, OH = int(FW * PAD) // 2 * 2, int(FH * PAD) // 2 * 2
    ar = FW / FH
    sw, sh = cx1 - cx0, cy1 - cy0
    cw, ch = (sh * ar, sh) if sw / sh > ar else (sw, sw / ar)
    x = cx0 + (sw - cw) * fx; y = cy0 + (sh - ch) / 2
    if path.endswith(".jpg"):
        im = Image.open(path).convert("RGB").crop((int(x), int(y), int(x + cw), int(y + ch))).resize((OW, OH), Image.LANCZOS)
        for _ in range(n): yield im
        return
    vf = (f"crop={int(cw)}:{int(ch)}:{int(x)}:{int(y)},scale={OW}:{OH}:flags=bicubic,"
          f"eq=contrast=1.12:saturation=1.15:brightness=0.01,unsharp=7:7:0.9,fps={FPS}")
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-ss", str(start), "-i", path, "-t", str(n / FPS + 0.5),
                          "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    last = None
    for _ in range(n):
        buf = p.stdout.read(OW * OH * 3)
        if len(buf) == OW * OH * 3: last = Image.frombytes("RGB", (OW, OH), buf)
        yield last
    p.stdout.close(); p.kill()

# ------------------------------------------------------------------ voice ---
async def _tts(text, path):
    import edge_tts
    c = edge_tts.Communicate(text, VOICE, rate="+12%", boundary="WordBoundary")
    words = []
    with open(path, "wb") as f:
        async for ch in c.stream():
            if ch["type"] == "audio": f.write(ch["data"])
            elif ch["type"] == "WordBoundary":
                words.append((ch["text"], ch["offset"] / 1e7, (ch["offset"] + ch["duration"]) / 1e7))
    return words

def tts(text, work, i):
    try:
        import certifi; certifi.where = (lambda: os.environ["SSL_CERT_FILE"]) if os.environ.get("SSL_CERT_FILE") else certifi.where
    except ImportError: pass
    mp3 = os.path.join(work, f"v{i:02d}.mp3"); wav = os.path.join(work, f"v{i:02d}.wav")
    for attempt in range(6):
        try:
            words = asyncio.run(_tts(text, mp3)); break
        except Exception as ex:
            print("tts retry", attempt, ex); import time; time.sleep(2 + attempt * 2)
    else:
        raise RuntimeError("tts failed")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp3, "-ar", str(SR), "-ac", "1",
                    "-af", "highpass=f=70,acompressor=threshold=-20dB:ratio=3:attack=5:release=80,volume=1.4", wav], check=True)
    with wave.open(wav) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    # trim leading silence (edge-tts adds ~50-100ms)
    lead = max(0.0, words[0][1] - 0.04) if words else 0
    a = a[int(lead * SR):]
    words = [(t, s - lead, e - lead) for t, s, e in words]
    end = words[-1][2] + 0.12 if words else len(a) / SR
    return a[:int(end * SR)], words

# ---------------------------------------------------------------- captions --
EMPH = {"atomic", "bomb", "second", "explodes", "alive", "nagasaki", "again", "white", "93", "both", "lost", "worst",
        "unluckiest", "luckiest", "train", "blind", "can't", "see", "burned", "burst", "real"}
def chunk_words(words, maxw=3, maxc=15):
    out, cur = [], []
    for i, (w, s, e) in enumerate(words):
        cur.append(i)
        n = sum(len(words[j][0]) for j in cur)
        gap = (words[i + 1][1] - e) if i + 1 < len(words) else 1
        if len(cur) >= maxw or n >= maxc or gap > 0.18:
            out.append(cur); cur = []
    if cur: out.append(cur)
    return out

_cc = {}
def caption(words, idxs, act):
    k = (tuple(words[i][0] for i in idxs), tuple(idxs), act)
    if k in _cc: return _cc[k]
    f = font(INTER_BLACK, 104); sp = 26
    parts = []
    for i in idxs:
        w = words[i][0].upper()
        core = w.strip(".,!?").lower()
        col = (255, 255, 255)
        if core in EMPH: col = (255, 70, 70)
        if i == act: col = (255, 226, 0) if core not in EMPH else (255, 70, 70)
        parts.append((w, col))
    ws = [f.getbbox(p[0], stroke_width=12)[2] for p in parts]
    tot = sum(ws) + sp * (len(parts) - 1)
    im = Image.new("RGBA", (tot + 40, 190)); d = ImageDraw.Draw(im); x = 20
    for (w, col), wd in zip(parts, ws):
        d.text((x, 30), w, font=f, fill=col, stroke_width=12, stroke_fill=(0, 0, 0)); x += wd + sp
    s = min(1.0, 1000 / tot)
    if s < 1: im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    _cc[k] = shadowed(im, 10, 150); return _cc[k]

# ---------------------------------------------------------------- overlays --
def draw_overlay(base, kind, arg, lt, tags_state):
    if lt < 0: return
    a = eob(lt / 0.28)
    if kind == "big":
        txt, col = arg
        im = shadowed(text_img(txt, 210 if len(txt) < 8 else 140, fill=col, stroke=16))
        paste_c(base, im, W / 2, FY + FH * .42, scale=a * (1 + .03 * math.sin(lt * 9)), rot=-4)
    elif kind == "date":
        t = text_img(arg, 64, fill=(0, 0, 0), stroke=0)
        p = pill(t.width + 70, 112, (255, 226, 0, 255), r=22); p.alpha_composite(t, (35, (112 - t.height) // 2))
        paste_c(base, shadowed(p), W / 2, FY + 40, scale=a, rot=-2)
    elif kind == "loc":
        t = text_img(arg, 52, stroke=0); e = emoji("📍", 64)
        p = pill(t.width + e.width + 90, 100, (10, 10, 14, 220), r=50)
        p.alpha_composite(e, (30, (100 - e.height) // 2)); p.alpha_composite(t, (e.width + 50, (100 - t.height) // 2))
        paste_c(base, shadowed(p), W / 2, FY + FH - 110, scale=a)
    elif kind == "tag":
        em, txt = arg
        idx = tags_state.setdefault("n", 0); tags_state["n"] += 1
        t = text_img(txt, 54, stroke=0); e = emoji(em, 70)
        p = pill(t.width + e.width + 90, 108, (12, 12, 16, 230), outline=(255, 70, 70, 255), r=28, width=5)
        p.alpha_composite(e, (28, (108 - e.height) // 2)); p.alpha_composite(t, (e.width + 52, (108 - t.height) // 2))
        paste_c(base, shadowed(p), W / 2, FY + 80 + idx * 130, scale=a)
    elif kind == "emoji":
        paste_c(base, emoji(arg, 260), W * .76, FY + FH * .62, scale=a * (1 + .05 * math.sin(lt * 10)), rot=10)
    elif kind == "map":
        paste_c(base, shadowed(map_panel(lt, arg)), W - MS / 2 - 30, FY + MS / 2 + 30, scale=a * .92)
    elif kind == "rec":
        t = text_img(arg, 40, stroke=0)
        p = pill(t.width + 100, 76, (0, 0, 0, 170), r=14)
        if int(lt * 2) % 2 == 0: ImageDraw.Draw(p).ellipse((24, 24, 52, 52), fill=(255, 40, 40, 255))
        p.alpha_composite(t, (70, (76 - t.height) // 2))
        base.alpha_composite(p, (int((W - p.width) / 2), FY + 30))
    elif kind == "count":
        n = int(arg * eo(lt / 1.6))
        num = shadowed(text_img(str(n), 300, fill=(255, 226, 0), stroke=16))
        paste_c(base, num, W / 2, FY + 250, scale=a)
        paste_c(base, shadowed(text_img("YEARS OLD", 80, stroke=10)), W / 2, FY + 450, scale=a)
    elif kind == "badge":
        lines = ["ONLY OFFICIAL", "DOUBLE SURVIVOR"]
        p = pill(900, 250, (255, 226, 0, 255), r=40)
        for i, ln in enumerate(lines):
            t = text_img(ln, 72, fill=(0, 0, 0), stroke=0); p.alpha_composite(t, ((900 - t.width) // 2, 40 + i * 95))
        paste_c(base, shadowed(p), W / 2, FY + FH - 230, scale=a, rot=-2)
        paste_c(base, emoji("🏅", 150), W / 2 + 420, FY + FH - 350, scale=eob((lt - .2) / .3), rot=15)
    elif kind == "follow":
        b = pill(680, 160, (230, 20, 30, 255), r=80)
        t = text_img("FOLLOW", 84, stroke=0); b.alpha_composite(t, (70, (160 - t.height) // 2))
        e = emoji("🔔", 100); b.alpha_composite(e, (680 - e.width - 70, (160 - e.height) // 2))
        paste_c(base, shadowed(b), W / 2, FY + FH * .45, scale=a * (1 + .05 * math.sin(lt * 9)))

# ------------------------------------------------------------------- audio --
def ks_pluck(freq, dur, rng, bright=0.5):
    N = max(2, int(SR / freq)); L = int(dur * SR)
    buf = rng.uniform(-1, 1, N).astype(np.float32)
    out = np.zeros(L, np.float32); out[:N] = buf
    i = N
    while i < L:
        seg = out[i - N:i]
        nxt = 0.5 * (seg + np.roll(seg, -1)) * 0.996
        nxt = bright * seg * .996 + (1 - bright) * nxt
        m = min(N, L - i); out[i:i + m] = nxt[:m]; i += m
    return out * np.exp(-np.arange(L) / SR * 2.2)

def add(buf, clip, at, g=1.0):
    i = int(at * SR)
    if i < 0: clip = clip[-i:]; i = 0
    if i >= len(buf): return
    m = min(len(clip), len(buf) - i); buf[i:i + m] += clip[:m] * g

def lowpass(x, k): return np.convolve(x, np.ones(k) / k, mode="same").astype(np.float32)

def score(n, drops, quiet, beat_on):
    """Dark cinematic loop: sub drone + plucked ostinato + 808s/hats, cut out around blasts."""
    rng = np.random.default_rng(2); t = np.arange(n) / SR
    bpm = 140; spb = 60 / bpm; s16 = spb / 4
    roots = [73.42, 58.27, 65.41, 55.0]          # D, Bb, C, A
    arp = [0, 7, 12, 15, 12, 7, 10, 7]           # minor-ish
    mus = np.zeros(n, np.float32)
    # drone
    bar = spb * 4 * 2
    for k in range(int(t[-1] / bar) + 1):
        r = roots[k % 4]; a = int(k * bar * SR); b = min(n, int((k + 1) * bar * SR))
        if a >= n: break
        tt = t[a:b] - t[a]
        env = np.minimum(1, np.minimum(tt / .4, (tt[-1] - tt + 1e-3) / .4))
        mus[a:b] += env * (np.sin(2 * np.pi * r * tt) * .16 + np.sin(2 * np.pi * r * 2 * tt) * .04 + np.sin(2 * np.pi * r * 1.5 * tt) * .025)
    # pluck ostinato (8th notes)
    plucks = {}
    k = 0; tt0 = 0.0
    while tt0 < t[-1]:
        r = roots[int(tt0 / bar) % 4] * 4
        f = r * 2 ** (arp[k % 8] / 12)
        key = round(f, 1)
        if key not in plucks: plucks[key] = ks_pluck(f, .5, rng, .2) * .07
        add(mus, plucks[key], tt0)
        k += 1; tt0 += spb / 2
    # drums from beat_on
    kl = int(.5 * SR); kt = np.arange(kl) / SR
    kick = (np.sin(2 * np.pi * (45 + 120 * np.exp(-kt * 28)) * kt) * np.exp(-kt * 5)).astype(np.float32) * .55
    hl = int(.04 * SR); hat = np.diff(rng.standard_normal(hl + 1)).astype(np.float32) * np.exp(-np.arange(hl) / SR * 120) * .05
    sl = int(.25 * SR); snare = (lowpass(rng.standard_normal(sl), 3) * np.exp(-np.arange(sl) / SR * 18)).astype(np.float32) * .16
    drums = np.zeros(n, np.float32)
    pat_k = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    pat_s = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    step = 0; tt0 = beat_on
    while tt0 < t[-1]:
        st = step % 16
        if pat_k[st]: add(drums, kick, tt0)
        if pat_s[st]: add(drums, snare, tt0)
        if st % 2 == 0 or (step // 16) % 2 and st in (13, 15): add(drums, hat, tt0)
        step += 1; tt0 += s16
    gate = np.ones(n, np.float32)
    for a, b in quiet: gate[int(a * SR):int(b * SR)] = 0.0
    gate = lowpass(gate, 3000)
    dgate = np.ones(n, np.float32)
    for a, b in quiet: dgate[int(a * SR):int((b + 2.5) * SR)] = 0.0
    dgate = lowpass(dgate, 3000)
    return (mus * gate + drums * gate * dgate) * .9

def sfx():
    rng = np.random.default_rng(9); fx = {}
    L = int(4 * SR); tt = np.arange(L) / SR
    boom = lowpass(rng.standard_normal(L), 60) * 4 + np.sin(2 * np.pi * (30 + 70 * np.exp(-tt * 3)) * tt) * 1.1
    boom += lowpass(rng.standard_normal(L), 8) * np.exp(-tt * 6) * 1.2   # crack
    fx["boom"] = (boom * np.exp(-tt * 1.1) * np.minimum(1, tt / .005)).astype(np.float32) * .75
    L = int(3 * SR); tt = np.arange(L) / SR
    fx["ring"] = (np.sin(2 * np.pi * 3900 * tt) * np.exp(-tt * 1.0) * .045).astype(np.float32)
    L = int(.4 * SR); tt = np.arange(L) / SR
    w = np.diff(lowpass(rng.standard_normal(L + 1), 5)) * np.sin(np.pi * tt / tt[-1]) ** 2
    fx["whoosh"] = (w * .5).astype(np.float32)
    L = int(2.4 * SR); tt = np.arange(L) / SR
    r = rng.standard_normal(L) * (tt / tt[-1]) ** 2.5
    r = r + np.sin(2 * np.pi * (200 + 900 * (tt / tt[-1]) ** 2) * tt) * (tt / tt[-1]) ** 2 * .5
    fx["riser"] = (r * .22).astype(np.float32)
    L = int(.12 * SR); tt = np.arange(L) / SR
    fx["pop"] = (np.sin(2 * np.pi * (1100 - 600 * tt / tt[-1]) * tt) * np.exp(-tt * 40) * .22).astype(np.float32)
    L = int(.03 * SR); tt = np.arange(L) / SR
    fx["tick"] = (np.sin(2 * np.pi * 2800 * tt) * np.exp(-tt * 300) * .25).astype(np.float32)
    L = int(.7 * SR); tt = np.arange(L) / SR
    hb = np.sin(2 * np.pi * 50 * tt) * np.exp(-tt * 18); hb2 = np.roll(hb, int(.22 * SR)) * .7
    fx["heart"] = ((hb + hb2) * .7).astype(np.float32)
    L = int(1.0 * SR); tt = np.arange(L) / SR
    fx["hit"] = (np.sin(2 * np.pi * (40 + 60 * np.exp(-tt * 10)) * tt) * np.exp(-tt * 4) * .6
                 + lowpass(rng.standard_normal(L), 20) * np.exp(-tt * 8) * .8).astype(np.float32)
    return fx

# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="nuked_twice.mp4")
    ap.add_argument("--work", default=None)
    ap.add_argument("--preview", default=None)
    args = ap.parse_args()
    work = args.work or tempfile.mkdtemp(); os.makedirs(work, exist_ok=True)

    # ---- voice + timeline
    tl, t = [], 0.15
    for i, s in enumerate(SEG):
        t += s.get("pause", 0.12)
        cache = os.path.join(work, f"v{i:02d}.npz")
        if os.path.exists(cache) and open(cache + ".txt").read() == s["say"]:
            z = np.load(cache, allow_pickle=True); a, words = z["a"], [tuple(x) for x in z["w"].tolist()]
            words = [(w, float(s0), float(e0)) for w, s0, e0 in words]
        else:
            a, words = tts(s["say"], work, i)
            np.savez(cache, a=a, w=np.array(words, dtype=object)); open(cache + ".txt", "w").write(s["say"])
        d = len(a) / SR
        tl.append(dict(s=s, start=t, end=t + d, audio=a,
                       words=[(w, t + s0, t + e0) for w, s0, e0 in words]))
        t += d
    total = t + 1.2
    for i, x in enumerate(tl):
        x["vs"] = x["start"] - (x["s"].get("pause", .12) if i else x["start"])
        x["ve"] = tl[i + 1]["start"] - tl[i + 1]["s"].get("pause", .12) if i + 1 < len(tl) else total
        x["chunks"] = chunk_words(x["words"])
    print(f"duration {total:.1f}s")

    # ---- shot list (global frames)
    shots = []
    for x in tl:
        sh = x["s"]["shots"]; wsum = sum(s[3] for s in sh)
        f0 = int(round(x["vs"] * FPS)); f1 = int(round(x["ve"] * FPS)); acc = f0
        for k, (src, st, fx, wgt) in enumerate(sh):
            n = (f1 - acc) if k == len(sh) - 1 else int((f1 - f0) * wgt / wsum)
            shots.append(dict(src=src, st=st, fx=fx, f0=acc, n=n, seg=x, k=k))
            acc += n
    flashes = [x["vs"] for x in tl if x["s"].get("flash")]
    nframes = int(total * FPS)

    # ---- audio mix
    n = int(total * SR); fx = sfx()
    vo = np.zeros(n, np.float32)
    for x in tl: add(vo, x["audio"], x["start"])
    quiet = [(f - .25, f + 1.8) for f in flashes]
    for x in tl:
        if x["s"].get("riser"): quiet.append((x["vs"] + .2, x["ve"]))
        if x["s"].get("ominous"): quiet.append((x["vs"], x["ve"] - .2))
    beat_on = tl[3]["vs"]
    mus = score(n, flashes, quiet, beat_on)
    env = lowpass(np.abs(vo), 4410); duck = 1 - .5 * np.clip(env * 10, 0, 1)
    sfxb = np.zeros(n, np.float32)
    for f in flashes: add(sfxb, fx["boom"], f); add(sfxb, fx["ring"], f + .3)
    for s in shots[1:]:
        tt = s["f0"] / FPS
        if not any(abs(tt - f) < .2 for f in flashes): add(sfxb, fx["whoosh"], tt - .15, .8)
    for x in tl:
        d = x["ve"] - x["vs"]
        if x["s"].get("riser"): add(sfxb, fx["riser"], x["ve"] - 2.4)
        if x["s"].get("heartbeat"):
            for k in range(int(d / .8) + 1): add(sfxb, fx["heart"], x["vs"] + k * .8)
        if x["s"].get("tick"):
            for k in range(int(d / .5) + 1): add(sfxb, fx["tick"], x["vs"] + k * .5)
        if x["s"].get("ominous"): add(sfxb, fx["hit"], x["words"][-1][1])
        for kind, arg, at in x["s"]["ov"]:
            if kind in ("big", "tag", "emoji", "badge", "follow", "date", "loc"): add(sfxb, fx["pop"], x["vs"] + at * d)
            if kind == "big": add(sfxb, fx["hit"], x["vs"] + at * d, .6)
    mix = vo + mus * duck * .8 + sfxb
    mix = np.tanh(mix * 1.15) * .93
    apath = os.path.join(work, "mix.wav")
    with wave.open(apath, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())

    # ---- frames
    def compose(fi, raw, shot):
        ft = fi / FPS; x = shot["seg"]; lt = ft - x["vs"]; sd = x["ve"] - x["vs"]
        st = (fi - shot["f0"]) / max(1, shot["n"] - 1)
        OW, OH = raw.size
        # pan window (alternate direction per shot) + punch-in at the cut
        rnd = random.Random(shot["f0"])
        sx0, sy0 = rnd.uniform(0, OW - FW), rnd.uniform(0, OH - FH)
        sx1, sy1 = OW - FW - sx0, OH - FH - sy0
        px, py = sx0 + (sx1 - sx0) * st, sy0 + (sy1 - sy0) * st
        z = 1 + .09 * (1 - eo((fi - shot["f0"]) / 7))
        for f in flashes:
            if 0 <= ft - f < 1.3:
                k = (1 - (ft - f) / 1.3) * 34; px += math.sin(ft * 91) * k; py += math.cos(ft * 77) * k
        cw, chh = FW / z, FH / z
        px = min(max(px + (FW - cw) / 2, 0), OW - cw); py = min(max(py + (FH - chh) / 2, 0), OH - chh)
        fg = raw.resize((FW, FH), Image.BILINEAR, box=(px, py, px + cw, py + chh))
        if x["s"].get("blind") and lt > sd * .82:
            k = c01((lt - sd * .82) / (sd * .18))
            fg = Image.blend(fg.filter(ImageFilter.GaussianBlur(4 + 20 * k)), Image.new("RGB", fg.size, (0, 0, 0)), .65 * k)
        if x["s"].get("ominous"):
            fg = Image.blend(fg, Image.new("RGB", fg.size, (120, 0, 0)), .25 + .1 * math.sin(lt * 6))
        # blur-fill background
        bg = fg.resize((54, 62), Image.BILINEAR).filter(ImageFilter.GaussianBlur(2)).resize((W, H), Image.BILINEAR)
        bg = Image.blend(bg, Image.new("RGB", (W, H), (0, 0, 0)), .45).convert("RGBA")
        fgr = fg.convert("RGBA"); fgr.putalpha(FGMASK)
        bg.alpha_composite(fgr, (0, FY))
        base = bg
        # RGB split right after flashes
        for f in flashes:
            if 0 <= ft - f < .7:
                o = int((1 - (ft - f) / .7) * 22)
                r, g, b, a_ = base.split()
                base = Image.merge("RGBA", (ImageChops.offset(r, o, 0), g, ImageChops.offset(b, -o, 0), a_))
        base.alpha_composite(SCRATCH[(fi // 2) % 12]); base.alpha_composite(GRAIN[fi % 8])
        # flicker
        fl = .04 * math.sin(fi * 1.7) * math.sin(fi * .31)
        if fl > 0: base.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, int(fl * 255))))
        base.alpha_composite(VIGNETTE)
        # overlays
        state = {}
        for kind, arg, at in x["s"]["ov"]:
            draw_overlay(base, kind, arg, lt - at * sd, state)
        # flash
        for f in flashes:
            if -0.05 <= ft - f < 1.5:
                a = int(255 * (1 - max(0, ft - f) / 1.5) ** 1.4)
                base.alpha_composite(Image.new("RGBA", (W, H), (255, 252, 240, a)))
        # captions
        for idxs in x["chunks"]:
            s0, e0 = x["words"][idxs[0]][1], x["words"][idxs[-1]][2]
            nxt = x["chunks"][x["chunks"].index(idxs) + 1] if idxs is not x["chunks"][-1] else None
            hold = x["words"][nxt[0]][1] if nxt else e0 + .35
            if s0 - .03 <= ft < hold:
                cur = next((i for i in idxs if x["words"][i][1] <= ft < x["words"][i][2]), idxs[-1] if ft >= e0 else idxs[0])
                paste_c(base, caption(x["words"], idxs, cur), W / 2, FY + FH + 150, scale=.8 + .2 * eob((ft - s0) / .16))
                break
        # progress bar
        ImageDraw.Draw(base).rectangle((0, H - 10, int(W * ft / total), H), fill=(255, 226, 0, 255))
        return base.convert("RGB")

    if args.preview:
        ts = [float(v) for v in args.preview.split(",")]
        sheet = Image.new("RGB", (270 * 5, 480 * ((len(ts) + 4) // 5)))
        for k, tt in enumerate(ts):
            fi = int(tt * FPS); sh = next(s for s in shots if s["f0"] <= fi < s["f0"] + s["n"])
            raw = next(shot_frames(sh["src"], sh["st"] + (fi - sh["f0"]) / FPS, sh["fx"], 1))
            sheet.paste(compose(fi, raw, sh).resize((270, 480)), ((k % 5) * 270, (k // 5) * 480))
        sheet.save(args.out); return

    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-i", apath, "-c:v", "libx264", "-preset", "medium", "-crf", "21",
                           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
                           args.out], stdin=subprocess.PIPE)
    fi = 0
    for sh in shots:
        for raw in shot_frames(sh["src"], sh["st"], sh["fx"], sh["n"]):
            if fi >= nframes: break
            ff.stdin.write(compose(fi, raw, sh).tobytes()); fi += 1
            if fi % 150 == 0: print(f"frame {fi}/{nframes}", flush=True)
    ff.stdin.close(); ff.wait(); print("wrote", args.out)

if __name__ == "__main__":
    main()
