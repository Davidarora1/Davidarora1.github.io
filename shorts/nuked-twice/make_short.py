#!/usr/bin/env python3
"""
"He Got Nuked TWICE" — a YouTube Shorts generator (1080x1920, 30fps).

True story: Tsutomu Yamaguchi, the only person officially recognized by Japan
as surviving both the Hiroshima and Nagasaki atomic bombings.

Pipeline:
  1. Piper TTS voices each script segment.
  2. PIL draws every frame procedurally (scenes + TikTok-style word captions).
  3. numpy synthesizes music + SFX and mixes with the voice.
  4. ffmpeg muxes it all into an MP4.

Usage:
  pip install piper-tts pillow numpy
  # grab a piper voice (en_US-ryan-high) as voice.onnx + voice.onnx.json
  python3 make_short.py --voice voice.onnx --out nuked_twice.mp4
"""
import argparse, math, os, random, subprocess, wave, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS = 1080, 1920, 30
SR = 44100
INTER_BLACK = "/usr/share/fonts/opentype/inter/Inter-Black.otf"
INTER_BOLD = "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"
EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"

# ---------------------------------------------------------------- script ----
# caption text: words wrapped in *stars* get the red "emphasis" colour.
# tts: what the voice actually says (numbers spelled out etc.)
SEGMENTS = [
    dict(scene="hook1", cap="This man got hit by an *atomic* *bomb*.",
         tts="This man got hit by an atomic bomb.", pause=0.0),
    dict(scene="hook2", cap="Then three days later... he got hit by *ANOTHER* one.",
         tts="Then three days later... he got hit by another one.", pause=0.15),
    dict(scene="hook3", cap="And somehow... he *survived* *both.*",
         tts="And somehow... he survived both.", pause=0.1),
    dict(scene="map_hiro", cap="August 6th, 1945. Tsutomu Yamaguchi is in Hiroshima on a business trip.",
         tts="August sixth, nineteen forty five. Tsutomu Yamaguchi is in Hiroshima, on a business trip.", pause=0.3),
    dict(scene="sky", cap="It's his last day there. He's about to head home... when he spots a plane in the sky.",
         tts="It's his last day there. He's about to head home... when he spots a plane in the sky.", pause=0.15),
    dict(scene="flash1", cap="Then... everything turns *WHITE.*",
         tts="Then... everything turns white.", pause=0.35),
    dict(scene="injuries", cap="He's 3 kilometers away. His eardrums burst. He's burned. He's temporarily blind.",
         tts="He's three kilometers away. His eardrums burst. He's burned. He's temporarily blind.", pause=0.2),
    dict(scene="alive", cap="But he's *alive.*", tts="But... he's alive.", pause=0.1),
    dict(scene="map_route", cap="So what does he do? He gets on a train... back to his hometown.",
         tts="So what does he do? He gets on a train... back to his hometown.", pause=0.25),
    dict(scene="nagasaki", cap="*Nagasaki.*", tts="Nagasaki.", pause=0.2),
    dict(scene="office1", cap="August 9th. Covered in bandages, he goes back to work and tells his boss one bomb destroyed an entire city.",
         tts="August ninth. Covered in bandages, he goes back to work, and tells his boss that one bomb destroyed an entire city.", pause=0.35),
    dict(scene="office2", cap="His boss says he's *crazy.* And right at that moment...",
         tts="His boss says he's crazy. And right at that moment...", pause=0.15),
    dict(scene="flash2", cap="The room flashes *white.* *AGAIN.*",
         tts="The room flashes white. Again.", pause=0.4),
    dict(scene="same", cap="Same flash. Same distance. 3 kilometers. *AGAIN.*",
         tts="Same flash. Same distance. Three kilometers. Again.", pause=0.3),
    dict(scene="alive2", cap="And he *survived.* *AGAIN.*", tts="And he survived. Again.", pause=0.2),
    dict(scene="legacy", cap="He lived to be 93. He's the only person Japan officially recognized as surviving *both* bombs.",
         tts="He lived to be ninety three. He's the only person Japan officially recognized as surviving both bombs.", pause=0.3),
    dict(scene="outro", cap="Bro said \"not today\" *TWICE.* Follow for more.",
         tts="Bro said, not today... twice. Follow for more.", pause=0.25),
]

# ------------------------------------------------------------ helpers -------
_font_cache = {}
def font(path, size):
    k = (path, size)
    if k not in _font_cache:
        _font_cache[k] = ImageFont.truetype(path, size)
    return _font_cache[k]

_emoji_cache = {}
def emoji(ch, size):
    k = (ch, size)
    if k not in _emoji_cache:
        f = ImageFont.truetype(EMOJI, 109)
        im = Image.new("RGBA", (200, 160))
        ImageDraw.Draw(im).text((4, 4), ch, font=f, embedded_color=True)
        im = im.crop(im.getbbox())
        s = size / max(im.size)
        _emoji_cache[k] = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    return _emoji_cache[k]

def paste_c(base, im, cx, cy, scale=1.0, alpha=1.0, rot=0):
    if scale <= 0.01 or alpha <= 0.01:
        return
    if scale != 1.0:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BILINEAR)
    if rot:
        im = im.rotate(rot, resample=Image.BICUBIC, expand=True)
    if alpha < 1.0:
        a = im.getchannel("A").point(lambda v: int(v * alpha))
        im = im.copy(); im.putalpha(a)
    base.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))

def text_img(txt, size, fill=(255, 255, 255), stroke=10, path=INTER_BLACK, stroke_fill=(0, 0, 0)):
    f = font(path, size)
    l, t, r, b = f.getbbox(txt, stroke_width=stroke)
    im = Image.new("RGBA", (r - l + 8, b - t + 8))
    ImageDraw.Draw(im).text((4 - l, 4 - t), txt, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
    return im

def ease_out_back(x):
    x = min(max(x, 0), 1)
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2

def ease_out(x):
    x = min(max(x, 0), 1)
    return 1 - (1 - x) ** 3

def clamp01(x):
    return min(max(x, 0.0), 1.0)

def pill(w, h, fill, outline=None, r=None, width=6):
    im = Image.new("RGBA", (w, h))
    ImageDraw.Draw(im).rounded_rectangle((0, 0, w - 1, h - 1), r or h // 2, fill=fill, outline=outline, width=width)
    return im

def card(label, em, color):
    """A pop-in stat card: emoji + label."""
    c = pill(860, 170, (18, 18, 24, 235), outline=color, r=40, width=7)
    e = emoji(em, 110)
    c.alpha_composite(e, (40, (170 - e.height) // 2))
    t = text_img(label, 72, stroke=0)
    c.alpha_composite(t, (180, (170 - t.height) // 2))
    return c

# ------------------------------------------------------- static assets ------
def vgrad(top, bot, h=H, w=W):
    a = np.linspace(0, 1, h)[:, None, None]
    g = np.array(top)[None, None, :] * (1 - a) + np.array(bot)[None, None, :] * a
    return np.repeat(g, w, axis=1).astype(np.uint8)

BG_DARK = Image.fromarray(vgrad((14, 10, 30), (2, 2, 6))).convert("RGBA")
BG_RED = Image.fromarray(vgrad((60, 6, 10), (6, 0, 2))).convert("RGBA")
BG_SKY = Image.fromarray(vgrad((90, 160, 230), (230, 220, 200))).convert("RGBA")
BG_ORANGE = Image.fromarray(vgrad((70, 25, 10), (220, 110, 40))).convert("RGBA")

def make_vignette():
    y, x = np.mgrid[0:H, 0:W]
    d = np.sqrt(((x - W / 2) / (W * 0.75)) ** 2 + ((y - H / 2) / (H * 0.7)) ** 2)
    a = np.clip((d - 0.45) * 1.6, 0, 1) * 200
    im = np.zeros((H, W, 4), np.uint8); im[..., 3] = a.astype(np.uint8)
    return Image.fromarray(im)
VIGNETTE = make_vignette()

def make_city():
    rnd = random.Random(7)
    im = Image.new("RGBA", (W, 520))
    d = ImageDraw.Draw(im)
    x = -20
    while x < W:
        bw = rnd.randint(50, 130); bh = rnd.randint(120, 380)
        d.rectangle((x, 520 - bh, x + bw, 520), fill=(25, 25, 40, 255))
        for wy in range(520 - bh + 20, 500, 40):
            for wx in range(x + 12, x + bw - 18, 28):
                if rnd.random() < 0.35:
                    d.rectangle((wx, wy, wx + 12, wy + 18), fill=(255, 210, 120, 200))
        x += bw + rnd.randint(0, 8)
    return im
CITY = make_city()

def make_cloud():
    cw, ch = 1000, 1150
    P = 160
    im = Image.new("RGBA", (cw, ch))
    d = ImageDraw.Draw(im)
    rnd = random.Random(3)
    # stem
    for i in range(60):
        y = 450 + i * 10
        w = 70 + (i / 60) ** 2.2 * 140
        c = (int(150 - i), int(90 - i * 0.6), int(60 - i * 0.4), 255)
        d.ellipse((cw / 2 - w, y - 30, cw / 2 + w, y + 30), fill=c)
    # base skirt
    for i in range(40):
        x = rnd.uniform(150, 850); y = rnd.uniform(1000, 1100); r = rnd.uniform(60, 120)
        d.ellipse((x - r, y - r * 0.6, x + r, y + r * 0.6), fill=(120, 80, 60, 255))
    # cap
    for i in range(140):
        a = rnd.uniform(0, math.tau); rr = rnd.uniform(0, 1) ** 0.6
        x = cw / 2 + math.cos(a) * rr * 330; y = 300 + math.sin(a) * rr * 190
        r = rnd.uniform(70, 140)
        heat = 1 - rr
        c = (int(170 + 85 * heat), int(90 + 110 * heat), int(50 + 60 * heat), 255)
        d.ellipse((x - r, y - r, x + r, y + r), fill=c)
    # ring
    d.ellipse((cw / 2 - 260, 560, cw / 2 + 260, 640), outline=(220, 170, 120, 255), width=26)
    im = im.filter(ImageFilter.GaussianBlur(9))
    big = Image.new("RGBA", (cw + 2 * P, ch + 2 * P)); big.alpha_composite(im, (P, P))
    glow = big.filter(ImageFilter.GaussianBlur(60))
    out = Image.new("RGBA", big.size); out.alpha_composite(glow); out.alpha_composite(glow); out.alpha_composite(big)
    return out
CLOUD = make_cloud()

def make_plane():
    im = Image.new("RGBA", (420, 200))
    d = ImageDraw.Draw(im)
    d.ellipse((20, 85, 400, 115), fill=(30, 30, 40, 255))                     # fuselage
    d.polygon([(170, 100), (250, 100), (215, 10), (190, 10)], fill=(30, 30, 40, 255))   # wing up
    d.polygon([(170, 100), (250, 100), (215, 190), (190, 190)], fill=(30, 30, 40, 255)) # wing down
    d.polygon([(20, 100), (70, 100), (40, 55), (20, 55)], fill=(30, 30, 40, 255))      # tail
    return im.resize((210, 100), Image.LANCZOS)
PLANE = make_plane()

# ---- stylized map of southern Japan (rough lon/lat outlines) ----
KYUSHU = [(129.7, 33.35), (130.05, 33.85), (130.55, 33.95), (130.95, 33.92), (131.05, 33.6),
          (131.7, 33.35), (131.6, 32.9), (131.9, 32.5), (131.45, 31.4), (131.1, 31.25), (130.7, 31.0),
          (130.2, 31.3), (130.15, 31.75), (130.35, 32.1), (130.6, 32.7), (130.2, 32.65),
          (129.95, 32.55), (129.7, 32.75), (129.85, 33.05), (129.55, 33.2)]
HONSHU = [(130.9, 33.95), (130.95, 34.35), (131.4, 34.45), (131.8, 34.65), (132.3, 35.1), (132.9, 35.5),
          (133.4, 35.55), (134.3, 35.6), (135.2, 35.7), (135.6, 35.5), (135.7, 34.6), (135.2, 34.65),
          (134.7, 34.75), (134.2, 34.6), (133.6, 34.45), (133.0, 34.35), (132.5, 34.3), (132.2, 33.95),
          (131.8, 34.0), (131.3, 33.95)]
SHIKOKU = [(132.05, 33.3), (132.4, 33.75), (132.75, 33.95), (133.2, 33.95), (133.6, 34.25), (134.2, 34.3),
           (134.65, 34.2), (134.75, 33.85), (134.3, 33.6), (133.9, 33.4), (133.4, 33.35), (133.0, 32.75),
           (132.5, 32.85)]
CITIES = {"HIROSHIMA": (132.46, 34.39), "NAGASAKI": (129.87, 32.75)}

def lonlat(lon, lat):
    return (80 + (lon - 129.3) * 142, 330 + (36.0 - lat) * 172)

def make_map():
    im = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(im)
    for x in range(0, W, 60):
        d.line((x, 0, x, H), fill=(60, 80, 140, 40), width=1)
    for y in range(0, H, 60):
        d.line((0, y, W, y), fill=(60, 80, 140, 40), width=1)
    land = Image.new("RGBA", (W, H)); ld = ImageDraw.Draw(land)
    for poly in (KYUSHU, HONSHU, SHIKOKU):
        pts = [lonlat(*p) for p in poly]
        ld.polygon(pts, fill=(40, 52, 90, 255))
        ld.line(pts + [pts[0]], fill=(120, 170, 255, 255), width=5, joint="curve")
    glow = land.filter(ImageFilter.GaussianBlur(18))
    im.alpha_composite(glow); im.alpha_composite(land)
    t = text_img("JAPAN", 52, fill=(150, 180, 255), stroke=0, path=INTER_BOLD)
    im.alpha_composite(t, (720, 1120))
    return im
MAP = make_map()

def route_points(n=120):
    (x0, y0), (x1, y1) = lonlat(*CITIES["HIROSHIMA"]), lonlat(*CITIES["NAGASAKI"])
    cx, cy = (x0 + x1) / 2 + 40, (y0 + y1) / 2 - 170
    return [((1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t * t * x1,
             (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t * t * y1) for t in np.linspace(0, 1, n)]
ROUTE = route_points()

def pin(base, name, t, color=(255, 60, 60), pulse=True, big=False):
    x, y = lonlat(*CITIES[name])
    d = ImageDraw.Draw(base)
    if pulse:
        for k in range(2):
            ph = (t * 1.2 + k * 0.5) % 1
            r = 20 + ph * 90
            d.ellipse((x - r, y - r, x + r, y + r), outline=color + (int(255 * (1 - ph)),), width=5)
    r = 22 if big else 16
    d.ellipse((x - r, y - r, x + r, y + r), fill=color + (255,), outline=(255, 255, 255, 255), width=5)
    lab = text_img(name, 58 if big else 46, stroke=8)
    if name == "NAGASAKI":
        base.alpha_composite(lab, (int(x - 30), int(y + 36)))
    else:
        base.alpha_composite(lab, (int(x - lab.width / 2 + 60), int(y + 36)))

# --------------------------------------------------------------- scenes -----
def scene_flash_cloud(base, t, dur, again=False):
    base.alpha_composite(BG_ORANGE)
    g = ease_out(t / 3.0)
    paste_c(base, CLOUD, W / 2, 1180 - 480 * (0.4 + 0.6 * g), scale=0.4 + 0.55 * g)
    base.alpha_composite(CITY, (0, 1400))
    if again:
        b = text_img("AGAIN.", 200, fill=(255, 40, 40), stroke=14)
        paste_c(base, b, W / 2, 330, scale=ease_out_back((t - 0.9) / 0.35) if t > 0.9 else 0, rot=-6)

def s_hook1(base, t, dur):
    scene_flash_cloud(base, t, dur)
    tag = text_img("NUKED.", 220, fill=(255, 230, 0), stroke=16)
    paste_c(base, tag, W / 2, 330, scale=ease_out_back((t - 0.5) / 0.35) if t > 0.5 else 0, rot=-5)

def s_hook2(base, t, dur):
    base.alpha_composite(BG_RED)
    for i, cx in enumerate((300, 780)):
        appear = ease_out_back((t - i * 1.1) / 0.4)
        if t > i * 1.1:
            paste_c(base, CLOUD, cx, 980, scale=0.42 * appear)
            lab = text_img(["AUG 6", "AUG 9"][i], 70, stroke=8)
            paste_c(base, lab, cx, 1330, scale=appear)
    x2 = text_img("x2", 330, fill=(255, 40, 40), stroke=18)
    paste_c(base, x2, W / 2, 380, scale=ease_out_back((t - dur * 0.62) / 0.3) if t > dur * 0.62 else 0, rot=8)

def s_hook3(base, t, dur):
    base.alpha_composite(BG_DARK)
    s = 1 + 0.05 * math.sin(t * 9)
    paste_c(base, emoji("🤯", 420), W / 2, 760, scale=ease_out_back(t / 0.4) * s)
    h = text_img("HOW?!", 190, fill=(255, 230, 0), stroke=14)
    paste_c(base, h, W / 2, 330, scale=ease_out_back((t - 0.4) / 0.3) if t > 0.4 else 0)

def s_map_hiro(base, t, dur):
    base.alpha_composite(BG_DARK); base.alpha_composite(MAP)
    if t > 1.6:
        pin(base, "HIROSHIMA", t)
    date = pill(560, 120, (255, 230, 0, 255), r=30)
    dt = text_img("AUG 6, 1945", 70, fill=(0, 0, 0), stroke=0)
    date.alpha_composite(dt, ((560 - dt.width) // 2, (120 - dt.height) // 2))
    paste_c(base, date, W / 2, 170, scale=ease_out_back(t / 0.35), rot=-3)
    if t > 3.2:
        paste_c(base, emoji("💼", 130), 860, 1300, scale=ease_out_back((t - 3.2) / 0.3))

def s_sky(base, t, dur):
    base.alpha_composite(BG_SKY)
    sun = Image.new("RGBA", (300, 300)); ImageDraw.Draw(sun).ellipse((50, 50, 250, 250), fill=(255, 250, 220, 255))
    paste_c(base, sun.filter(ImageFilter.GaussianBlur(20)), 820, 420)
    base.alpha_composite(CITY, (0, 1400))
    p = clamp01((t - dur * 0.55) / (dur * 0.45))
    if t > dur * 0.5:
        paste_c(base, PLANE, -150 + p * 900, 520 - p * 60, scale=0.9)
    if t > dur * 0.7:
        paste_c(base, emoji("👀", 170), W / 2, 1150, scale=ease_out_back((t - dur * 0.7) / 0.3))

def s_flash1(base, t, dur):
    scene_flash_cloud(base, t + 0.5, dur)

def s_injuries(base, t, dur):
    base.alpha_composite(BG_DARK)
    items = [("3 KM AWAY", "📍", (255, 230, 0)), ("EARDRUMS BURST", "👂", (255, 140, 0)),
             ("BURNED", "🔥", (255, 70, 40)), ("TEMPORARILY BLIND", "😵", (200, 90, 255))]
    starts = [0.05, 0.3, 0.52, 0.7]
    for i, (lab, em, col) in enumerate(items):
        st = starts[i] * dur
        if t > st:
            paste_c(base, card(lab, em, col), W / 2, 330 + i * 215, scale=ease_out_back((t - st) / 0.3))

def ecg(base, t, color=(60, 255, 120)):
    pts = []
    for x in range(0, W, 6):
        ph = ((x / W) * 2.5 - t * 0.9) % 1
        if 0.40 < ph < 0.44: y = -60 * (ph - 0.40) / 0.04
        elif 0.44 < ph < 0.48: y = -60 + 280 * (ph - 0.44) / 0.04
        elif 0.48 < ph < 0.53: y = 220 - 260 * (ph - 0.48) / 0.05
        elif 0.53 < ph < 0.56: y = -40 + 40 * (ph - 0.53) / 0.03
        else: y = 0
        pts.append((x, 900 - y))
    layer = Image.new("RGBA", (W, H)); ImageDraw.Draw(layer).line(pts, fill=color + (255,), width=10, joint="curve")
    base.alpha_composite(layer.filter(ImageFilter.GaussianBlur(12))); base.alpha_composite(layer)

def s_alive(base, t, dur, again=False):
    base.alpha_composite(BG_DARK)
    ecg(base, t)
    a = text_img("ALIVE." if not again else "SURVIVED. AGAIN.", 190 if not again else 120, fill=(60, 255, 120), stroke=14)
    paste_c(base, a, W / 2, 400, scale=ease_out_back((t - 0.3) / 0.35) * (1 + 0.04 * math.sin(t * 8)) if t > 0.3 else 0)

def s_map_route(base, t, dur):
    base.alpha_composite(BG_DARK); base.alpha_composite(MAP)
    pin(base, "HIROSHIMA", t, color=(255, 140, 0), pulse=False)
    p = clamp01((t - dur * 0.3) / (dur * 0.6))
    n = max(2, int(p * len(ROUTE)))
    layer = Image.new("RGBA", (W, H)); ld = ImageDraw.Draw(layer)
    seg = ROUTE[:n]
    for i in range(0, len(seg) - 1, 2):
        ld.line((seg[i], seg[i + 1]), fill=(255, 230, 0, 255), width=10)
    base.alpha_composite(layer)
    if p > 0:
        x, y = seg[-1]
        paste_c(base, emoji("🚆", 120), x, y - 50)
    if p >= 1:
        pin(base, "NAGASAKI", t, pulse=False)

def s_nagasaki(base, t, dur):
    base.alpha_composite(BG_DARK); base.alpha_composite(MAP)
    tint = Image.new("RGBA", (W, H), (120, 0, 0, int(90 + 40 * math.sin(t * 6))))
    base.alpha_composite(tint)
    pin(base, "HIROSHIMA", 0, color=(255, 140, 0), pulse=False)
    pin(base, "NAGASAKI", t, big=True)
    paste_c(base, emoji("💀", 200), W / 2, 1260, scale=ease_out_back((t - 0.3) / 0.3) if t > 0.3 else 0)

def bubble(txt_lines, w=820, flip=False):
    hl = 82
    b = Image.new("RGBA", (w, 70 + hl * len(txt_lines) + 60))
    d = ImageDraw.Draw(b)
    d.rounded_rectangle((0, 0, w - 1, b.height - 61), 40, fill=(255, 255, 255, 255))
    tail = [(120, b.height - 62), (200, b.height - 62), (110, b.height - 1)]
    if flip: tail = [(w - x, y) for x, y in tail]
    d.polygon(tail, fill=(255, 255, 255, 255))
    for i, ln in enumerate(txt_lines):
        tx = text_img(ln, 64, fill=(10, 10, 10), stroke=0)
        b.alpha_composite(tx, ((w - tx.width) // 2, 40 + i * hl))
    return b

def office_base(base, t):
    if not hasattr(office_base, "bg"):
        bg = Image.fromarray(vgrad((40, 36, 50), (16, 14, 20))).convert("RGBA")
        d = ImageDraw.Draw(bg)
        d.rectangle((140, 220, 940, 760), fill=(150, 200, 255, 255), outline=(70, 60, 60, 255), width=24)
        d.line((540, 220, 540, 760), fill=(70, 60, 60, 255), width=18)
        d.line((140, 490, 940, 490), fill=(70, 60, 60, 255), width=18)
        d.rectangle((0, 1300, W, 1360), fill=(90, 60, 40, 255))
        office_base.bg = bg

def s_office1(base, t, dur):
    office_base(base, t); base.alpha_composite(office_base.bg)
    paste_c(base, emoji("🤕", 300), 280, 1150, scale=ease_out_back(t / 0.4))
    paste_c(base, emoji("🧐", 300), 800, 1150, scale=ease_out_back((t - 0.3) / 0.4) if t > 0.3 else 0)
    if t > dur * 0.55:
        b = bubble(["ONE bomb wiped out", "the WHOLE city!"])
        paste_c(base, b, 470, 820, scale=ease_out_back((t - dur * 0.55) / 0.3))

def s_office2(base, t, dur):
    office_base(base, t); base.alpha_composite(office_base.bg)
    paste_c(base, emoji("🤕", 300), 280, 1150)
    paste_c(base, emoji("🤨", 300), 800, 1150)
    if t < dur * 0.6:
        b = bubble(["You're crazy."], w=600, flip=True)
        paste_c(base, b, 660, 880, scale=ease_out_back(t / 0.3))
    # the window starts glowing...
    g = clamp01((t - dur * 0.6) / (dur * 0.4))
    if g > 0:
        base.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 240, int(g * 170))))

def s_flash2(base, t, dur):
    scene_flash_cloud(base, t + 0.5, dur, again=True)

def s_same(base, t, dur):
    base.alpha_composite(BG_RED)
    rows = [("HIROSHIMA", "AUG 6"), ("NAGASAKI", "AUG 9")]
    for i, (c, d) in enumerate(rows):
        st = 0.05 + i * 0.25 * dur
        if t > st:
            cd = pill(900, 260, (15, 10, 12, 240), outline=(255, 60, 60), r=40, width=8)
            a = text_img(c, 76, stroke=0); b = text_img(d + "  •  3 KM", 60, fill=(255, 230, 0), stroke=0)
            cd.alpha_composite(a, (50, 45)); cd.alpha_composite(b, (50, 150))
            e = emoji("☢️", 140); cd.alpha_composite(e, (900 - e.width - 40, (260 - e.height) // 2))
            paste_c(base, cd, W / 2, 380 + i * 320, scale=ease_out_back((t - st) / 0.3))
    if t > dur * 0.72:
        s = text_img("SAME. EXACT.", 120, fill=(255, 230, 0), stroke=12)
        paste_c(base, s, W / 2, 1100, scale=ease_out_back((t - dur * 0.72) / 0.3), rot=-4)

def s_alive2(base, t, dur):
    s_alive(base, t, dur, again=True)

def s_legacy(base, t, dur):
    base.alpha_composite(BG_DARK)
    n = int(93 * ease_out(t / (dur * 0.35)))
    num = text_img(str(n), 360, fill=(255, 230, 0), stroke=18)
    paste_c(base, num, W / 2, 420)
    yrs = text_img("YEARS OLD", 90, stroke=10)
    paste_c(base, yrs, W / 2, 660)
    if t > dur * 0.4:
        bd = pill(940, 300, (255, 230, 0, 255), r=50)
        a = text_img("ONLY OFFICIAL", 78, fill=(0, 0, 0), stroke=0)
        b = text_img("DOUBLE SURVIVOR", 78, fill=(0, 0, 0), stroke=0)
        bd.alpha_composite(a, ((940 - a.width) // 2, 55)); bd.alpha_composite(b, ((940 - b.width) // 2, 160))
        paste_c(base, bd, W / 2, 1030, scale=ease_out_back((t - dur * 0.4) / 0.35), rot=-2)
        paste_c(base, emoji("🏅", 160), 930, 870, scale=ease_out_back((t - dur * 0.5) / 0.3) if t > dur * 0.5 else 0, rot=15)

def s_outro(base, t, dur):
    base.alpha_composite(BG_DARK)
    paste_c(base, emoji("💀", 330), W / 2, 520, scale=ease_out_back(t / 0.35) * (1 + 0.05 * math.sin(t * 10)))
    a = text_img("NOT TODAY x2", 140, fill=(255, 230, 0), stroke=14)
    paste_c(base, a, W / 2, 870, scale=ease_out_back((t - 0.3) / 0.3) if t > 0.3 else 0, rot=-3)
    if t > dur * 0.55:
        btn = pill(640, 150, (230, 20, 30, 255), r=75)
        tx = text_img("FOLLOW 🔔".replace(" 🔔", ""), 80, stroke=0)
        btn.alpha_composite(tx, (60, (150 - tx.height) // 2))
        e = emoji("🔔", 95); btn.alpha_composite(e, (640 - e.width - 60, (150 - e.height) // 2))
        pulse = 1 + 0.05 * math.sin(t * 9)
        paste_c(base, btn, W / 2, 1100, scale=ease_out_back((t - dur * 0.55) / 0.3) * pulse)

SCENES = {k[2:]: v for k, v in globals().items() if k.startswith("s_")}

# ------------------------------------------------------------ captions -----
def parse_words(cap):
    out = []
    for w in cap.split():
        emph = w.startswith("*")
        clean = w.replace("*", "")
        out.append((clean, emph))
    return out

def word_timings(words, start, dur):
    """Approximate per-word timing by syllable-ish weight + punctuation pauses."""
    wts = []
    for w, _ in words:
        core = w.strip('.,?!"')
        wt = 0.35 + len(core) * 0.085
        if core.isdigit():
            wt += 0.4 * len(core)
        if w.endswith((".", "?", "!", '..."', "...")): wt += 0.35
        elif w.endswith(","): wt += 0.18
        wts.append(wt)
    tot = sum(wts)
    t = start; out = []
    for wt in wts:
        d = dur * wt / tot
        out.append((t, t + d)); t += d
    return out

def chunk(words, maxw=3, maxc=16):
    chunks, cur = [], []
    for i, (w, e) in enumerate(words):
        cur.append(i)
        n = sum(len(words[j][0]) for j in cur)
        if len(cur) >= maxw or n >= maxc or w.endswith((".", "?", "!", ",", "...")):
            chunks.append(cur); cur = []
    if cur: chunks.append(cur)
    return chunks

_cap_cache = {}
def caption_img(words, idxs, active):
    k = (tuple(words[i][0] for i in idxs), tuple(idxs), active, id(words))
    if k in _cap_cache: return _cap_cache[k]
    size = 108
    f = font(INTER_BLACK, size)
    parts = []
    for i in idxs:
        w, e = words[i]
        w = w.upper()
        col = (255, 255, 255)
        if e: col = (255, 60, 60)
        if i == active: col = (255, 230, 0) if not e else (255, 60, 60)
        parts.append((w, col, i == active))
    space = 28
    widths = [f.getbbox(p[0], stroke_width=12)[2] for p in parts]
    total = sum(widths) + space * (len(parts) - 1)
    scale = min(1.0, 980 / total)
    im = Image.new("RGBA", (int(total + 40), 200))
    d = ImageDraw.Draw(im)
    x = 20
    for (w, col, act), wd in zip(parts, widths):
        d.text((x, 30), w, font=f, fill=col, stroke_width=12, stroke_fill=(0, 0, 0))
        x += wd + space
    if scale < 1:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    _cap_cache[k] = im
    return im

# --------------------------------------------------------------- audio -----
def tts(text, voice, path, length_scale=0.9):
    raw = path + ".raw.wav"
    subprocess.run(["piper", "-m", voice, "--length_scale", str(length_scale), "-f", raw],
                   input=text.encode(), check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-ar", str(SR), "-ac", "1",
                    "-af", "highpass=f=80,acompressor=threshold=-18dB:ratio=3:attack=5:release=60,"
                           "silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
                           "silenceremove=start_periods=1:start_threshold=-45dB,areverse",
                    path], check=True)
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768

def synth_music(n, flash_times, beat_from):
    t = np.arange(n) / SR
    rng = np.random.default_rng(1)
    # dark pad: minor chord drones (A minor -> F -> G -> E), slow tremolo
    chords = [(110, 130.81, 164.81), (87.31, 110, 130.81), (98, 123.47, 146.83), (82.41, 103.83, 123.47)]
    pad = np.zeros(n, np.float32)
    bar = 4 * 60 / 96
    for ci in range(int(t[-1] / bar) + 1):
        a, b = int(ci * bar * SR), min(n, int((ci + 1) * bar * SR))
        if a >= n: break
        tt = t[a:b]
        env = np.minimum(1, np.minimum((tt - tt[0]) / 0.3, (tt[-1] - tt + 1e-3) / 0.3))
        for f in chords[ci % 4]:
            pad[a:b] += env * (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * 2 * f * tt + 0.5)) * 0.05
    pad *= 0.75 + 0.25 * np.sin(2 * np.pi * 0.2 * t)
    # pulse: kick + hat at 96bpm from beat_from on
    beat = np.zeros(n, np.float32)
    spb = 60 / 96
    kick_len = int(0.25 * SR); kt = np.arange(kick_len) / SR
    kick = np.sin(2 * np.pi * (50 + 90 * np.exp(-kt * 30)) * kt) * np.exp(-kt * 12) * 0.5
    hat_len = int(0.05 * SR)
    hat = rng.standard_normal(hat_len) * np.exp(-np.arange(hat_len) / SR * 90) * 0.06
    hat = np.diff(hat, prepend=0)
    k = beat_from
    while k < t[-1]:
        i = int(k * SR)
        beat[i:i + kick_len] += kick[:n - i]
        j = int((k + spb / 2) * SR)
        if j < n: beat[j:j + hat_len] += hat[:n - j]
        k += spb
    # silence the beat briefly around flashes for drama
    gate = np.ones(n, np.float32)
    for ft in flash_times:
        a, b = int((ft - 0.5) * SR), int((ft + 1.6) * SR)
        gate[max(0, a):min(n, b)] = 0
    gate = np.convolve(gate, np.ones(2000) / 2000, mode="same")
    return pad + beat * gate

def sfx_boom(rng):
    L = int(3.0 * SR); tt = np.arange(L) / SR
    noise = rng.standard_normal(L)
    # crude lowpass via cumulative smoothing
    lp = np.convolve(noise, np.ones(80) / 80, mode="same")
    sub = np.sin(2 * np.pi * (35 + 60 * np.exp(-tt * 4)) * tt)
    env = np.exp(-tt * 1.4) * np.minimum(1, tt / 0.01)
    return ((lp * 3.0 + sub * 0.9) * env * 0.8).astype(np.float32)

def sfx_whoosh(rng):
    L = int(0.35 * SR); tt = np.arange(L) / SR
    n = rng.standard_normal(L)
    n = np.diff(np.convolve(n, np.ones(6) / 6, mode="same"), prepend=0)
    env = np.sin(np.pi * tt / tt[-1]) ** 2
    return (n * env * 0.12).astype(np.float32)

def sfx_pop(rng):
    L = int(0.12 * SR); tt = np.arange(L) / SR
    return (np.sin(2 * np.pi * (900 - 500 * tt / tt[-1]) * tt) * np.exp(-tt * 35) * 0.18).astype(np.float32)

def sfx_ring(rng):
    L = int(2.5 * SR); tt = np.arange(L) / SR
    return (np.sin(2 * np.pi * 4200 * tt) * np.exp(-tt * 1.5) * 0.03).astype(np.float32)

def add(buf, clip, at):
    i = int(at * SR)
    if i >= len(buf): return
    m = min(len(clip), len(buf) - i)
    buf[i:i + m] += clip[:m]

# ---------------------------------------------------------------- main -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True)
    ap.add_argument("--out", default="nuked_twice.mp4")
    ap.add_argument("--work", default=None)
    ap.add_argument("--preview", default=None, help="comma-separated times: render a contact sheet PNG instead")
    args = ap.parse_args()
    work = args.work or tempfile.mkdtemp()
    os.makedirs(work, exist_ok=True)

    # 1. voice
    timeline, voice_clips, t = [], [], 0.25
    for i, s in enumerate(SEGMENTS):
        t += s["pause"]
        clip = tts(s["tts"], args.voice, os.path.join(work, f"seg{i:02d}.wav"))
        d = len(clip) / SR
        words = parse_words(s["cap"])
        timeline.append(dict(seg=s, start=t, end=t + d, words=words,
                             wt=word_timings(words, t, d), chunks=chunk(words)))
        voice_clips.append((t, clip))
        t += d + 0.08
    total = t + 0.9
    # scene spans: each scene runs until next segment starts
    for i, tl in enumerate(timeline):
        tl["scene_start"] = tl["start"] - (SEGMENTS[i]["pause"] if i else tl["start"])
        tl["scene_end"] = timeline[i + 1]["start"] - SEGMENTS[i + 1]["pause"] if i + 1 < len(timeline) else total
    print(f"total duration {total:.1f}s")

    flash_times = [tl["scene_start"] for tl in timeline if tl["seg"]["scene"] in ("hook1", "flash1", "flash2")]

    # 2. audio mix
    n = int(total * SR)
    rng = np.random.default_rng(5)
    vo = np.zeros(n, np.float32)
    for st, c in voice_clips: add(vo, c, st)
    music = synth_music(n, flash_times, beat_from=timeline[3]["scene_start"])
    # duck music under voice
    envv = np.convolve(np.abs(vo), np.ones(4410) / 4410, mode="same")
    duck = 1 - 0.55 * np.clip(envv * 12, 0, 1)
    fx = np.zeros(n, np.float32)
    boom, wh, pop, ring = sfx_boom(rng), sfx_whoosh(rng), sfx_pop(rng), sfx_ring(rng)
    for ft in flash_times:
        add(fx, boom, ft); add(fx, ring, ft + 0.2)
    for tl in timeline[1:]:
        if tl["seg"]["scene"] not in ("flash1", "flash2"):
            add(fx, wh, tl["scene_start"] - 0.12)
    for tl in timeline:
        if tl["seg"]["scene"] in ("injuries",):
            d = tl["scene_end"] - tl["scene_start"]
            for p in (0.05, 0.3, 0.52, 0.7): add(fx, pop, tl["scene_start"] + p * d)
        if tl["seg"]["scene"] in ("hook2",):
            add(fx, pop, tl["scene_start"]); add(fx, pop, tl["scene_start"] + 1.1)
    mix = vo * 1.0 + music * duck * 0.9 + fx
    mix = np.tanh(mix * 1.1) * 0.92
    audio_path = os.path.join(work, "mix.wav")
    with wave.open(audio_path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())

    # 3. frames
    grain = [Image.fromarray(np.dstack([np.random.default_rng(i).integers(0, 255, (H // 4, W // 4), dtype=np.uint8)] * 3 +
                                       [np.full((H // 4, W // 4), 14, np.uint8)])).resize((W, H)) for i in range(6)]
    nframes = int(total * FPS)

    def render(fi):
        ft = fi / FPS
        tl = next((x for x in timeline if x["scene_start"] <= ft < x["scene_end"]), timeline[-1])
        lt = ft - tl["scene_start"]; dur = tl["scene_end"] - tl["scene_start"]
        base = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        SCENES[tl["seg"]["scene"]](base, lt, dur)
        # camera: punch-in at cut + slow push + shake after flashes
        z = 1.0 + 0.07 * (1 - ease_out(lt / 0.25)) + 0.03 * (lt / max(dur, 0.1))
        sx = sy = 0
        for fl in flash_times:
            if 0 <= ft - fl < 1.2:
                k = (1 - (ft - fl) / 1.2) * 28
                sx += math.sin(ft * 83) * k; sy += math.cos(ft * 71) * k
        if z != 1.0 or sx or sy:
            cw, ch = W / z, H / z
            if z < 1.04: z = 1.04; cw, ch = W / z, H / z  # room for shake
            x0 = min(max((W - cw) / 2 + sx, 0), W - cw); y0 = min(max((H - ch) / 2 + sy, 0), H - ch)
            base = base.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))
        # flash overlay
        for fl in flash_times:
            if 0 <= ft - fl < 1.4:
                a = int(255 * (1 - (ft - fl) / 1.4) ** 1.6)
                base.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 250, a)))
        base.alpha_composite(VIGNETTE)
        base.alpha_composite(grain[fi % 6])
        # captions
        act = None
        for ci, idxs in enumerate(tl["chunks"]):
            s0, e0 = tl["wt"][idxs[0]][0], tl["wt"][idxs[-1]][1]
            if s0 <= ft < e0 + 0.15:
                act = (idxs, s0)
                break
        if act:
            idxs, s0 = act
            cur = next((i for i in idxs if tl["wt"][i][0] <= ft < tl["wt"][i][1]), idxs[-1])
            im = caption_img(tl["words"], idxs, cur)
            paste_c(base, im, W / 2, 1520, scale=0.75 + 0.25 * ease_out_back((ft - s0) / 0.18))
        # progress bar
        ImageDraw.Draw(base).rectangle((0, H - 14, int(W * ft / total), H), fill=(255, 230, 0, 255))
        return base.convert("RGB")

    if args.preview is not None:
        ts = [float(x) for x in args.preview.split(",")]
        sheet = Image.new("RGB", (270 * 5, 480 * ((len(ts) + 4) // 5)))
        for k, tt in enumerate(ts):
            sheet.paste(render(int(tt * FPS)).resize((270, 480)), ((k % 5) * 270, (k // 5) * 480))
        sheet.save(args.out); return

    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", audio_path,
                           "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", args.out],
                          stdin=subprocess.PIPE)
    for fi in range(nframes):
        ff.stdin.write(render(fi).tobytes())
        if fi % 150 == 0: print(f"frame {fi}/{nframes}", flush=True)
    ff.stdin.close(); ff.wait()
    print("wrote", args.out)

if __name__ == "__main__":
    main()
