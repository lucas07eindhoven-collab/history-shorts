"""
History Shorts generator — cartoon-style history shorts for adults, €0 stack.

Pipeline: topic -> script (Gemini, Groq fallback) -> voice (Edge TTS)
          -> cartoon images (Pollinations) -> video (FFmpeg) -> YouTube upload.

Run:  python generate.py            # full run (needs API keys in env)
      python generate.py --offline  # local test: canned script, placeholder art, silent audio
      python generate.py --no-upload
"""
import argparse
import asyncio
import datetime
import json
import os
import pathlib
import random
import re
import subprocess
import sys
import time
import urllib.parse

import requests
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "out"
POSTED = ROOT / "posted.json"
TOPICS = ROOT / "topics.txt"

LANG = os.getenv("SHORTS_LANG", "nl")  # "nl" or "en"
VOICES = {"nl": "nl-NL-MaartenNeural", "en": "en-US-GuyNeural"}
VOICE = os.getenv("SHORTS_VOICE", VOICES.get(LANG, "en-US-GuyNeural"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

W, H, FPS = 1080, 1920, 30
FONT = os.getenv("SHORTS_FONT", "DejaVu Sans")

# One fixed style string = every short looks like the same channel.
# Doodle meme style: one fixed string = every short looks like the same channel.
STYLE = (
    "simple 2D doodle cartoon, hand-drawn thin black outlines, oversized round heads, "
    "tiny thin noodle bodies, big bulging expressive eyes, exaggerated funny facial expression, "
    "flat soft pastel colors, white-ish skin tones, simple minimal background, internet meme comic style, "
    "square composition, centered characters, no text, no letters, no words, no speech bubbles"
)

# Series = fixed top box + "part N" -> people binge the series
SERIES = os.getenv("SHORTS_SERIES") or {
    "nl": "geschiedenis die compleet uit de hand liep",
    "en": "history moments that went way too far",
}.get(LANG, "history moments that went way too far")

IMG = 1080           # square cartoon panel
PANEL_Y = 470        # top of the panel on the 1080x1920 canvas
BOLD = os.getenv("SHORTS_FONT_FILE", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

LANG_NAME = {"nl": "Dutch (Nederlands)", "en": "English"}[LANG] if LANG in ("nl", "en") else LANG

SCRIPT_PROMPT = """You write VIRAL scripts for a faceless Shorts/TikTok/Reels channel: cartoon history for ADULTS.
Goal: maximum retention. Viewers decide in 1 second whether to swipe.
Tone: dry, savage, darkly funny, like a friend telling the most insane true story at a bar.
Real history only - never invent facts, numbers or quotes. Mature themes OK (war, scandal, death,
politics, betrayal) but nothing graphic or gory.

Topic: {topic}

STRUCTURE (follow exactly):
1. HOOK (scene 1): one short, shocking statement with a concrete detail or number. Start mid-action.
   Good: "In 1672 the Dutch ate their own prime minister." Bad: "Did you know...", "Let me tell you...".
2. CONTEXT (1-2 scenes): just enough setup, super fast, address the viewer as "you" where natural.
3. ESCALATION (4-7 scenes): every scene makes it crazier. Put a re-hook around the middle
   ("But that's not even the weird part." / "And then it got worse.").
4. PAYOFF (1 scene): the most absurd fact or twist.
5. LOOP (last scene): a final line that flows naturally back into the hook, so a rewatch feels seamless.

STYLE RULES:
- Narration language: {lang}. Image prompts: English.
- 10 to 13 scenes. ONE short sentence per scene, max 14 words. Total 130-160 words (~50 sec).
- Short punchy words. No filler, no dates unless they shock, no "like and subscribe".
- Mark the 2-4 most shocking scenes with "punch": true.
- Image prompt per scene: one funny, exaggerated cartoon moment with clear characters, emotion and
  action, era-appropriate clothing. Comedic expressions. Never any text or writing in the image.
- "title": max 55 characters, curiosity gap, in {lang}.

Return ONLY JSON:
{{"title": "...",
  "scenes": [{{"narration": "...", "image": "...", "punch": false}}],
  "description": "2 sentences in {lang}",
  "tags": ["5-8 tags without #"]}}"""

TOPIC_PROMPT = """Suggest ONE specific, little-known but true history story that adults would find
shocking, funny or absurd, suitable for a 50-second cartoon short. Not one of these: {used}.
Return ONLY JSON: {{"topic": "..."}}"""


# ---------------------------------------------------------------- helpers
def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"Command failed: {' '.join(map(str, cmd))}\n{r.stderr[-2000:]}")
    return r.stdout


def duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nw=1:nk=1", str(path)])
    return float(out.strip())


def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


# ---------------------------------------------------------------- LLM
def ask_llm(prompt):
    gkey, qkey = os.getenv("GEMINI_API_KEY"), os.getenv("GROQ_API_KEY")
    errors = []
    if gkey:
        try:
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{GEMINI_MODEL}:generateContent?key={gkey}")
            r = requests.post(url, timeout=90, json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.9},
            })
            r.raise_for_status()
            return parse_json(r.json()["candidates"][0]["content"]["parts"][0]["text"])
        except Exception as e:  # fall through to Groq
            errors.append(f"Gemini: {e}")
    if qkey:
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", timeout=90,
                              headers={"Authorization": f"Bearer {qkey}"},
                              json={"model": GROQ_MODEL, "temperature": 0.9,
                                    "response_format": {"type": "json_object"},
                                    "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            return parse_json(r.json()["choices"][0]["message"]["content"])
        except Exception as e:
            errors.append(f"Groq: {e}")
    sys.exit("No LLM worked. Set GEMINI_API_KEY and/or GROQ_API_KEY.\n" + "\n".join(errors))


def pick_topic(posted):
    used = set(posted["used"])
    if TOPICS.exists():
        fresh = [t.strip() for t in TOPICS.read_text(encoding="utf-8").splitlines()
                 if t.strip() and not t.startswith("#") and t.strip() not in used]
        if fresh:
            return fresh[0]
    return ask_llm(TOPIC_PROMPT.format(used="; ".join(list(used)[-60:]) or "none"))["topic"]


OFFLINE_SCRIPT = {
    "title": "Deze oorlog duurde korter dan je lunchpauze",
    "hook_text": "OORLOG VAN 38 MINUTEN",
    "scenes": [
        {"narration": "De kortste oorlog ooit duurde achtendertig minuten.", "image": "x", "punch": True},
        {"narration": "Zanzibar, 1896. De sultan sterft plotseling.", "image": "x", "punch": False},
        {"narration": "Zijn neef rent naar het paleis en claimt de troon.", "image": "x", "punch": False},
        {"narration": "Probleem: de Britten hadden al iemand anders gekozen.", "image": "x", "punch": False},
        {"narration": "Ze geven hem tot negen uur 's ochtends om te vertrekken.", "image": "x", "punch": False},
        {"narration": "Hij antwoordt met tweeduizend soldaten en één oud kanon.", "image": "x", "punch": False},
        {"narration": "Maar dat is niet eens het gekste deel.", "image": "x", "punch": True},
        {"narration": "Om twee over negen opent de Britse vloot het vuur.", "image": "x", "punch": False},
        {"narration": "Zijn enige oorlogsschip zinkt bijna meteen.", "image": "x", "punch": False},
        {"narration": "De sultan vlucht via de achterdeur.", "image": "x", "punch": False},
        {"narration": "Daarna sturen de Britten Zanzibar de rekening voor hun eigen kogels.", "image": "x", "punch": True},
        {"narration": "En zo duurde de kortste oorlog ooit...", "image": "x", "punch": False},
    ],
    "description": "De kortste oorlog uit de geschiedenis. Achtendertig minuten.",
    "tags": ["geschiedenis", "history", "zanzibar", "shorts"],
}


# ---------------------------------------------------------------- voice
async def _tts(text, path):
    """Returns word timings [(start_sec, end_sec, word)] from Edge TTS WordBoundary events."""
    import edge_tts
    words = []
    comm = edge_tts.Communicate(text, VOICE, rate=os.getenv("SHORTS_RATE", "+14%"), boundary="WordBoundary")
    with open(path, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                st = chunk["offset"] / 1e7
                words.append((st, st + chunk["duration"] / 1e7, chunk["text"]))
    return words


def fake_timings(text, secs):
    ws = text.split()
    total = sum(len(w) + 1 for w in ws) or 1
    out, t = [], 0.0
    for w in ws:
        d = secs * (len(w) + 1) / total
        out.append((t, t + d * 0.92, w))
        t += d
    return out


def make_voice(text, path, offline):
    if offline:  # silent placeholder, ~2.9 words/sec
        secs = max(1.5, len(text.split()) / 2.9)
        run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
             "-t", f"{secs:.2f}", "-q:a", "9", str(path)])
        return fake_timings(text, secs)
    for attempt in range(3):
        try:
            words = asyncio.run(_tts(text, path))
            return words or fake_timings(text, duration(path))
        except Exception as e:
            log(f"TTS retry {attempt + 1}: {e}")
            time.sleep(3)
    sys.exit("Edge TTS failed 3 times.")


# ---------------------------------------------------------------- images
def placeholder_image(text, path, seed):
    """Offline stand-in: a doodle guy with a big head, so layout tests look like the real thing."""
    rnd = random.Random(seed)
    bg = (rnd.randint(200, 235), rnd.randint(215, 240), rnd.randint(200, 230))
    img = Image.new("RGB", (IMG, IMG), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 760, IMG, IMG], fill=(150, 190, 120))
    cx = rnd.randint(380, 700)
    d.line([(cx, 600), (cx, 860)], fill="black", width=8)                 # body
    d.line([(cx, 680), (cx - 110, 760)], fill="black", width=8)           # arms
    d.line([(cx, 680), (cx + 110, 610)], fill="black", width=8)
    d.line([(cx, 860), (cx - 70, 990)], fill="black", width=8)            # legs
    d.line([(cx, 860), (cx + 70, 990)], fill="black", width=8)
    d.ellipse([cx - 190, 210, cx + 190, 620], fill=(250, 240, 228), outline="black", width=8)  # head
    for ex in (-75, 75):                                                  # bulging eyes
        d.ellipse([cx + ex - 55, 330, cx + ex + 55, 440], fill="white", outline="black", width=6)
        d.ellipse([cx + ex - 14, 372, cx + ex + 14, 400], fill="black")
    d.ellipse([cx - 45, 480, cx + 45, 560], fill=(40, 20, 20))            # shocked mouth
    img.save(path)


def make_image(prompt, path, seed, offline):
    if offline:
        return placeholder_image(prompt, path, seed)
    full = f"{prompt}, {STYLE}"
    url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(full) +
           f"?width={IMG}&height={IMG}&nologo=true&seed={seed}&model=flux")
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=180)
            if r.ok and r.headers.get("content-type", "").startswith("image"):
                path.write_bytes(r.content)
                Image.open(path).convert("RGB").resize((IMG, IMG)).save(path)
                return
            log(f"Image retry {attempt + 1}: HTTP {r.status_code}")
        except Exception as e:
            log(f"Image retry {attempt + 1}: {e}")
        time.sleep(8 * (attempt + 1))
    log("Image failed, using placeholder")
    placeholder_image(prompt, path, seed)


def make_background(img_path, out_path):
    """Full-screen background: the same panel, heavily blurred and washed out (like the reference)."""
    from PIL import ImageFilter, ImageEnhance
    im = Image.open(img_path).convert("RGB").resize((H, H)).crop(((H - W) // 2, 0, (H - W) // 2 + W, H))
    im = im.filter(ImageFilter.GaussianBlur(45))
    im = ImageEnhance.Brightness(im).enhance(1.12)
    im = ImageEnhance.Color(im).enhance(0.7)
    im.save(out_path)


def make_overlay(series, part, path):
    """Transparent PNG: white rounded title box on top + yellow 'part N' at the bottom."""
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    font = ImageFont.truetype(BOLD, 58)
    # wrap series title to max ~22 chars per line
    words, lines, cur = series.split(), [], ""
    for w in words:
        if len(cur + " " + w) > 27 and cur:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    lh = 70
    box_h = lh * len(lines) + 50
    widths = [d.textlength(l, font=font) for l in lines]
    box_w = int(max(widths)) + 80
    x0, y0 = (W - box_w) // 2, PANEL_Y - box_h - 40
    d.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=32, fill=(255, 255, 255, 240))
    for i, (l, lw) in enumerate(zip(lines, widths)):
        d.text(((W - lw) / 2, y0 + 22 + i * lh), l, font=font, fill=(15, 15, 15))
    pf = ImageFont.truetype(BOLD, 74)
    label = f"part {part}"
    pw = d.textlength(label, font=pf)
    d.text(((W - pw) / 2, PANEL_Y + IMG + 120), label, font=pf, fill=(255, 225, 0),
           stroke_width=5, stroke_fill=(40, 40, 40))
    ov.save(path)


# ---------------------------------------------------------------- video
TAIL = 0.12  # tiny gap after each line = fast pacing


def scene_clip(img, audio, out, idx, punch):
    secs = duration(audio) + TAIL
    frames = max(1, int(secs * FPS))
    bg = img.with_name(img.stem + "_bg.png")
    make_background(img, bg)
    # Every cut starts with a quick punch-in, then drifts slowly. Punch scenes zoom harder + shake.
    if punch:
        z = "if(lt(on,6),1.35-0.04*on,1.11+0.0012*(on-6))"
        shake = "+if(lt(on,10),14*sin(on*2.7),0)"
    else:
        drift = "0.0007*(on-6)" if idx % 2 == 0 else "-0.0007*(on-6)"
        z = f"if(lt(on,6),1.22-0.025*on,1.07+{drift}+{0.04 if idx % 2 else 0})"
        shake = ""
    x = f"iw/2-(iw/zoom/2){shake}"
    y = f"ih/2-(ih/zoom/2){shake.replace('sin', 'cos')}"
    fc = (f"[1:v]scale={IMG * 2}:{IMG * 2},zoompan=z='{z}':x='{x}':y='{y}'"
          f":d={frames}:s={IMG}x{IMG}:fps={FPS},eq=saturation=1.1[fg];"
          f"[0:v][fg]overlay=0:{PANEL_Y}:shortest=1,format=yuv420p[v]")
    run(["ffmpeg", "-y", "-loop", "1", "-framerate", str(FPS), "-i", str(bg), "-i", str(img), "-i", str(audio),
         "-filter_complex", fc, "-map", "[v]", "-map", "2:a",
         "-af", f"apad=pad_dur={TAIL}", "-t", f"{secs:.2f}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", str(FPS),
         "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2", str(out)])
    return secs


def ass_time(t):
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def esc(t):
    return t.replace("{", "(").replace("}", ")").replace("\\", "/")


def make_subs(script, timings, durations, path):
    """Karaoke captions: 2-3 words on screen, the spoken word pops in yellow. Hook headline on top."""
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{FONT},66,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,6,2,2,90,90,{H - PANEL_Y - IMG + 70},1
Style: Hook,{FONT},84,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,3,22,0,8,90,90,260,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    YEL, WHT = r"\c&H00E5FF&", r"\c&HFFFFFF&"
    ev = []
    t0 = 0.0
    for words, dur in zip(timings, durations):
        groups = [words[i:i + 3] for i in range(0, len(words), 3)]
        for gi, g in enumerate(groups):
            g_end = groups[gi + 1][0][0] if gi + 1 < len(groups) else dur - TAIL
            for wi, (ws, we, _) in enumerate(g):
                nxt = g[wi + 1][0] if wi + 1 < len(g) else g_end
                parts = []
                for wj, (_, _, w) in enumerate(g):
                    w = esc(w)
                    if wj == wi:  # active word: yellow + pop
                        parts.append(r"{" + YEL + r"\fscx112\fscy112\t(0,90,\fscx100\fscy100)}" + w.lower() + r"{" + WHT + "}")
                    else:
                        parts.append(w.lower())
                ev.append(f"Dialogue: 0,{ass_time(t0 + ws)},{ass_time(t0 + max(nxt, ws + 0.08))},Cap,,0,0,0,,"
                          + " ".join(parts))
        t0 += dur
    path.write_text(head + "\n".join(ev) + "\n", encoding="utf-8")


def make_sfx(workdir):
    """Royalty-free SFX synthesized with ffmpeg: a whoosh for every cut, a bass hit for punch scenes."""
    whoosh, boom = workdir / "whoosh.wav", workdir / "boom.wav"
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anoisesrc=d=0.35:c=pink:a=0.6",
         "-af", "highpass=f=400,lowpass=f=5000,afade=t=in:d=0.15,afade=t=out:st=0.18:d=0.17,volume=0.5",
         "-ar", "44100", "-ac", "2", str(whoosh)])
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=f=55:d=0.6",
         "-af", "volume=3,afade=t=out:st=0.05:d=0.55,lowpass=f=200",
         "-ar", "44100", "-ac", "2", str(boom)])
    return whoosh, boom


def build_video(script, workdir, offline, part=1):
    clips, durs, timings = [], [], []
    scenes = script["scenes"]
    for i, sc in enumerate(scenes):
        log(f"Scene {i + 1}/{len(scenes)}")
        audio, img, clip = workdir / f"a{i}.mp3", workdir / f"i{i}.png", workdir / f"c{i}.mp4"
        timings.append(make_voice(sc["narration"], audio, offline))
        make_image(sc["image"], img, seed=random.randint(1, 10**6), offline=offline)
        durs.append(scene_clip(img, audio, clip, i, bool(sc.get("punch"))))
        clips.append(clip)

    listing = workdir / "list.txt"
    listing.write_text("".join(f"file '{c.name}'\n" for c in clips))
    joined = workdir / "joined.mp4"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(joined)])
    make_subs(script, timings, durs, workdir / "subs.ass")

    # SFX mix: whoosh at every cut, boom at punch scenes
    make_overlay(SERIES, part, workdir / "overlay.png")
    whoosh, boom = make_sfx(workdir)
    inputs, filters, labels, t = [], [], [], 0.0
    for i, (sc, d) in enumerate(zip(scenes, durs)):
        src = boom if sc.get("punch") else whoosh
        n = len(inputs) // 2 + 1
        inputs += ["-i", src.name]
        ms = int(max(0, t - (0 if sc.get("punch") else 0.15)) * 1000)
        filters.append(f"[{n}:a]adelay={ms}|{ms},volume=0.55[s{i}]")
        labels.append(f"[s{i}]")
        t += d
    fc = (";".join(filters) + ";" + "".join(labels) + f"amix=inputs={len(labels)}:normalize=0[sfx];"
          "[0:a][sfx]amix=inputs=2:normalize=0,alimiter=limit=0.95[aout]")
    final = workdir / "final.mp4"
    cmd = (["ffmpeg", "-y", "-i", joined.name] + inputs +
           ["-i", "overlay.png", "-filter_complex",
            fc + f";[0:v][{len(inputs) // 2 + 1}:v]overlay=0:0,ass=subs.ass[vout]",
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", final.name])
    r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("Final render failed:\n" + r.stderr[-2500:])
    return final, sum(durs)


# ---------------------------------------------------------------- upload
def upload_youtube(video, script):
    cid, csec, rtok = (os.getenv(k) for k in
                       ("YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN"))
    if not all((cid, csec, rtok)):
        log("YouTube secrets not set -> skipping upload (video kept as artifact).")
        return None
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    creds = Credentials(None, refresh_token=rtok, client_id=cid, client_secret=csec,
                        token_uri="https://oauth2.googleapis.com/token",
                        scopes=["https://www.googleapis.com/auth/youtube.upload"])
    yt = build("youtube", "v3", credentials=creds, cache_discovery=False)
    tags = script.get("tags", [])[:10]
    body = {
        "snippet": {
            "title": script["title"][:95],
            "description": script.get("description", "") + "\n\n" + " ".join(f"#{t.replace(' ', '')}" for t in tags[:5]) + " #shorts",
            "tags": tags,
            "categoryId": "27",  # Education
            "defaultLanguage": LANG,
            "defaultAudioLanguage": LANG,
        },
        "status": {"privacyStatus": os.getenv("YT_PRIVACY", "public"),
                   "selfDeclaredMadeForKids": False},
    }
    req = yt.videos().insert(part="snippet,status", body=body,
                             media_body=MediaFileUpload(str(video), chunksize=-1, resumable=True))
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    log(f"Uploaded: https://youtube.com/shorts/{resp['id']}")
    return resp["id"]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--topic")
    args = ap.parse_args()

    posted = json.loads(POSTED.read_text()) if POSTED.exists() else {"used": []}
    if args.offline:
        topic, script = "Anglo-Zanzibar War (test)", OFFLINE_SCRIPT
    else:
        topic = args.topic or pick_topic(posted)
        log(f"Topic: {topic}")
        script = ask_llm(SCRIPT_PROMPT.format(topic=topic, lang=LANG_NAME))
    log(f"Title: {script['title']}  ({len(script['scenes'])} scenes)")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    workdir = OUT / stamp
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")

    part = len(posted["used"]) + 1
    final, secs = build_video(script, workdir, args.offline, part)
    if secs > 59:
        log(f"WARNING: {secs:.1f}s — over 60s won't count as a Short on every platform.")
    log(f"Video ready: {final} ({secs:.1f}s)")

    # Clean copy for the artifact / weekly TikTok+IG batch
    pub = OUT / "final"
    pub.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", script["title"].lower()).strip("-")[:50]
    (pub / f"{stamp}-{slug}.mp4").write_bytes(final.read_bytes())
    (pub / f"{stamp}-{slug}.txt").write_text(
        f"{script['title']}\n\n{script.get('description', '')}\n\n"
        + " ".join(f"#{t.replace(' ', '')}" for t in script.get("tags", [])), encoding="utf-8")

    if not args.offline and not args.no_upload:
        upload_youtube(final, script)
    if not args.offline:
        posted["used"].append(topic)
        POSTED.write_text(json.dumps(posted, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
