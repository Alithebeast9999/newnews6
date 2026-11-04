import json
import logging
import os
import random
import re
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
import psycopg
from bs4 import BeautifulSoup
from fastapi import BackgroundTasks, FastAPI, HTTPException

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bot")

# ---------- настройки (переменные окружения) ----------
BOT_TOKEN = os.environ["BOT_TOKEN"]
CHAT_ID = os.environ["TARGET_CHAT_ID"]          # id или @username вашей группы/канала
DATABASE_URL = os.environ["DATABASE_URL"]
LLM_API_KEY = os.environ["LLM_API_KEY"]         # ключ любого провайдера с OpenAI-совместимым API
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
IMAGE_PROVIDER = os.environ.get("IMAGE_PROVIDER", "pixabay")  # pixabay | unsplash | none
IMAGE_KEY = os.environ.get("IMAGE_KEY", "")
TICK_SECRET = os.environ["TICK_SECRET"]         # любая длинная случайная строка
SOURCES = [s.strip().lstrip("@") for s in os.environ["SOURCES"].split(",") if s.strip()]

LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
TZ_NAME = os.environ.get("TZ_NAME", "Europe/Moscow")
COLLECT_HOUR = int(os.environ.get("COLLECT_HOUR", "6"))
POSTS_PER_DAY = int(os.environ.get("POSTS_PER_DAY", "4"))
MIN_LEN = int(os.environ.get("MIN_LEN", "200"))
PUBLISH_TIMES = [
    tuple(int(x) for x in t.split(":"))
    for t in os.environ.get("PUBLISH_TIMES", "10:00,13:00,16:00,19:00").split(",")
]

AD_RE = re.compile(r"реклам|erid|промокод|партн[её]рск", re.I)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
RUNNING = threading.Lock()


# ---------- база ----------
def db():
    return psycopg.connect(DATABASE_URL, autocommit=True)


def init_db():
    with db() as c:
        c.execute("CREATE TABLE IF NOT EXISTS seen (source TEXT, post_id INT, PRIMARY KEY (source, post_id))")
        c.execute(
            "CREATE TABLE IF NOT EXISTS queue (id SERIAL PRIMARY KEY, text TEXT NOT NULL, image_url TEXT, "
            "publish_at TIMESTAMPTZ NOT NULL, status TEXT NOT NULL DEFAULT 'pending')"
        )
        c.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT)")


def get_state(key):
    with db() as c:
        row = c.execute("SELECT value FROM state WHERE key=%s", (key,)).fetchone()
    return row[0] if row else None


def set_state(key, value):
    with db() as c:
        c.execute(
            "INSERT INTO state VALUES (%s,%s) ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value",
            (key, value),
        )


def take_attempt(today):
    """Не больше 3 попыток сбора в сутки, чтобы ошибка не жгла лимиты API каждые 5 минут."""
    d, _, n = (get_state("attempts") or "").partition(":")
    n = int(n) if d == today and n else 0
    if n >= 3:
        return False
    set_state("attempts", f"{today}:{n + 1}")
    return True


# ---------- сбор постов ----------
def fetch_posts(channel):
    r = httpx.get(f"https://t.me/s/{channel}", headers=UA, timeout=20, follow_redirects=True)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    cutoff = datetime.now(timezone.utc) - timedelta(hours=36)
    out = []
    for m in soup.select("div.tgme_widget_message"):
        ref = m.get("data-post", "")
        body = m.select_one("div.tgme_widget_message_text")
        t = m.select_one("a.tgme_widget_message_date time")
        if not (ref and body and t and t.get("datetime")):
            continue
        if datetime.fromisoformat(t["datetime"]) < cutoff:
            continue
        for br in body.find_all("br"):
            br.replace_with("\n")
        text = body.get_text().strip()
        if len(text) < MIN_LEN or AD_RE.search(text):
            continue
        out.append((channel, int(ref.split("/")[-1]), text))
    return out


# ---------- LLM и картинки ----------
PROMPT = (
    "Перепиши пост для Telegram на русском языке своими словами. Сохрани факты и смысл, "
    "но измени структуру и формулировки. Убери ссылки, хештеги, упоминания каналов и призывы подписаться. "
    "Длина не более 900 символов. Если пост является рекламой или не годится для публикации, верни skip=true. "
    'Ответ строго в JSON: {"skip": false, "text": "...", "keywords": "2-3 английских слова для поиска фото по теме"}.\n\n'
    "Пост:\n"
)


def rewrite(text):
    r = httpx.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": PROMPT + text[:3000]}],
            "temperature": 0.7,
        },
        timeout=90,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"] or ""
    m = re.search(r"\{.*\}", content, re.S)
    return json.loads(m.group(0))


def find_image(query):
    if not query.strip() or IMAGE_PROVIDER == "none":
        return None
    if IMAGE_PROVIDER == "unsplash":
        r = httpx.get(
            "https://api.unsplash.com/search/photos",
            headers={"Authorization": f"Client-ID {IMAGE_KEY}"},
            params={"query": query, "per_page": 10, "orientation": "landscape"},
            timeout=20,
        )
        r.raise_for_status()
        urls = [p["urls"]["regular"] for p in r.json().get("results", [])]
    else:  # pixabay
        r = httpx.get(
            "https://pixabay.com/api/",
            params={"key": IMAGE_KEY, "q": query, "image_type": "photo",
                    "orientation": "horizontal", "safesearch": "true", "per_page": 10},
            timeout=20,
        )
        r.raise_for_status()
        urls = [h["largeImageURL"] for h in r.json().get("hits", [])]
    return random.choice(urls) if urls else None


def schedule_time(i, now, force):
    if force:
        return datetime.now(timezone.utc) + timedelta(minutes=2 * i)
    hh, mm = PUBLISH_TIMES[i % len(PUBLISH_TIMES)]
    t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return t if t > now else t + timedelta(days=1)


def collect(force=False):
    now = datetime.now(ZoneInfo(TZ_NAME))
    cands = []
    for ch in SOURCES:
        try:
            cands += fetch_posts(ch)
        except Exception as e:
            log.warning("source %s failed: %s", ch, e)
    with db() as c:
        seen = set(c.execute("SELECT source, post_id FROM seen").fetchall())
    cands = [x for x in cands if (x[0], x[1]) not in seen]
    random.shuffle(cands)
    log.info("candidates: %d", len(cands))

    made = 0
    for ch, pid, text in cands[: POSTS_PER_DAY * 3]:
        if made >= POSTS_PER_DAY:
            break
        try:
            res = rewrite(text)
        except Exception as e:  # лимит или сбой API: пост не помечаем как просмотренный
            log.warning("rewrite failed: %s", e)
            continue
        with db() as c:
            c.execute("INSERT INTO seen VALUES (%s,%s) ON CONFLICT DO NOTHING", (ch, pid))
        if res.get("skip") or not res.get("text"):
            continue
        try:
            img = find_image(res.get("keywords", ""))
        except Exception as e:
            log.warning("image failed: %s", e)
            img = None
        with db() as c:
            c.execute(
                "INSERT INTO queue (text, image_url, publish_at) VALUES (%s,%s,%s)",
                (res["text"].strip(), img, schedule_time(made, now, force)),
            )
        made += 1
    log.info("queued: %d", made)


# ---------- публикация ----------
def tg(method, files=None, **data):
    r = httpx.post(f"https://api.telegram.org/bot{BOT_TOKEN}/{method}", data=data, files=files, timeout=60)
    r.raise_for_status()


def send(text, image):
    if image:
        try:
            pic = httpx.get(image, headers=UA, timeout=30, follow_redirects=True)
            pic.raise_for_status()
            short = len(text) <= 1024
            tg("sendPhoto", files={"photo": ("photo.jpg", pic.content)}, chat_id=CHAT_ID,
               **({"caption": text} if short else {}))
            if short:
                return
        except httpx.HTTPError as e:
            log.warning("sendPhoto failed: %s", e)
    tg("sendMessage", chat_id=CHAT_ID, text=text[:4096])


def publish_due():
    with db() as c:
        ids = [
            r[0]
            for r in c.execute(
                "SELECT id FROM queue WHERE status='pending' AND publish_at <= now() ORDER BY publish_at LIMIT 5"
            ).fetchall()
        ]
    for i in ids:
        with db() as c:  # атомарно «занимаем» пост, чтобы параллельные тики не продублировали
            row = c.execute(
                "UPDATE queue SET status='sending' WHERE id=%s AND status='pending' RETURNING text, image_url",
                (i,),
            ).fetchone()
        if not row:
            continue
        try:
            send(*row)
            status = "sent"
        except Exception as e:
            log.warning("publish %s failed: %s", i, e)
            status = "failed"
        with db() as c:
            c.execute("UPDATE queue SET status=%s WHERE id=%s", (status, i))


# ---------- веб-сервис ----------
def collect_job(force):
    if not RUNNING.acquire(blocking=False):
        return
    try:
        collect(force)
        if not force:
            set_state("collect_day", datetime.now(ZoneInfo(TZ_NAME)).date().isoformat())
    except Exception:
        log.exception("collect failed")
    finally:
        RUNNING.release()


@asynccontextmanager
async def lifespan(_):
    init_db()
    yield


app = FastAPI(lifespan=lifespan)


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"ok": True}


@app.api_route("/tick", methods=["GET", "HEAD"])
def tick(bg: BackgroundTasks, key: str = "", force: bool = False):
    if key != TICK_SECRET:
        raise HTTPException(status_code=403)
    now = datetime.now(ZoneInfo(TZ_NAME))
    today = now.date().isoformat()
    if force or (
        now.hour >= COLLECT_HOUR
        and get_state("collect_day") != today
        and not RUNNING.locked()
        and take_attempt(today)
    ):
        bg.add_task(collect_job, force)
    publish_due()
    return {"ok": True}
