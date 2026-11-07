import json
import logging
import os
import random
import re
import threading
from collections import Counter, deque
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
def env(name, default=None, required=False):
    v = os.environ.get(name)
    if v is None or not v.strip():
        if required:
            raise RuntimeError(f"Не задана переменная окружения {name}")
        return default
    return v.strip()  # убираем случайные пробелы и переводы строк при копировании


def norm_chat(v):
    m = re.match(r"^(?:https?://)?t\.me/([A-Za-z0-9_]+)/?$", v)
    return "@" + m.group(1) if m else v


def norm_channel(v):
    return re.sub(r"^(?:https?://)?t\.me/(?:s/)?", "", v.strip()).lstrip("@").strip("/")


BOT_TOKEN = env("BOT_TOKEN", required=True)
CHAT_ID = norm_chat(env("TARGET_CHAT_ID", required=True))
DATABASE_URL = env("DATABASE_URL", required=True)
LLM_API_KEY = env("LLM_API_KEY", required=True)
TICK_SECRET = env("TICK_SECRET", required=True)
SOURCES = [norm_channel(s) for s in env("SOURCES", required=True).split(",") if s.strip()]

LLM_BASE_URL = env("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
LLM_MODEL = env("LLM_MODEL", "openai/gpt-oss-120b")
IMAGE_PROVIDER = env("IMAGE_PROVIDER", "pixabay").lower()  # pixabay | unsplash | none
IMAGE_KEY = env("IMAGE_KEY", "")
TZ_NAME = env("TZ_NAME", "Europe/Moscow")
COLLECT_HOUR = int(env("COLLECT_HOUR", "6"))
POSTS_PER_DAY = int(env("POSTS_PER_DAY", "4"))
MIN_LEN = int(env("MIN_LEN", "100"))
MAX_AGE_HOURS = int(env("MAX_AGE_HOURS", "72"))
PUBLISH_TIMES = [
    tuple(int(x) for x in t.split(":"))
    for t in env("PUBLISH_TIMES", "10:00,13:00,16:00,19:00").split(",")
]

AD_RE = re.compile(r"реклам|erid|промокод|партн[её]рск", re.I)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
RUNNING = threading.Lock()
ERRORS = deque(maxlen=40)   # последние ошибки (видны в /status)
LAST = {"collect": None}    # отчёт о последнем сборе (виден в /status)


# ---------- ошибки без секретов ----------
def err_text(e):
    if isinstance(e, httpx.HTTPStatusError):
        msg = f"HTTP {e.response.status_code}: {e.response.text[:300]}"
    else:
        msg = f"{type(e).__name__}: {e}"
    for secret in (BOT_TOKEN, LLM_API_KEY, IMAGE_KEY, TICK_SECRET):
        if secret:
            msg = msg.replace(secret, "***")
    return msg


def note(where, e):
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z {where}: {err_text(e)}"
    ERRORS.append(line)
    log.warning(line)


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
        c.execute("ALTER TABLE queue ADD COLUMN IF NOT EXISTS error TEXT")
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
    """Возвращает (посты, статистика). Статистика показывает, почему посты отсеялись."""
    r = httpx.get(f"https://t.me/s/{channel}", headers=UA, timeout=20, follow_redirects=True)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    cutoff = datetime.now(timezone.utc) - timedelta(hours=MAX_AGE_HOURS)
    st, out = Counter(), []
    for m in soup.select("div.tgme_widget_message"):
        st["messages"] += 1
        ref = m.get("data-post", "")
        body = m.select_one("div.tgme_widget_message_text")
        t = m.select_one("a.tgme_widget_message_date time")
        if not (ref and body and t and t.get("datetime")):
            st["no_text"] += 1  # посты без текста: только картинка, видео, опрос
            continue
        if datetime.fromisoformat(t["datetime"]) < cutoff:
            st["too_old"] += 1
            continue
        for br in body.find_all("br"):
            br.replace_with("\n")
        text = body.get_text().strip()
        if len(text) < MIN_LEN:
            st["too_short"] += 1
            continue
        if AD_RE.search(text):
            st["ad"] += 1
            continue
        st["ok"] += 1
        out.append((channel, int(ref.split("/")[-1]), text))
    if not st["messages"]:
        st["hint"] = "лента не найдена: у канала отключён веб-просмотр, он не публичный или имя неверное"
    return out, dict(st)


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
    if not m:
        raise ValueError("модель вернула не JSON: " + content[:200])
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
    rep = {"started": now.isoformat(timespec="seconds"), "force": force, "sources": {}}
    LAST["collect"] = rep
    cands = []
    for ch in SOURCES:
        try:
            posts, st = fetch_posts(ch)
            rep["sources"][ch] = st
            cands += posts
        except Exception as e:
            rep["sources"][ch] = {"error": err_text(e)}
            note(f"source {ch}", e)
    with db() as c:
        seen = set(c.execute("SELECT source, post_id FROM seen").fetchall())
    cands = [x for x in cands if (x[0], x[1]) not in seen]
    random.shuffle(cands)
    rep["new_candidates"] = len(cands)

    made = skipped = llm_errors = 0
    for ch, pid, text in cands[: POSTS_PER_DAY * 3]:
        if made >= POSTS_PER_DAY:
            break
        try:
            res = rewrite(text)
        except Exception as e:  # сбой API: пост не помечаем как просмотренный
            llm_errors += 1
            note("rewrite", e)
            continue
        with db() as c:
            c.execute("INSERT INTO seen VALUES (%s,%s) ON CONFLICT DO NOTHING", (ch, pid))
        if res.get("skip") or not res.get("text"):
            skipped += 1
            continue
        try:
            img = find_image(res.get("keywords", ""))
        except Exception as e:
            note("image", e)
            img = None
        with db() as c:
            c.execute(
                "INSERT INTO queue (text, image_url, publish_at) VALUES (%s,%s,%s)",
                (res["text"].strip(), img, schedule_time(made, now, force)),
            )
        made += 1
    rep.update(queued=made, skipped_by_model=skipped, llm_errors=llm_errors)
    log.info("collect: %s", rep)
    return made


# ---------- публикация ----------
def tg(method, files=None, **data):
    r = httpx.post(f"https://api.telegram.org/bot{BOT_TOKEN}/{method}", data=data, files=files, timeout=60)
    if r.status_code != 200:  # без raise_for_status: в его тексте был бы токен бота
        raise RuntimeError(f"Telegram {method}: HTTP {r.status_code} {r.text[:300]}")


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
        except Exception as e:
            note("sendPhoto", e)
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
        error = None
        try:
            send(*row)
            status = "sent"
        except Exception as e:
            status, error = "failed", err_text(e)
            note(f"publish #{i}", e)
        with db() as c:
            c.execute("UPDATE queue SET status=%s, error=%s WHERE id=%s", (status, error, i))


# ---------- веб-сервис ----------
def collect_job(force):
    if not RUNNING.acquire(blocking=False):
        return
    try:
        made = collect(force)
        if made and not force:  # день считается сделанным, только если что-то реально поставлено в очередь
            set_state("collect_day", datetime.now(ZoneInfo(TZ_NAME)).date().isoformat())
    except Exception as e:
        note("collect", e)
    finally:
        RUNNING.release()


@asynccontextmanager
async def lifespan(_):
    init_db()
    yield


app = FastAPI(lifespan=lifespan)


def check(key):
    if key != TICK_SECRET:
        raise HTTPException(status_code=403)


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"ok": True}


@app.api_route("/tick", methods=["GET", "HEAD"])
def tick(bg: BackgroundTasks, key: str = "", force: bool = False):
    check(key)
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


@app.get("/status")
def status(key: str = ""):
    """Состояние очереди, последний сбор и последние ошибки (без секретов)."""
    check(key)
    with db() as c:
        counts = dict(c.execute("SELECT status, count(*) FROM queue GROUP BY status").fetchall())
        rows = c.execute(
            "SELECT id, status, publish_at, left(text, 70), error FROM queue ORDER BY id DESC LIMIT 10"
        ).fetchall()
        seen_n = c.execute("SELECT count(*) FROM seen").fetchone()[0]
    return {
        "now": datetime.now(ZoneInfo(TZ_NAME)).isoformat(timespec="seconds"),
        "collect_running": RUNNING.locked(),
        "collect_day": get_state("collect_day"),
        "attempts": get_state("attempts"),
        "queue_counts": counts,
        "seen_posts": seen_n,
        "last_queue_rows": [list(r) for r in rows],
        "last_collect": LAST["collect"],
        "last_errors": list(ERRORS),
        "config": {
            "target_chat": CHAT_ID, "sources": SOURCES, "llm_base_url": LLM_BASE_URL, "llm_model": LLM_MODEL,
            "image_provider": IMAGE_PROVIDER, "min_len": MIN_LEN, "max_age_hours": MAX_AGE_HOURS,
            "posts_per_day": POSTS_PER_DAY, "publish_times": PUBLISH_TIMES, "collect_hour": COLLECT_HOUR,
        },
    }


@app.get("/reset")
def reset(key: str = "", seen: bool = False):
    """Вернуть неудавшиеся посты в очередь и разрешить новый сбор. seen=1 ещё и забывает просмотренные посты."""
    check(key)
    with db() as c:
        n = c.execute(
            "UPDATE queue SET status='pending', publish_at=now(), error=NULL WHERE status IN ('failed','sending')"
        ).rowcount
        c.execute("DELETE FROM state WHERE key IN ('collect_day','attempts')")
        if seen:
            c.execute("DELETE FROM seen")
    return {"requeued": n, "seen_cleared": seen}


SAMPLE = (
    "Разработчики опубликовали новый трейлер игры, в котором показали город, главных героев и несколько "
    "игровых механик. По словам студии, работа над проектом идёт по плану, а подробности о версиях для "
    "консолей объявят ближе к релизу."
)


@app.get("/selftest")
def selftest(key: str = ""):
    """Проверяет по очереди базу, Telegram, нейросеть, картинки и каждый источник."""
    check(key)
    out = {}

    def run(name, fn):
        try:
            out[name] = fn()
        except Exception as e:
            out[name] = {"ERROR": err_text(e)}

    def t_db():
        with db() as c:
            c.execute("SELECT 1")
        return "ok"

    def t_telegram():
        base = f"https://api.telegram.org/bot{BOT_TOKEN}"
        me = httpx.get(base + "/getMe", timeout=20).json()
        if not me.get("ok"):
            return {"ERROR": "getMe: " + str(me.get("description"))}
        res = {"bot": "@" + me["result"]["username"]}
        chat = httpx.get(base + "/getChat", params={"chat_id": CHAT_ID}, timeout=20).json()
        if not chat.get("ok"):
            res["ERROR"] = f"getChat({CHAT_ID}): {chat.get('description')}"
            return res
        res["chat"] = {"title": chat["result"].get("title"), "type": chat["result"].get("type")}
        mem = httpx.get(base + "/getChatMember",
                        params={"chat_id": CHAT_ID, "user_id": me["result"]["id"]}, timeout=20).json()
        if not mem.get("ok"):
            res["ERROR"] = "getChatMember: " + str(mem.get("description"))
            return res
        res["bot_status"] = mem["result"].get("status")
        res["can_post_messages"] = mem["result"].get("can_post_messages")
        return res

    def t_llm():
        return {"model": LLM_MODEL, "base_url": LLM_BASE_URL, "answer": rewrite(SAMPLE)}

    def t_image():
        return find_image("video game controller") or "ключ работает, но картинок не найдено"

    def t_sources():
        res = {}
        for ch in SOURCES:
            try:
                res[ch] = fetch_posts(ch)[1]
            except Exception as e:
                res[ch] = {"ERROR": err_text(e)}
        return res

    run("database", t_db)
    run("telegram", t_telegram)
    run("llm", t_llm)
    run("image", t_image)
    run("sources", t_sources)
    return out
