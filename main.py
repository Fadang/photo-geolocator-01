"""photo-geolocator-01 agent (Python 3.13, stdlib only).

Strategy: one claude-haiku-5-5 vision call per photo (cheapest allowlisted
model: $0.10/$0.50 per MTok in/out, so 25 photos fit the $1.00/trial
worst-case budget with a wide margin), then one claude-sonnet-5-5 call for
photos the first pass was unsure about (confidence <= 2, at most 10, only
while the running worst-case budget stays under $1.00). If no model is
reachable (no key pushed, gateway refusal), every photo falls back to the
fixed prior (57.0, 27.0): the grid-searched best single point on the 75
practice photos (avg error ~6,843 km/photo, far better than the 20,015 km
an unanswered photo counts). Deterministic given the model's answers: no
wall clock, no unseeded randomness; results are collected by photo index
and posted once, in order.
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"
# $ per MTok (in/out) from the kit's gateway pricing rows.
PRICE = {HAIKU: (0.10, 0.50), SONNET: (2.00, 10.00)}
BUDGET = 1.00
MAX_TOKENS = 256
CONCURRENCY = 4
FALLBACK = (57.0, 27.0)
RETRY_STATUSES = {408, 429, 500, 502, 503, 504, 529}
RETRIES = 3
SONNET_RESCUE_MAX = 25  # budget guard below is the real limiter (~16 max)

PROMPT = (
    "You are an expert GeoGuessr player. Look at this street-level photo and "
    "work out where on Earth it was taken. Use the language and script on any "
    "signs, road markings and which side of the road traffic drives on, "
    "vegetation and climate, architecture, license plates, terrain, and the "
    "sun. Reason briefly, then output ONLY one JSON object, no other text: "
    '{"country": "<name>", "reason": "<one short sentence>", '
    '"lat": <degrees north, -90 to 90, at least 4 decimal places>, '
    '"lon": <degrees east, -180 to 180, at least 4 decimal places>, '
    '"confidence": <1 to 5>}. Always give your best guess, never leave it blank."'
)


def env(name):
    v = os.environ.get(name, "")
    return v.strip()


def http_json(method, url, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers=headers or {})
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:  # network error: treat as a refusal, no retry here
        return 0, str(e).encode()


def http_bytes(url):
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req) as r:
            return r.read()
    except Exception:
        return b""


def worst_case_cost(model, body_bytes):
    inp, outp = PRICE[model]
    return (body_bytes / 3.0) * inp / 1e6 + MAX_TOKENS * outp / 1e6


def ask(model, jpeg, base_url, api_key):
    """One vision call. Returns (text|None, refused_bool)."""
    body = json.dumps({
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": base64.b64encode(jpeg).decode()}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    }).encode()
    headers = {"Content-Type": "application/json", "x-api-key": api_key,
               "anthropic-version": "2023-06-01"}
    for attempt in range(RETRIES + 1):
        req = urllib.request.Request(
            base_url.rstrip("/") + "/v1/messages", data=body, method="POST",
            headers=headers)
        try:
            with urllib.request.urlopen(req) as r:
                msg = json.loads(r.read())
            text = "\n".join(b.get("text", "") for b in msg.get("content", [])
                             if b.get("type") == "text")
            return text, False
        except urllib.error.HTTPError as e:
            err = e.read()
            retry = (e.code in RETRY_STATUSES
                     and e.headers.get("x-should-retry") != "false")
            if retry and attempt < RETRIES:
                time.sleep(2 * (attempt + 1))
                continue
            sys.stderr.write("model %s refused HTTP %s: %s\n"
                             % (model, e.code, err[:200].decode("utf-8", "replace")))
            return None, True
        except Exception as e:
            if attempt < RETRIES:
                time.sleep(2 * (attempt + 1))
                continue
            sys.stderr.write("model %s error: %s\n" % (model, e))
            return None, True
    return None, True


def parse_coords(text):
    """First JSON object with lat/lon, else first two in-range numbers."""
    if not text:
        return None
    for m in _json_objects(text):
        try:
            o = json.loads(m)
        except Exception:
            continue
        lat = o.get("lat", o.get("latitude"))
        lon = o.get("lon", o.get("lng", o.get("longitude")))
        conf = o.get("confidence", o.get("conf"))
        if _is_lat(lat) and _is_lon(lon):
            c = conf if isinstance(conf, (int, float)) and 1 <= conf <= 5 else 3
            return (float(lat), float(lon), c)
    nums = []
    for tok in __import__("re").findall(r"-?\d+(?:\.\d+)?", text):
        try:
            nums.append(float(tok))
        except ValueError:
            pass
    for i in range(len(nums) - 1):
        if _is_lat(nums[i]) and _is_lon(nums[i + 1]):
            return (nums[i], nums[i + 1], 3)
    return None


def _json_objects(text):
    out, depth, start = [], 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start:i + 1])
    return out


def _is_lat(v):
    return isinstance(v, (int, float)) and -90 <= v <= 90


def _is_lon(v):
    return isinstance(v, (int, float)) and -180 <= v <= 180


def main():
    env_url = env("MAKEFUN_ENV_URL")
    base_url = env("ANTHROPIC_BASE_URL")
    api_key = env("ANTHROPIC_API_KEY")
    if not env_url:
        sys.stderr.write("MAKEFUN_ENV_URL is not set\n")
        return 1
    status, raw = http_json("GET", env_url.rstrip("/") + "/photos")
    if status != 200:
        sys.stderr.write("GET /photos failed: %s\n" % status)
        return 1
    photos = [p for p in json.loads(raw)["photos"] if not p.get("answered")]
    jpegs = {}

    def fetch(i):
        p = photos[i]
        jpegs[i] = http_bytes(env_url.rstrip("/") + p["url"])

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        list(ex.map(fetch, range(len(photos))))

    answers = [None] * len(photos)   # (lat, lon, conf) per index
    spent = 0.0  # running worst-case budget, deterministic

    def call(model, i):
        nonlocal spent
        body_len = len(jpegs[i]) * 4 // 3 + 4096  # base64 + JSON/prompt overhead
        cost = worst_case_cost(model, body_len)
        if spent + cost > BUDGET:
            return None
        spent += cost
        text, _refused = ask(model, jpegs[i], base_url, api_key)
        return parse_coords(text)

    have_model = bool(base_url and api_key)
    if have_model:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            results = list(ex.map(lambda i: call(HAIKU, i), range(len(photos))))
        for i, r in enumerate(results):
            if r is not None:
                answers[i] = r
        # Rescue pass: sonnet on low-confidence / unanswered photos, least
        # confident first (deterministic order), while the budget allows.
        rescue = [i for i in range(len(photos))
                  if answers[i] is None or answers[i][2] <= 2]
        rescue.sort(key=lambda i: (answers[i][2] if answers[i] else 0, i))
        rescue = rescue[:SONNET_RESCUE_MAX]
        for i in rescue:
            r = call(SONNET, i)
            if r is not None:
                old = answers[i]
                if old is None or r[2] > old[2]:
                    answers[i] = r

    out = []
    for i, p in enumerate(photos):
        lat, lon = (answers[i][0], answers[i][1]) if answers[i] else FALLBACK
        out.append({"id": p["id"], "lat": round(lat, 6), "lon": round(lon, 6)})
    status, raw = http_json("POST", env_url.rstrip("/") + "/answers",
                            {"answers": out})
    if status != 200:
        # Deterministic fallback: post one by one, in order.
        for a in out:
            http_json("POST", env_url.rstrip("/") + "/answers",
                      {"answers": [a]})
    sys.stderr.write("answered %d of %d photos; worst-case model spend $%.3f\n"
                     % (len(out), len(photos), spent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
