"""Full site + model test suite. Prints PASS/FAIL per check."""
import io
import json
import urllib.request
import urllib.parse

BASE = "http://127.0.0.1:8000"
ok = True

def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + (f" | {extra}" if extra else ""))
    if not cond:
        ok = False

def get(path, timeout=30):
    return urllib.request.urlopen(BASE + path, timeout=timeout).read().decode()

def post_form(path, fields, files=None, timeout=150):
    B = "----testbdry"
    body = b""
    for k, v in fields.items():
        body += ("--" + B + '\r\nContent-Disposition: form-data; name="' + k + '"\r\n\r\n' + v + "\r\n").encode()
    for k, (fn, data, ct) in (files or {}).items():
        body += ("--" + B + '\r\nContent-Disposition: form-data; name="' + k + '"; filename="' + fn + '"\r\nContent-Type: ' + ct + "\r\n\r\n").encode() + data + ("\r\n--" + B + "--\r\n").encode()
    if not files:
        body += ("--" + B + "--\r\n").encode()
    req = urllib.request.Request(BASE + path, data=body, headers={"Content-Type": "multipart/form-data; boundary=" + B})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

# 1. health
h = json.loads(get("/api/health"))
check("health ok + 3 trained heads", h["status"] == "ok" and h["text_model"] and h["image_model"] and h["fusion"], str(h))

# 2. homepage = new dual-mode UI
home = get("/")
for needle in ["modeFake", "modeAI", "inPrompt", "inAiText", "stepper", "quickView", "detailTabs", "Analyze Now"]:
    check(f"ui has {needle}", needle in home)

# 3. fake-news analyze
d = post_form("/api/analyze", {"url": "", "headline": "SHOCKING miracle cure exposed",
    "content": "Unbelievable secret cure guaranteed 100 percent. Share urgently before deleted!"})
check("fake analyze label+probs", d["label"] in ("LIKELY FAKE", "UNCERTAIN / NEEDS REVIEW", "LIKELY REAL") and 0 <= d["fake_probability"] <= 1, d["label"])
check("fake evidence live", d["evidence"]["status"] in ("ok", "insufficient"), d["evidence"]["status"])
check("fake contradiction", d["contradiction"]["overall"] in ("SUPPORTS", "CONTRADICTS", "NOT ENOUGH INFO"), d["contradiction"]["overall"])
check("fake explanation non-empty", len(d["explanation"]) >= 5, str(len(d["explanation"])))

# 4. AI text-only: LLM-ish prose must score HIGHER than human prose (directional + separation)
ai1 = post_form("/api/analyze_ai", {"text": "Furthermore, it is important to note that the tapestry of modern society delves into numerous facets. Moreover, the landscape leverages pivotal paradigms. Additionally, in conclusion, the vibrant realm overall showcases a testament to progress.", "url": ""})
check("ai text engine active", ai1["text"]["present"] and ai1["text"]["p_ai"] is not None, str(ai1["text"].get("p_ai")))
check("ai indicators present", len(ai1["text"]["indicators"]) >= 3, str(len(ai1["text"]["indicators"])))

# 5. AI text-only (human prose -> low)
ai2 = post_form("/api/analyze_ai", {"text": "I saw it myself yesterday, I swear - omg haha you wont believe what happened, it was wild! I dunno what to say, honestly it cracked me up.", "url": ""})
check("ai separates LLM prose from human prose", ai1["text"]["p_ai"] > ai2["text"]["p_ai"] + 0.15,
      f"llm={ai1['text']['p_ai']} human={ai2['text']['p_ai']}")

# 6. AI image-only
from PIL import Image
buf = io.BytesIO(); Image.new("RGB", (400, 400), (120, 140, 200)).save(buf, "PNG")
ai3 = post_form("/api/analyze_ai", {"text": "", "url": ""}, {"image": ("t.png", buf.getvalue(), "image/png")})
check("ai image scored", ai3["image"]["present"] and ai3["image"]["p_ai"] is not None, str(ai3["image"]["p_ai"]))
check("ai image indicators", len(ai3["image"]["indicators"]) >= 2, str(len(ai3["image"]["indicators"])))

# 7. AI combined
ai4 = post_form("/api/analyze_ai", {"text": "The council met on Tuesday and approved the budget with a majority vote.", "url": ""}, {"image": ("t.png", buf.getvalue(), "image/png")})
check("ai combined label", ai4["label"] in ("LIKELY AI-GENERATED", "UNCERTAIN — MIXED SIGNALS", "LIKELY HUMAN-MADE"), ai4["label"])
check("ai explanation", len(ai4["explanation"]) >= 4, str(len(ai4["explanation"])))

# 8. explore + stats
ex = json.loads(get("/api/explore?topic=" + urllib.parse.quote("metro rail Hyderabad"), timeout=90))
check("explore evidence", ex["status"] == "ok" and len(ex["items"]) > 0, str(len(ex.get("items", []))))
st = json.loads(get("/api/stats"))
check("stats trained metrics", st["trained"] and st["metrics"]["fusion"]["test"]["accuracy"] is not None)
check("stats datasets", len(st["datasets"]) == 5)

print("\nALL PASS" if ok else "\nSOME FAILURES")
