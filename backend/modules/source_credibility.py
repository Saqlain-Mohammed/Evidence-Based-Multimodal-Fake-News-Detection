"""Source credibility: documented, rule-based scoring (no arbitrary numbers)."""
import re
from urllib.parse import urlparse
import tldextract

# Use bundled suffix snapshot only — never fetch over network (avoids first-run stall).
try:
    _EXT = tldextract.TLDExtract(suffix_list_urls=None)
    def _split(domain: str):
        e = _EXT(domain)
        return e.domain, e.suffix, e.subdomain
except Exception:
    def _split(domain: str):
        parts = domain.rsplit(".", 2)
        if len(parts) == 3:
            return parts[1], parts[2], parts[0]
        if len(parts) == 2:
            return parts[0], parts[1], ""
        return domain, "", ""

# Methodology (shown in UI + API):
# base 50 + up to +45 for trusted tiers, -40 for known-misinfo / suspicious TLD,
# +5 https, -10 for sensational subdomains/slugs, clamped 5..98.
TRUSTED_TIER1 = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org",
    "pbs.org", "theguardian.com", "nytimes.com", "washingtonpost.com",
    "wsj.com", "ft.com", "economist.com", "nature.com", "science.org",
    "who.int", "un.org", "gov.uk", "nih.gov", "cdc.gov",
}
TRUSTED_TIER2 = {
    "cnn.com", "nbcnews.com", "abcnews.go.com", "cbsnews.com", "aljazeera.com",
    "dw.com", "france24.com", "ndtv.com", "thehindu.com", "indianexpress.com",
    "toi.in", "timesofindia.indiatimes.com", "hindustantimes.com", "bbc.in",
    "moneycontrol.com", "livemint.com", "deccanherald.com",
}
KNOWN_MISINFO = {
    "dailybuzzlive.com", "worldnewsdailyreport.com", "theonion.com",  # satire example
    "fakenews-network.com", "viraltruths.com", "clickbaitdaily.com",
}
SUSPICIOUS_TLDS = {"click", "buzz", "gq", "cf", "tk", "ml", "ga", "top", "xyz", "win", "loan"}
GOV_EDU_BONUS_SUFFIX = (".gov", ".edu", ".int")


def extract_source(url: str | None, content: str | None) -> dict:
    domain, full_url = "", (url or "").strip()
    if full_url and not re.match(r"^https?://", full_url, re.I):
        full_url = "http://" + full_url
    if full_url:
        try:
            dom, suf, _sub = _split(full_url)
            domain = ".".join(p for p in [dom, suf] if p)
            host = urlparse(full_url).hostname or domain
        except Exception:
            host = ""
            domain = ""
    else:
        host = ""
        # try to find a domain mentioned in the article text
        m = re.search(r"((?:https?://)?(?:www\.)?[a-z0-9-]+\.[a-z]{2,}(?:\.[a-z]{2,})?)", (content or ""), re.I)
        if m:
            domain = m.group(1).lower().replace("http://", "").replace("https://", "").replace("www.", "")
            host = domain
    return {"url": full_url, "domain": domain.lower(), "host": (host or "").lower()}


def score_source(url: str | None, content: str | None) -> dict:
    src = extract_source(url, content)
    domain = src["domain"]
    breakdown = []
    if not domain:
        return {**src, "credibility_score": 50, "tier": "Unknown (no source)",
                "breakdown": ["No URL/domain supplied — neutral score 50 assigned."],
                "fake_signal": 0.5}
    score = 50
    tier = "Unrated"
    if domain in TRUSTED_TIER1:
        score += 45; tier = "Tier 1 — internationally trusted"
        breakdown.append("+45 Tier-1 trusted outlet list")
    elif domain in TRUSTED_TIER2:
        score += 30; tier = "Tier 2 — established newsroom"
        breakdown.append("+30 Tier-2 established newsroom list")
    elif domain in KNOWN_MISINFO:
        score -= 40; tier = "Known misinformation / satire domain"
        breakdown.append("-40 known-misinfo/satire list")
    else:
        tier = "Unrated — not on curated lists"
        breakdown.append("+0 not on curated lists (neutral)")
    _dom, suf, _sub = _split(domain)
    if suf in SUSPICIOUS_TLDS:
        score -= 15; breakdown.append(f"-15 suspicious TLD (.{suf})")
    if any(s in domain for s in GOV_EDU_BONUS_SUFFIX) or domain.endswith(".gov") or domain.endswith(".edu"):
        score += 5; breakdown.append("+5 official gov/edu domain")
    if (url or "").lower().startswith("https://"):
        score += 5; breakdown.append("+5 serves HTTPS")
    slug = (url or "").lower()
    if any(w in slug for w in ["shocking", "viral", "exposed", "miracle", "secret"]):
        score -= 10; breakdown.append("-10 sensational URL slug")
    if len(domain.split(".")) > 3 or domain.count("-") >= 3:
        score -= 5; breakdown.append("-5 lookalike/complex domain shape")
    score = int(max(5, min(98, score)))
    fake_signal = round((100 - score) / 100.0, 3)
    return {**src, "credibility_score": score, "tier": tier,
            "breakdown": breakdown, "fake_signal": fake_signal,
            "methodology": "base 50 ± tier lists, TLD, HTTPS, slug/shape; clamped 5–98."}
