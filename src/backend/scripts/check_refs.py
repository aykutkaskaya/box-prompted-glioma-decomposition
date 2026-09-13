"""Verify every reference in the manuscript against Crossref and arXiv.

Each entry carries a DOI, an arXiv id, or both. The DOI is looked up on
Crossref and the arXiv id on the arXiv API, and the title that comes back is
compared with the title in the draft. A mismatch means the identifier points
somewhere else, which is the failure mode that survives proofreading -- the
reference reads correctly and resolves to the wrong paper.

Writes reports/reference_check.json and prints a summary.

    python scripts/check_refs.py
"""
from __future__ import annotations

import io
import os
import json
import html
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

DRAFT = DRIVE_ROOT / "PAPER_DRAFT.md"
OUT = DRIVE_ROOT / "reports" / "reference_check.json"
# Crossref asks for a contact address in the user agent for its polite pool.
# Set CROSSREF_MAILTO to your own; without it the request is anonymous.
UA = "reference-check/1.0" + (
    f" (mailto:{os.environ['CROSSREF_MAILTO']})"
    if os.environ.get("CROSSREF_MAILTO") else "")


def norm(s: str) -> str:
    s = re.sub(r"\s+", " ", s.lower())
    return re.sub(r"[^a-z0-9 ]", "", s).strip()


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def carries(entry: str, title: str) -> float:
    """How much of `title` the reference entry contains, 0 to 1.

    Scoring a retrieved title against a *parsed* title meant parsing the
    author list off the front, and the rule for that assumed a surname starts
    with a capital -- so "van de Mortel LA, van Wingen GA." stayed in the
    title and dragged a correct reference down to 0.87. Matching the title
    against the whole entry needs no author parsing: a title the entry
    carries scores 1.0 wherever it sits in the line.
    """
    e, t = norm(entry), norm(title)
    if not t:
        return 0.0
    m = SequenceMatcher(None, e, t).find_longest_match(0, len(e), 0, len(t))
    return round(m.size / len(t), 3)


def get(url: str, timeout: int = 20) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            if attempt == 2:
                print(f"    ! {type(e).__name__}: {e}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def parse(md: str) -> list[dict]:
    body = md[md.index("## References"):]
    out = []
    for line in body.split("\n"):
        m = re.match(r"^(\d+)\.\s+(.*)$", line.strip())
        if not m:
            continue
        n, text = int(m.group(1)), m.group(2)
        # title is the run between the first full stop after the author list
        # and the italicised venue
        t = re.search(r"^(.*?)\.\s+\*", text)
        title = t.group(1) if t else text[:120]
        title = re.sub(r"^[A-ZÅÄÖ][^.]*?(?:et al|[A-Z]{1,3})\.\s+", "", title, count=1)
        doi = re.search(r"DOI:\s*(10\.\S+?)\.?(?:\s|$)", text)
        arx = re.search(r"arXiv:(\d{4}\.\d{4,5})", text)
        out.append({
            "n": n,
            "title": title.strip(),
            "doi": doi.group(1).rstrip(".") if doi else None,
            "arxiv": arx.group(1) if arx else None,
            "raw": text,
            "url": (u.group(1).rstrip(".") if (u := re.search(
                r"URL:\s*(https?://\S+?)\.?(?:\s|$)", text)) else None),
            "flagged": "UNRESOLVED" in text,
        })
    return out


def landing(url: str) -> dict | None:
    """The citation_title a landing page declares, for entries with no DOI.

    NeurIPS and similar proceedings are not in Crossref, so an entry citing
    one used to score zero and be reported as a mismatch -- which claims the
    identifier resolves to the wrong paper, when nothing had been looked up
    at all. The page's own citation metadata settles it.
    """
    raw = get(url)
    if not raw:
        return None
    page = raw.decode("utf-8", "replace")
    t = re.search(r'citation_title"\s+content="([^"]+)"', page)
    return {"title": html.unescape(t.group(1))} if t else None


def crossref(doi: str) -> dict | None:
    raw = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
    if not raw:
        return None
    try:
        m = json.loads(raw)["message"]
    except Exception:
        return None
    ctr = m.get("container-title") or []
    parts = (m.get("issued", {}).get("date-parts") or [[None]])[0]
    return {
        "title": (m.get("title") or [""])[0],
        "venue": ctr[0] if ctr else "",
        "year": parts[0],
        "type": m.get("type"),
        "authors": len(m.get("author") or []),
    }


def arxiv(aid: str) -> dict | None:
    raw = get("http://export.arxiv.org/api/query?id_list=" + aid)
    if not raw:
        return None
    try:
        root = ET.fromstring(raw)
    except Exception:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    e = root.find("a:entry", ns)
    if e is None:
        return None
    title = (e.findtext("a:title", "", ns) or "").strip()
    if not title or title == "Error":
        return None
    return {
        "title": re.sub(r"\s+", " ", title),
        "authors": len(e.findall("a:author", ns)),
        "published": (e.findtext("a:published", "", ns) or "")[:10],
    }


def datacite(aid: str) -> dict | None:
    """arXiv metadata by way of its DataCite DOI.

    Preferred over the arXiv API because arXiv throttles aggressively and
    answers a throttled request with 429, which reads downstream as "not
    verified" rather than "not asked".
    """
    raw = get("https://api.datacite.org/dois/10.48550/arXiv." + aid)
    if not raw:
        return None
    try:
        at = json.loads(raw)["data"]["attributes"]
    except Exception:
        return None
    titles = at.get("titles") or []
    title = re.sub(r"\s+", " ", (titles[0].get("title") if titles else "") or "")
    if not title:
        return None
    return {
        "title": title,
        "authors": len(at.get("creators") or []),
        "published": str(at.get("publicationYear") or ""),
    }


def main() -> None:
    refs = parse(io.open(DRAFT, encoding="utf-8").read())
    print(f"{len(refs)} references\n")
    for r in refs:
        marks = []
        # 10.48550 is arXiv's own prefix, registered with DataCite rather than
        # Crossref, so looking it up there 404s on a reference that is fine;
        # the arXiv id beside it is what verifies those.
        if r["doi"] and not r["doi"].lower().startswith("10.48550/"):
            got = crossref(r["doi"])
            r["crossref"] = got
            if got:
                r["doi_match"] = carries(r["raw"], got["title"])
                marks.append(f"doi {r['doi_match']:.2f}")
            else:
                r["doi_match"] = None
                marks.append("doi UNRESOLVED")
            time.sleep(0.4)
        if r["arxiv"]:
            # DataCite first: the arXiv API answers a throttled request
            # with 429, and an unanswered lookup is not a verification
            got = datacite(r["arxiv"]) or arxiv(r["arxiv"])
            r["arxiv_meta"] = got
            if got:
                r["arxiv_match"] = carries(r["raw"], got["title"])
                marks.append(f"arxiv {r['arxiv_match']:.2f}")
            else:
                r["arxiv_match"] = None
                marks.append("arxiv UNRESOLVED")
            time.sleep(0.4)
        if not marks and r["url"]:
            got = landing(r["url"])
            r["landing"] = got
            if got:
                r["url_match"] = carries(r["raw"], got["title"])
                marks.append(f"url {r['url_match']:.2f}")
            else:
                r["url_match"] = None
                marks.append("url UNRESOLVED")
            time.sleep(0.4)
        if not marks:
            marks.append("no identifier")
        scores = [x for x in (r.get("doi_match"), r.get("arxiv_match"),
                              r.get("url_match")) if x is not None]
        # nothing looked up is not the same as a title that disagrees
        r["verdict"] = ("unchecked" if not scores else
                        "ok" if max(scores) >= 0.90 else
                        "check" if max(scores) >= 0.60 else "MISMATCH")
        print(f"  [{r['n']:>2}] {r['verdict']:<8} {' · '.join(marks):<26} "
              f"{r['title'][:58]}")

    OUT.write_text(json.dumps(refs, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    n_ok = sum(1 for r in refs if r["verdict"] == "ok")
    n_chk = sum(1 for r in refs if r["verdict"] == "check")
    n_bad = sum(1 for r in refs if r["verdict"] == "MISMATCH")
    # an entry carrying an identifier that returned nothing was not
    # verified, and saying only "0 mismatched" reads as though it was
    none_id = [r["n"] for r in refs if r["verdict"] == "unchecked"
               and not (r.get("doi") or r.get("arxiv") or r.get("url"))]
    failed = [r["n"] for r in refs if r["verdict"] == "unchecked"
              and (r.get("doi") or r.get("arxiv") or r.get("url"))]
    print(f"\n{n_ok} verified, {n_chk} to look at, {n_bad} mismatched")
    if none_id:
        print(f"  {len(none_id)} carry no identifier to check: {none_id}")
    if failed:
        print(f"  {len(failed)} NOT CHECKED -- the lookup failed, most "
              f"likely rate limiting; re-run before trusting this: {failed}")
    print(f"-> {OUT}")
    if n_bad or failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
