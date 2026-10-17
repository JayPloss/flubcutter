"""Get book text in: split an EPUB or a plain-text file into chapters."""

import posixpath
import re
import zipfile
from html.parser import HTMLParser
from io import BytesIO
from xml.etree import ElementTree as ET

MIN_CHAPTER_WORDS = 250

_BLOCK = {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "tr", "section"}
_HEAD = {"h1", "h2", "h3"}
_SKIP = {"script", "style", "head", "title"}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.heading, self._in_head, self._skip = [], [], 0, 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self._skip += 1
        if tag in _BLOCK:
            self.parts.append("\n")
        if tag in _HEAD:
            self._in_head += 1

    def handle_endtag(self, tag):
        if tag in _SKIP and self._skip:
            self._skip -= 1
        if tag in _HEAD and self._in_head:
            self._in_head -= 1
            self.heading.append("\n")
        if tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_head:
            self.heading.append(data)
        else:
            self.parts.append(data)


def _clean(text: str) -> str:
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def epub_chapters(data: bytes) -> list[dict]:
    """[{title, text}] for each document in the EPUB's reading order that has
    enough text to be a chapter (front matter and tiny pages are skipped)."""
    z = zipfile.ZipFile(BytesIO(data))
    ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container",
          "o": "http://www.idpf.org/2007/opf"}
    opf_path = ET.fromstring(z.read("META-INF/container.xml")).find(
        ".//c:rootfile", ns).attrib["full-path"]
    opf = ET.fromstring(z.read(opf_path))
    base = posixpath.dirname(opf_path)
    manifest = {i.attrib["id"]: i.attrib["href"] for i in opf.findall(".//o:manifest/o:item", ns)}
    spine = [i.attrib["idref"] for i in opf.findall(".//o:spine/o:itemref", ns)]

    chapters = []
    for idref in spine:
        href = manifest.get(idref)
        if not href:
            continue
        path = posixpath.normpath(posixpath.join(base, href.split("#")[0]))
        try:
            html = z.read(path).decode("utf-8", errors="replace")
        except KeyError:
            continue
        p = _Text()
        p.feed(html)
        body = _clean("".join(p.parts))
        heading = " ".join(_clean("".join(p.heading)).split("\n")).strip()
        if len(body.split()) < MIN_CHAPTER_WORDS:
            continue
        chapters.append({"title": heading or f"Chapter {len(chapters) + 1}", "text": body})
    return chapters


_CHAPTER_RE = re.compile(r"^\s*(chapter\s+([0-9]+|[ivxlc]+|[a-z-]+)\b.*)$", re.IGNORECASE | re.MULTILINE)


def text_chapters(text: str) -> list[dict]:
    """Split plain text on lines starting with "Chapter ..."; no headings = one chapter."""
    marks = list(_CHAPTER_RE.finditer(text))
    if not marks:
        return [{"title": "Chapter 1", "text": _clean(text)}]
    out = []
    for k, m in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(text)
        body = text[m.end():end]
        # A title often sits on the line after "CHAPTER ONE".
        lines = [ln.strip() for ln in body.strip().splitlines()]
        title = m.group(1).strip()
        if lines and 0 < len(lines[0].split()) <= 8 and not lines[0].endswith((".", "?", "!", '"')):
            title = f"{title}: {lines[0]}"
            body = "\n".join(lines[1:])
        out.append({"title": title.title() if title.isupper() else title, "text": _clean(body)})
    return out
