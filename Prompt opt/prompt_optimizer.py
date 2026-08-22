#!/usr/bin/env python3
"""
Prompt Optimizer
Rule-based prompt compression: strips filler, compresses verbose phrasing, and
leaves code, placeholders and document structure untouched. Telegraphic mode
goes one step further and rewrites an imperative line as a note — "Create a txt
on the desktop with the ingredients" -> "Create desktop txt: ingredients".

No API and no model. tiktoken is the only optional dependency, and only for
exact token counts — without it the app falls back to a word-based estimate.

CLI:
    python prompt_optimizer.py                 launch the GUI
    python prompt_optimizer.py --selftest      run the rule-engine test suite
    python prompt_optimizer.py --cli "text"    optimize one prompt and print it
        --safe              skip the aggressive rules
        --no-telegraphic    skip the note-style rewrite of imperative lines
"""

from __future__ import annotations

import os
import re
import sys
import traceback
from datetime import datetime

# ── Token counting ─────────────────────────────────────────────────────────────
try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text)) if text else 0

    TOKEN_METHOD = "cl100k exact"
except Exception:  # ImportError, or tiktoken failing to fetch its BPE file
    def count_tokens(text: str) -> int:
        return round(len(text.split()) / 0.75) if text.strip() else 0

    TOKEN_METHOD = "word estimate (install tiktoken for exact)"


# ══════════════════════════════════════════════════════════════════════════════
# Rule-based optimization
# ══════════════════════════════════════════════════════════════════════════════

# Spans that must never be rewritten: fenced code, inline code, quoted text,
# placeholders, URLs and file paths. Exactly ONE capturing group so re.split()
# alternates prose / protected.
#
# Quotes never cross a newline: one unbalanced quote would otherwise swallow
# the rest of the prompt and silently disable every rule after it. Single
# quotes also require a non-word character on each side, so the apostrophes in
# don't and developers' cannot open a span.
_PROTECT_RE = re.compile(
    r"("
    r"```.*?```"                       # fenced code block
    r"|~~~.*?~~~"                      # alternate fence
    r"|`[^`\n]+`"                      # inline code
    r"|\"[^\"\n]*\""                   # "quoted text"
    r"|“[^”\n]*”"                    # curly “quoted text”
    r"|(?<![A-Za-z0-9])'[^'\n]*'(?![A-Za-z0-9])"     # 'quoted text'
    r"|(?<![A-Za-z0-9])‘[^’\n]*’(?![A-Za-z0-9])"   # curly ‘quoted text’
    r"|<[A-Za-z_/][^<>\n]{0,80}>"      # <placeholder> / <html tag>
    r"|\{\{[^}\n]{0,80}\}\}"           # {{template}}
    r"|\$\{[^}\n]{0,80}\}"             # ${var}
    r"|\{[A-Za-z_][A-Za-z0-9_]{0,40}\}"  # {name} format placeholder
    r"|https?://\S+"                   # URL
    r"|[A-Za-z]:\\\\?\S+"              # windows path
    r")",
    re.DOTALL,
)

# Whole sentences that carry no instruction value.
_REMOVE_SENTENCES = [
    r"Feel free to (?:ask|reach out)(?: to me)?(?: if you (?:have|need) (?:any )?(?:further )?(?:questions?|clarification|help))?[^.!?\n]*[.!?]",
    r"Let me know if you (?:need|have|want) (?:anything|any(?:thing)?(?:\s+else)?|further (?:questions?|help)|any (?:clarification|questions?|help))[^.!?\n]*[.!?]",
    r"Don'?t hesitate to ask[^.!?\n]*[.!?]",
    r"I'?m here to help[^.!?\n]*[.!?]",
    r"Thank you (?:so much )?for (?:your (?:help|assistance|question|response|time)|(?:helping|asking))[^.!?\n]*[.!?]",
    r"I appreciate (?:your (?:help|assistance|patience|time)|it)[^.!?\n]*[.!?]",
    r"(?:Try|Do) your (?:very )?best to[^.!?\n]*[.!?]",
    r"I hope (?:this helps|you (?:can help|understand))[^.!?\n]*[.!?]",
    r"Please (?:keep in mind|remember|note|be aware) that you are an AI[^.!?\n]*[.!?]",
    r"You are a (?:helpful|smart|knowledgeable|intelligent|advanced|powerful) (?:AI|assistant|language model)[^.!?\n]*[.!?]",
    r"As an AI(?: language model)?, you[^.!?\n]*[.!?]",
    r"(?:Your|The) (?:response|answer|reply|output) (?:should|must|needs? to|has to) be (?:clear|concise|accurate|detailed|thorough|helpful|correct|precise)[^.!?\n]*[.!?]",
    r"(?:Your|The) (?:response|answer|reply|output) is (?:as )?(?:clear|concise|accurate|detailed|thorough|helpful|correct|precise)[^.!?\n]*[.!?]",
    r"(?:Make|Please make|Ensure|Please ensure) (?:sure )?(?:your|the) (?:response|answer) (?:is|are)[^.!?\n]*[.!?]",
    r"(?:Make|Please make) your (?:answer|response|reply) as (?:\w+ (?:and \w+ )?)?as possible[^.!?\n]*[.!?]",
    r"(?:Be|Please be|Try to be) as (?:\w+ (?:and \w+ )?)?as possible[^.!?\n]*[.!?]",
    r"(?:Write|Give|Provide)(?: your response| your answer)? in a clear and concise (?:manner|way|format)[^.!?\n]*[.!?]",
]

# Applied only at the very start of the prompt.
_STRIP_OPENERS = [
    r"^(?:hi|hello|hey|good (?:morning|afternoon|evening|day)|greetings|howdy)"
    r"(?:\s+(?:there|everyone|all|team))?\b[,!.]*\s*",
    r"^(?:Could|Can|Would|Will) you (?:please )?",
    r"^I(?:'d| would) like you to\s+",
    r"^I want you to\s+",
    r"^I need you to\s+",
    r"^Your task is to\s+",
    r"^Your job is to\s+",
    r"^You are (?:going|expected) to\s+",
]

# Applied at the end of any line.
_STRIP_CLOSERS = [
    r",? if possible\.?(?=$)",
    r",? as best (?:as )?you can\.?(?=$)",
    r",? to the best of your ability\.?(?=$)",
    r",? as (?:accurately|clearly|concisely|clear and concise) as possible\.?(?=$)",
    r",? in a clear and concise (?:manner|way)\.?(?=$)",
    r",? (?:and )?make sure to be (?:thorough|detailed|accurate)\.?(?=$)",
    r",? (?:and )?be as (?:detailed|thorough|helpful) as possible\.?(?=$)",
]

_INLINE_FILLERS = [
    (r"\bplease\b[ \t]*,?[ \t]*", "please"),
    (r"\bkindly\b[ \t]*", "kindly"),
    (r"\b(?:Please )?[Nn]ote that\b[ \t]*", "note that"),
    (r"\b(?:Please )?[Kk]eep in mind that\b[ \t]*", "keep in mind that"),
    (r"\bIt(?:'s| is) (?:important|worth noting) (?:that|to note that)\b[ \t]*", "it is important that"),
    (r"\bJust so you know,?[ \t]*", "just so you know"),
    (r"\bFor your information,?[ \t]*", "for your information"),
    (r"\bFYI,?[ \t]*", "FYI"),
    (r"\bAs an AI(?: language model)?,?[ \t]*", "as an AI"),
    (r"\bAs a helpful assistant,?[ \t]*", "as a helpful assistant"),
]

# ── Phrase compressions ───────────────────────────────────────────────────────
# SAFE: meaning-preserving in essentially every context.
_COMPRESSIONS_SAFE = [
    # Conjunctions / prepositions
    ("in order to",                   "to"),
    ("so as to",                      "to"),
    ("for the purpose of",            "to"),
    ("with the aim of",               "to"),
    ("with the goal of",              "to"),
    ("with a view to",                "to"),
    ("due to the fact that",          "because"),
    ("owing to the fact that",        "because"),
    ("in light of the fact that",     "because"),
    ("for the reason that",           "because"),
    ("in the event that",             "if"),
    ("at this point in time",         "now"),
    ("at the present time",           "now"),
    ("at the present moment",         "now"),
    ("in the near future",            "soon"),
    ("in the not too distant future", "soon"),
    ("prior to",                      "before"),
    ("subsequent to",                 "after"),
    ("in spite of the fact that",     "although"),
    ("despite the fact that",         "although"),
    ("regardless of the fact that",   "although"),
    ("in addition to this",           "also"),
    ("in addition",                   "also"),
    ("furthermore",                   "also"),
    ("moreover",                      "also"),
    ("additionally",                  "also"),
    ("consequently",                  "so"),
    ("therefore",                     "so"),
    ("hence",                         "so"),
    ("on the other hand",             "but"),
    ("nevertheless",                  "but"),
    # Verb phrases
    ("come to the conclusion",        "conclude"),
    ("reach a conclusion",            "conclude"),
    ("make a decision",               "decide"),
    ("make an attempt to",            "try to"),
    ("make use of",                   "use"),
    ("take into consideration",       "consider"),
    ("take into account",             "consider"),
    ("give consideration to",         "consider"),
    ("provide assistance",            "help"),
    ("provide a summary of",          "summarize"),
    ("provide an explanation of",     "explain"),
    ("provide information about",     "describe"),
    ("provide a description of",      "describe"),
    ("give an explanation of",        "explain"),
    ("give a description of",         "describe"),
    ("give a summary of",             "summarize"),
    ("provide a list of",             "list"),
    ("give a list of",                "list"),
    ("is able to",                    "can"),
    ("are able to",                   "can"),
    ("was able to",                   "could"),
    ("were able to",                  "could"),
    ("has the ability to",            "can"),
    ("have the ability to",           "can"),
    # Noun phrases
    ("the majority of",               "most"),
    ("a large number of",             "many"),
    ("a small number of",             "few"),
    ("a number of",                   "some"),
    ("the fact that",                 "that"),
    ("the reason why",                "why"),
    ("the way in which",              "how"),
    ("the extent to which",           "how much"),
    ("with regard to",                "about"),
    ("with respect to",               "about"),
    ("in relation to",                "about"),
    ("pertaining to",                 "about"),
    ("in the context of",             "in"),
    ("by means of",                   "by"),
    ("by virtue of",                  "by"),
    ("on the basis of",               "based on"),
    ("with the exception of",         "except"),
    ("with the help of",              "using"),
    ("with the use of",               "using"),
    ("in the process of",             "while"),
    ("for the sake of",               "for"),
    # Quantifiers
    ("all of the",                    "all the"),
    ("all of my",                     "my"),
    ("all of your",                   "your"),
    ("some of the",                   "some"),
    ("each and every",                "every"),
    ("each of the",                   "each"),
    ("any and all",                   "all"),
    # File-extension tautologies
    ("txt file",                      "txt"),
    ("pdf file",                      "pdf"),
    ("csv file",                      "csv"),
    ("json file",                     "json"),
    ("xml file",                      "xml"),
    ("html file",                     "html"),
    ("markdown file",                 "markdown"),
    ("yaml file",                     "yaml"),
    ("toml file",                     "toml"),
    ("log file",                      "log"),
    # Tautologies
    ("the end result",                "the result"),
    ("a true fact",                   "a fact"),
    ("added bonus",                   "bonus"),
    ("advance planning",              "planning"),
    ("close proximity",               "proximity"),
    ("past history",                  "history"),
    ("sum total",                     "total"),
    ("current status",                "status"),
    ("new innovation",                "innovation"),
    ("very unique",                   "unique"),
    ("completely unique",             "unique"),
    ("absolutely essential",          "essential"),
    ("completely finished",           "finished"),
    ("end result",                    "result"),
    ("future plans",                  "plans"),
    ("past experience",               "experience"),
    ("unexpected surprise",           "surprise"),
    ("free gift",                     "gift"),
]

# AGGRESSIVE: shorter, but can shift nuance. Toggleable in the UI.
_COMPRESSIONS_AGGRESSIVE = [
    ("thus",                          "so"),
    ("in terms of",                   "in"),
    # Relative clause -> preposition
    ("that will have",                "with"),
    ("that would have",               "with"),
    ("that will contain",             "with"),
    ("that contains",                 "with"),
    ("that will include",             "with"),
    ("that includes",                 "with"),
    ("which will have",               "with"),
    ("which contains",                "with"),
    ("which includes",                "with"),
    ("a file that contains",          "a file with"),
    ("a file that has",               "a file with"),
    ("a file that includes",          "a file with"),
    # "that is/are VERB"
    ("that is used to",               "used to"),
    ("that are used to",              "used to"),
    ("that is designed to",           "designed to"),
    ("that are designed to",          "designed to"),
    ("that is able to",               "able to"),
    ("that are able to",              "able to"),
    ("that is needed to",             "to"),
    ("that are needed to",            "to"),
    ("that is required to",           "to"),
    ("that are required to",          "to"),
    ("that needs to",                 "to"),
    ("that need to",                  "to"),
    ("that will be used to",          "to"),
    ("that can be used to",           "to"),
    # List phrasing
    ("a list of all the",             "all the"),
    ("a list of all",                 "all"),
    ("a list of the",                 "the"),
]

# Adverbs that almost never carry instruction weight.
_WEAK_ADVERBS_SAFE = [
    "basically", "essentially", "fundamentally", "literally", "truly",
]
# These can be load-bearing ("just the headers", "completely rewrite").
_WEAK_ADVERBS_AGGRESSIVE = [
    "actually", "absolutely", "completely", "totally", "entirely", "simply", "just",
]

# ── Telegraphic mode ──────────────────────────────────────────────────────────
# Turns a full-sentence instruction into a note-style directive:
#   "Create a txt on the desktop with the ingredients of a good prompt"
#   -> "Create desktop txt: ingredients of a good prompt"
# Three moves: fold a place phrase in front of the thing it holds, replace the
# content clause with a colon, and drop the article after the leading verb.
# Lossy by design, so every rule only fires on a line that actually looks like
# an imperative instruction, and the whole pass sits behind its own toggle.

_IMPERATIVES = [
    "add", "analyse", "analyze", "build", "compile", "compose", "convert",
    "create", "describe", "design", "draft", "explain", "export", "extract",
    "generate", "give", "implement", "list", "make", "output", "prepare",
    "produce", "provide", "refactor", "return", "review", "rewrite", "save",
    "send", "show", "summarise", "summarize", "translate", "update", "write",
]

# Place-like nouns that may move in front of the thing they hold. A whitelist,
# because "on"/"in" are also topic prepositions: "a report on the economy"
# must not become "economy report".
_LOCATIONS = [
    "clipboard", "cloud", "dashboard", "database", "desktop", "directory",
    "disk", "documents", "downloads", "drive", "folder", "home", "homepage",
    "project", "readme", "repo", "repository", "root", "sandbox", "server",
    "sidebar", "wiki", "workspace",
]

# A head containing one of these is a clause, not a thing, and must not be
# pulled in front of a location.
_NP_STOP = (r"(?:of|and|or|for|to|with|in|on|at|that|which|from|by|as|is|are"
            r"|it|one|ones|this|these|those)\b")

_VERB_ALT = "|".join(_IMPERATIVES)
_LOC_ALT = "|".join(_LOCATIONS)
_BULLET = r"(?:[-*\u2022][ \t]+|\d+[.)][ \t]+)?"

# "a txt on the desktop" -> "desktop txt"
_TELE_LOCATIVE_RE = re.compile(
    r"\b(?:a|an|the|my|your|our)[ \t]+"
    r"((?!" + _NP_STOP + r")[\w.\-]+(?:[ \t]+(?!" + _NP_STOP + r")[\w.\-]+){0,2})"
    r"[ \t]+(?:on|in|at|inside|within|under)[ \t]+"
    # "the downloads folder" is the same place as "downloads": the generic
    # tail would otherwise be stranded behind ("downloads pdf folder").
    r"(?:the|my|your|our)[ \t]+(" + _LOC_ALT + r")(?:[ \t]+(?:folder|directory|dir))?\b",
    re.IGNORECASE,
)

# "Create X with the Y" -> "Create X: Y". The head is greedy, so the LAST
# connector on the line becomes the colon, and a line that already carries a
# colon is left alone.
_TELE_COLON_RE = re.compile(
    r"(?:^|(?<=[.!?]))"
    r"([ \t]*" + _BULLET + r"(?:" + _VERB_ALT + r")\b[ \t]+[^.:;!?\n]*[\w)\]\"'])"
    r"[ \t]+(?:with|containing|including|listing"
    r"|(?:that|which)[ \t]+(?:contains?|includes?|lists?|has|have))"
    # An article carries its own weight: with it one word of content is enough,
    # without it two, so a bare "with pandas" stays a phrase.
    r"(?:[ \t]+(?:all[ \t]+the|all|the|a|an)[ \t]+(?=\S)|[ \t]+(?=\S+[ \t]+\S))"
    r"(?![^\n]*:)",
    re.IGNORECASE | re.MULTILINE,
)

# "Write a summary ..." -> "Write summary ..."
_TELE_ARTICLE_RE = re.compile(
    r"(?:^|(?<=[.!?]))"
    r"([ \t]*" + _BULLET + r"(?:" + _VERB_ALT + r")\b)[ \t]+(?:a|an|the)[ \t]+(?=\w)",
    re.IGNORECASE | re.MULTILINE,
)


_ABBREVIATIONS = {
    "e.g.", "i.e.", "etc.", "vs.", "cf.", "approx.", "no.", "fig.", "eq.",
    "al.", "mr.", "mrs.", "ms.", "dr.", "st.", "inc.", "ltd.", "jr.", "sr.",
}


# ── Compiled patterns ─────────────────────────────────────────────────────────

def _build_phrase_regex(pairs):
    """One alternation for all phrases, longest first so the longest wins."""
    if not pairs:
        return None, {}
    mapping = {src.lower(): dst for src, dst in pairs}
    ordered = sorted(mapping, key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(p) for p in ordered) + r")\b"
    return re.compile(pattern, re.IGNORECASE), mapping


def _build_word_regex(words):
    if not words:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b[ \t]*",
                      re.IGNORECASE)


_REMOVE_SENTENCES_RE = [re.compile(p, re.IGNORECASE) for p in _REMOVE_SENTENCES]
_STRIP_OPENERS_RE = [re.compile(p, re.IGNORECASE) for p in _STRIP_OPENERS]
_STRIP_CLOSERS_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _STRIP_CLOSERS]
_INLINE_FILLERS_RE = [(re.compile(p, re.IGNORECASE), label) for p, label in _INLINE_FILLERS]

_COMPRESS_SAFE_RE, _COMPRESS_SAFE_MAP = _build_phrase_regex(_COMPRESSIONS_SAFE)
_COMPRESS_ALL_RE, _COMPRESS_ALL_MAP = _build_phrase_regex(
    _COMPRESSIONS_SAFE + _COMPRESSIONS_AGGRESSIVE)

_ADVERBS_SAFE_RE = _build_word_regex(_WEAK_ADVERBS_SAFE)
_ADVERBS_ALL_RE = _build_word_regex(_WEAK_ADVERBS_SAFE + _WEAK_ADVERBS_AGGRESSIVE)

# "a script that will parse logs" -> "a script to parse logs".
# Requires a determiner+noun before it so "I know that will work" is left alone.
_THAT_WILL_RE = re.compile(
    r"\b(?<!know )(?<!think )(?<!hope )(?<!said )(?<!sure )"
    r"that (?:will|would)\s+(?!have\b|contain\b|include\b|be\b)(?=[a-z])",
    re.IGNORECASE,
)

# Only after real sentence-ending punctuation — never at an arbitrary line start,
# which would capitalize wrapped lines and code that follows a protected span.
_SENT_START_RE = re.compile(r"((?<=[.!?])[ \t\n]+)([a-z])")


# ── Protected-span plumbing ───────────────────────────────────────────────────

def _over_prose(text, fn):
    """Apply fn(str) -> (str, changes) to prose only, leaving protected spans."""
    parts = _PROTECT_RE.split(text)
    changes = []
    for i in range(0, len(parts), 2):          # even indices are prose
        parts[i], ch = fn(parts[i])
        changes.extend(ch)
    return "".join(parts), changes


def _ellipsis(s, n):
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _match_case(src: str, dst: str) -> str:
    """Carry the source phrase's leading capitalization onto the replacement."""
    if src[:1].isupper():
        return dst[:1].upper() + dst[1:]
    return dst


# ── Individual passes ─────────────────────────────────────────────────────────

def _remove_sentences(text):
    changes = []
    for pat in _REMOVE_SENTENCES_RE:
        hits = []

        def _rep(m, _hits=hits):
            _hits.append(m.group())
            return " "

        new = pat.sub(_rep, text)
        if hits:
            extra = f"  (+{len(hits) - 1} more)" if len(hits) > 1 else ""
            changes.append(f'Removed: "{_ellipsis(hits[0], 70)}"{extra}')
            text = new
    return text, changes


def _strip_openers(text):
    changes = []
    stripped = text.lstrip()
    lead = text[: len(text) - len(stripped)]
    text = stripped
    for _ in range(3):                          # "Hi, could you please ..."
        before = text
        for pat in _STRIP_OPENERS_RE:
            new = pat.sub("", text, count=1)
            if new != text:
                changes.append("Stripped opener")
                text = new.lstrip()
                break
        if text == before:
            break
    return lead + text, changes


def _strip_closers(text):
    changes = []
    for pat in _STRIP_CLOSERS_RE:
        new = pat.sub(".", text)
        if new != text:
            changes.append("Stripped trailing filler")
            text = new
    return text, changes


def _remove_inline_fillers(text):
    changes = []
    for pat, label in _INLINE_FILLERS_RE:
        new, n = pat.subn("", text)
        if n:
            changes.append(f'Removed filler: "{label}"' + (f"  (x{n})" if n > 1 else ""))
            text = new
    return text, changes


def _apply_compressions(text, aggressive):
    rx = _COMPRESS_ALL_RE if aggressive else _COMPRESS_SAFE_RE
    mapping = _COMPRESS_ALL_MAP if aggressive else _COMPRESS_SAFE_MAP
    if rx is None:
        return text, []

    seen: dict[str, int] = {}

    def _rep(m):
        src = m.group(1)
        key = src.lower()
        seen[key] = seen.get(key, 0) + 1
        return _match_case(src, mapping[key])

    new = rx.sub(_rep, text)
    changes = [
        f'"{k}" -> "{mapping[k]}"' + (f"  (x{n})" if n > 1 else "")
        for k, n in seen.items()
    ]
    return new, changes


def _rewrite_relative_clauses(text):
    new, n = _THAT_WILL_RE.subn("to ", text)
    if n:
        return new, ['"that will/would VERB" -> "to VERB"' + (f"  (x{n})" if n > 1 else "")]
    return text, []


def _remove_weak_adverbs(text, aggressive):
    rx = _ADVERBS_ALL_RE if aggressive else _ADVERBS_SAFE_RE
    if rx is None:
        return text, []
    hits: dict[str, int] = {}

    def _rep(m):
        w = m.group().strip().lower()
        hits[w] = hits.get(w, 0) + 1
        return ""

    new = rx.sub(_rep, text)
    changes = [f'Removed weak adverb: "{w}"' + (f"  (x{n})" if n > 1 else "")
               for w, n in hits.items()]
    return new, changes


def _telegraphize(text):
    """Note-style rewrite of imperative lines. Only runs in telegraphic mode."""
    changes = []

    def _loc(m):
        return f"{m.group(2).lower()} {m.group(1)}"

    new, n = _TELE_LOCATIVE_RE.subn(_loc, text)
    if n:
        changes.append('"<thing> on the <place>" -> "<place> <thing>"'
                       + (f"  (x{n})" if n > 1 else ""))
        text = new

    new, n = _TELE_COLON_RE.subn(r"\1: ", text)
    if n:
        changes.append('Content clause -> colon' + (f"  (x{n})" if n > 1 else ""))
        text = new

    new, n = _TELE_ARTICLE_RE.subn(r"\1 ", text)
    if n:
        changes.append("Dropped article after leading verb"
                       + (f"  (x{n})" if n > 1 else ""))
        text = new

    return text, changes


# ── Cleanup (structure preserving) ────────────────────────────────────────────

def _fix_capitalization(chunk: str) -> str:
    def _rep(m):
        head = chunk[: m.start()].rstrip()
        tail = head[-10:].split()
        if tail and tail[-1].lower() in _ABBREVIATIONS:
            return m.group(0)                   # "e.g. json" stays lowercase
        return m.group(1) + m.group(2).upper()

    return _SENT_START_RE.sub(_rep, chunk)


def _cleanup(text: str) -> str:
    def _prose(chunk, at_line_start):
        chunk = re.sub(r"[ \t]+", " ", chunk)            # spaces only; keep \n
        chunk = re.sub(r"[ \t]*\n", "\n", chunk)         # trailing space per line
        chunk = re.sub(r"\n{3,}", "\n\n", chunk)         # cap blank runs
        chunk = re.sub(r" +([.,!?;:])", r"\1", chunk)    # " ." -> "."
        chunk = re.sub(r"([,;:])[ \t]*([.!?])", r"\2", chunk)   # ",." -> "."
        chunk = re.sub(r"([.!?])[ \t]*\1+", r"\1", chunk)       # ".." -> "."
        # Punctuation is only orphaned at a real line start. A chunk that merely
        # follows a protected span is mid-sentence, and a leading "." there is
        # that sentence's full stop: Say "done". has to keep it.
        head = r"(^|\n)" if at_line_start else r"(\n)"
        chunk = re.sub(head + r"[ \t]*([,;:.])[ \t]*", r"\1", chunk)
        return _fix_capitalization(chunk)

    parts = _PROTECT_RE.split(text)
    for i in range(0, len(parts), 2):           # even indices are prose
        parts[i] = _prose(parts[i], i == 0 or parts[i - 1].endswith("\n"))
    text = "".join(parts).strip()
    if text[:1].islower():                      # opener stripping can lowercase it
        text = text[0].upper() + text[1:]
    return text


# ── Public entry point ────────────────────────────────────────────────────────

def optimize_prompt(text: str, aggressive: bool = True, telegraphic: bool = False):
    """Rule-based pass. Returns (optimized, changes, orig_tokens, new_tokens).

    `telegraphic` adds the note-style rewrite of imperative lines ("Create a txt
    on the desktop with the ingredients" -> "Create desktop txt: ingredients").
    It is lossy — it drops articles and turns a clause into a colon — so it is a
    separate opt-in from `aggressive`."""
    original_tokens = count_tokens(text)
    result = text
    all_changes: list[str] = []

    for step in (_remove_sentences, _strip_openers, _strip_closers,
                 _remove_inline_fillers):
        result, ch = _over_prose(result, step)
        all_changes.extend(ch)

    for _ in range(4):                          # iterate until stable
        before = result

        result, ch = _over_prose(result, lambda t: _apply_compressions(t, aggressive))
        all_changes.extend(ch)

        if aggressive:
            result, ch = _over_prose(result, _rewrite_relative_clauses)
            all_changes.extend(ch)

        result, ch = _over_prose(result, lambda t: _remove_weak_adverbs(t, aggressive))
        all_changes.extend(ch)

        if telegraphic:
            result, ch = _over_prose(result, _telegraphize)
            all_changes.extend(ch)

        if result == before:
            break

    result = _cleanup(result)

    # De-duplicate log lines while preserving order.
    seen = set()
    changes = [c for c in all_changes if not (c in seen or seen.add(c))]
    return result, changes, original_tokens, count_tokens(result)



# ══════════════════════════════════════════════════════════════════════════════
# Word-level diff
# ══════════════════════════════════════════════════════════════════════════════

# Words, punctuation and whitespace as separate tokens, so a diff lands on word
# boundaries instead of mid-word.
_DIFF_TOKEN_RE = re.compile(r"\s+|\w+|[^\w\s]")


def _diff_spans(before: str, after: str):
    """Character ranges that were cut from `before` and added in `after`.

    Returns (removed, added) as lists of (start, end) offsets. Comparison is
    case-insensitive so the cleanup pass re-capitalizing a sentence does not
    show up as a change.
    """
    from difflib import SequenceMatcher

    a = [(m.group(), m.start(), m.end()) for m in _DIFF_TOKEN_RE.finditer(before)]
    b = [(m.group(), m.start(), m.end()) for m in _DIFF_TOKEN_RE.finditer(after)]
    sm = SequenceMatcher(None,
                         [t[0].lower() for t in a],
                         [t[0].lower() for t in b],
                         autojunk=False)

    def collapse(toks, i1, i2, src):
        """Merge a token run into one span, ignoring runs that are only spaces."""
        if i1 >= i2:
            return None
        start, end = toks[i1][1], toks[i2 - 1][2]
        if not src[start:end].strip():
            return None
        return start, end

    removed, added = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("delete", "replace"):
            span = collapse(a, i1, i2, before)
            if span:
                removed.append(span)
        if tag in ("insert", "replace"):
            span = collapse(b, j1, j2, after)
            if span:
                added.append(span)
    return removed, added


# ══════════════════════════════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════════════════════════════

C = {
    "bg":      "#1e1e2e",
    "surface": "#313244",
    "raised":  "#45475a",
    "primary": "#89b4fa",
    "green":   "#a6e3a1",
    "red":     "#f38ba8",
    "yellow":  "#f9e2af",
    "text":    "#cdd6f4",
    "sub":     "#a6adc8",
}


def _enable_dpi_awareness():
    """Without this Windows bitmap-stretches the window on scaled displays,
    which blurs the text and makes the requested geometry overshoot the screen."""
    if os.name != "nt":
        return
    import ctypes
    for call in (lambda: ctypes.windll.shcore.SetProcessDpiAwareness(1),
                 lambda: ctypes.windll.user32.SetProcessDPIAware()):
        try:
            call()
            return
        except Exception:
            continue


def _run_gui():
    _enable_dpi_awareness()

    import tkinter as tk
    from tkinter import messagebox

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Prompt Optimizer")
            self.minsize(880, 620)
            self._fit_geometry(1180, 820)
            self.configure(bg=C["bg"])

            self.history: list[dict] = []
            self._count_job = None
            self._diff_basis = None

            self._build()
            self._bind_keys()

        def _fit_geometry(self, want_w, want_h):
            """Never open larger than the screen — the log panel used to fall
            off the bottom on scaled displays."""
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            w = max(880, min(want_w, sw - 80))
            h = max(620, min(want_h, sh - 130))
            self.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")

        # ── layout ───────────────────────────────────────────────────────────

        def _build(self):
            hdr = tk.Frame(self, bg=C["bg"])
            hdr.pack(fill="x", padx=20, pady=(14, 6))
            tk.Label(hdr, text="Prompt Optimizer", font=("Segoe UI", 20, "bold"),
                     bg=C["bg"], fg=C["primary"]).pack(side="left")
            tk.Label(hdr, text=f"  tokens: {TOKEN_METHOD}", font=("Segoe UI", 9),
                     bg=C["bg"], fg=C["sub"]).pack(side="left", padx=6)

            def toggle(text, var, cmd):
                tk.Checkbutton(
                    hdr, text=text, variable=var, command=cmd,
                    bg=C["bg"], fg=C["text"], selectcolor=C["surface"],
                    activebackground=C["bg"], activeforeground=C["text"],
                    font=("Segoe UI", 9), bd=0, highlightthickness=0,
                ).pack(side="right", padx=(10, 0))

            self.aggressive = tk.BooleanVar(value=True)
            self.telegraphic = tk.BooleanVar(value=True)
            self.show_diff = tk.BooleanVar(value=True)
            toggle("Aggressive rules", self.aggressive, self._on_rules_toggle)
            toggle("Telegraphic", self.telegraphic, self._on_rules_toggle)
            toggle("Show diff", self.show_diff, self._on_diff_toggle)

            panels = tk.Frame(self, bg=C["bg"])
            panels.pack(fill="both", expand=True, padx=20, pady=4)
            panels.grid_columnconfigure(0, weight=1, uniform="p")
            panels.grid_columnconfigure(1, weight=1, uniform="p")
            panels.grid_rowconfigure(0, weight=1)

            self.input_text, self.lbl_in = self._panel(panels, 0, "Original Prompt")
            self.output_text, self.lbl_out = self._panel(panels, 1, "Optimized Prompt  (editable)")

            btn_row = tk.Frame(self, bg=C["bg"])
            btn_row.pack(fill="x", padx=20, pady=6)
            self._btn(btn_row, "Optimize  Ctrl+Enter", self._optimize, C["primary"]).pack(side="left", padx=(0, 8))
            self._btn(btn_row, "Copy Result", self._copy, C["green"]).pack(side="left", padx=(0, 8))
            self._btn(btn_row, "Clear  Ctrl+K", self._clear, C["raised"]).pack(side="left", padx=(0, 8))
            self._btn(btn_row, "History", self._history, C["raised"]).pack(side="left")

            self.lbl_savings = tk.Label(btn_row, text="", font=("Segoe UI", 12, "bold"),
                                        bg=C["bg"], fg=C["green"])
            self.lbl_savings.pack(side="right")

            # ── changes log ──────────────────────────────────────────────────
            log_frame = tk.Frame(self, bg=C["surface"])
            log_frame.pack(fill="x", padx=20, pady=(0, 16))
            tk.Label(log_frame, text="Changes applied:", font=("Segoe UI", 9, "bold"),
                     bg=C["surface"], fg=C["sub"], anchor="w", padx=10, pady=4).pack(fill="x")

            log_body = tk.Frame(log_frame, bg=C["surface"])
            log_body.pack(fill="x")
            self.log = tk.Text(log_body, height=5, width=40, bg=C["surface"], fg=C["text"],
                               font=("Consolas", 9), relief="flat", padx=10, pady=4,
                               state="disabled", wrap="word", highlightthickness=0)
            log_sb = tk.Scrollbar(log_body, command=self.log.yview)
            self.log.config(yscrollcommand=log_sb.set)
            log_sb.pack(side="right", fill="y")
            self.log.pack(side="left", fill="both", expand=True)

        def _panel(self, parent, col, title):
            frame = tk.Frame(parent, bg=C["surface"])
            frame.grid(row=0, column=col, sticky="nsew",
                       padx=(0, 8) if col == 0 else (0, 0), pady=(0, 8))

            top = tk.Frame(frame, bg=C["surface"])
            top.pack(fill="x", padx=10, pady=(8, 2))
            tk.Label(top, text=title, font=("Segoe UI", 11, "bold"),
                     bg=C["surface"], fg=C["text"]).pack(side="left")
            lbl = tk.Label(top, text="0 tokens", font=("Segoe UI", 9),
                           bg=C["surface"], fg=C["sub"])
            lbl.pack(side="right")

            body = tk.Frame(frame, bg=C["surface"])
            body.pack(fill="both", expand=True, padx=2, pady=(0, 8))
            # Small requested height + expand=True: the panels grow into
            # whatever is left instead of pushing the lower rows off-window.
            txt = tk.Text(body, bg=C["bg"], fg=C["text"], insertbackground=C["text"],
                          font=("Segoe UI", 11), relief="flat", padx=10, pady=8,
                          wrap="word", selectbackground=C["raised"],
                          undo=True, highlightthickness=0, height=8, width=40)
            sb = tk.Scrollbar(body, command=txt.yview)
            txt.config(yscrollcommand=sb.set)
            sb.pack(side="right", fill="y")
            txt.pack(side="left", fill="both", expand=True)

            # Diff highlighting: struck-through red for what the rules cut,
            # green for what they introduced.
            txt.tag_configure("cut", foreground=C["red"], overstrike=True)
            txt.tag_configure("new", foreground=C["green"])
            return txt, lbl

        def _btn(self, parent, label, cmd, color):
            # Dark buttons need light text; the accent buttons need dark text.
            dark = color in (C["raised"], C["surface"], C["bg"])
            return tk.Button(parent, text=label, command=cmd, bg=color,
                             fg=C["text"] if dark else C["bg"],
                             font=("Segoe UI", 10, "bold"), relief="flat",
                             padx=12, pady=6, cursor="hand2",
                             activebackground=C["primary"] if dark else C["raised"],
                             activeforeground=C["bg"] if dark else C["text"],
                             disabledforeground=C["sub"], bd=0, highlightthickness=0)

        # ── key bindings ─────────────────────────────────────────────────────

        def _bind_keys(self):
            def handler(fn):
                def _h(_event=None):
                    fn()
                    return "break"              # suppress Text's own binding
                return _h

            self.bind_all("<Control-Return>", handler(self._optimize))
            self.bind_all("<Control-k>", handler(self._clear))
            self.bind_all("<Control-K>", handler(self._clear))

            def select_all(event):
                event.widget.tag_add("sel", "1.0", "end-1c")
                event.widget.mark_set("insert", "1.0")
                return "break"

            for w in (self.input_text, self.output_text):
                w.bind("<Control-a>", select_all)
                w.bind("<Control-A>", select_all)

            self.input_text.bind("<KeyRelease>", self._schedule_count)
            self.output_text.bind("<KeyRelease>", self._schedule_count)

        # ── token counting (debounced) ───────────────────────────────────────

        def _schedule_count(self, _=None):
            # Editing either panel invalidates the highlighting, which is
            # anchored to character offsets that have just shifted. Compare
            # against the text the diff was built from: KeyRelease also fires
            # for the Ctrl+Enter that just ran the optimize, and that must not
            # wipe the highlighting it produced.
            if self._diff_basis != (self.input_text.get("1.0", "end-1c"),
                                    self.output_text.get("1.0", "end-1c")):
                self._clear_diff()
            if self._count_job is not None:
                self.after_cancel(self._count_job)
            self._count_job = self.after(200, self._update_counts)

        def _update_counts(self):
            self._count_job = None
            self.lbl_in.config(text=f"{count_tokens(self._get_in()):,} tokens")
            self.lbl_out.config(text=f"{count_tokens(self._get_out()):,} tokens")

        def _get_in(self):
            return self.input_text.get("1.0", "end-1c").strip()

        def _get_out(self):
            return self.output_text.get("1.0", "end-1c").strip()

        def _on_rules_toggle(self):
            if self._get_in():
                self._optimize()

        # ── diff highlighting ────────────────────────────────────────────────

        def _on_diff_toggle(self):
            if self.show_diff.get():
                self._apply_diff()
            else:
                self._clear_diff()

        def _clear_diff(self):
            self._diff_basis = None
            for w in (self.input_text, self.output_text):
                for tag in ("cut", "new"):
                    w.tag_remove(tag, "1.0", "end")

        def _apply_diff(self):
            """Strike through what was removed, highlight what was added."""
            self._clear_diff()
            if not self.show_diff.get():
                return
            # Raw widget text, NOT the stripped accessors — the offsets returned
            # by _diff_spans are used directly as tkinter indices.
            before = self.input_text.get("1.0", "end-1c")
            after = self.output_text.get("1.0", "end-1c")
            if not before.strip() or not after.strip():
                return
            try:
                removed, added = _diff_spans(before, after)
            except Exception:
                traceback.print_exc()
                return
            for widget, spans, tag in ((self.input_text, removed, "cut"),
                                       (self.output_text, added, "new")):
                for start, end in spans:
                    widget.tag_add(tag,
                                   f"1.0 + {start} chars",
                                   f"1.0 + {end} chars")
            self._diff_basis = (before, after)

        # ── main actions ─────────────────────────────────────────────────────

        def _optimize(self):
            original = self._get_in()
            if not original:
                messagebox.showinfo("Empty", "Paste a prompt to optimize.")
                return

            try:
                optimized, changes, orig_tok, rules_tok = optimize_prompt(
                    original, aggressive=self.aggressive.get(),
                    telegraphic=self.telegraphic.get())
            except Exception as exc:
                traceback.print_exc()
                messagebox.showerror("Optimize failed", f"{type(exc).__name__}: {exc}")
                return

            self._set_output(optimized)
            self.lbl_in.config(text=f"{orig_tok:,} tokens")
            self.lbl_out.config(text=f"{rules_tok:,} tokens")

            saved = orig_tok - rules_tok
            if saved > 0:
                pct = saved / orig_tok * 100 if orig_tok else 0
                self.lbl_savings.config(text=f"Rules saved {saved} tok ({pct:.1f}%)",
                                        fg=C["green"])
            else:
                self.lbl_savings.config(text="Already concise — no changes", fg=C["sub"])

            self._write_log(changes)
            self._apply_diff()
            self._push_history(optimized, orig_tok, rules_tok)

        def _push_history(self, optimized, orig_tok, new_tok):
            self.history.append(dict(
                time=datetime.now().strftime("%H:%M:%S"),
                original=self._get_in(), optimized=optimized,
                orig_tok=orig_tok, new_tok=new_tok,
            ))

        def _set_output(self, text):
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", text)

        def _write_log(self, changes):
            self.log.config(state="normal")
            self.log.delete("1.0", "end")
            if changes:
                self.log.insert("end", "\n".join(f"• {c}" for c in changes))
            else:
                self.log.insert("end", "No changes — prompt is already concise.")
            self.log.config(state="disabled")

        def _copy(self):
            text = self._get_out()
            if not text:
                messagebox.showinfo("Nothing to copy", "Run Optimize first.")
                return
            self.clipboard_clear()
            self.clipboard_append(text)
            self.update()                       # keep clipboard after exit
            self.lbl_savings.config(text="Copied to clipboard", fg=C["green"])

        def _clear(self):
            self._clear_diff()
            self.input_text.delete("1.0", "end")
            self.output_text.delete("1.0", "end")
            self.lbl_in.config(text="0 tokens")
            self.lbl_out.config(text="0 tokens")
            self.lbl_savings.config(text="")
            self._write_log([])

        def _history(self):
            if not self.history:
                messagebox.showinfo("History", "No optimizations yet this session.")
                return
            win = tk.Toplevel(self)
            win.title("History")
            win.geometry("780x460")
            win.configure(bg=C["bg"])
            txt = tk.Text(win, bg=C["bg"], fg=C["text"], font=("Consolas", 10),
                          relief="flat", padx=12, pady=10, wrap="word",
                          highlightthickness=0)
            sb = tk.Scrollbar(win, command=txt.yview)
            txt.config(yscrollcommand=sb.set)
            sb.pack(side="right", fill="y")
            txt.pack(side="left", fill="both", expand=True)
            for h in reversed(self.history):
                saved = h["orig_tok"] - h["new_tok"]
                txt.insert("end", f"[{h['time']}]  {h['orig_tok']} -> {h['new_tok']} tok "
                                  f"(saved {saved})\n")
                txt.insert("end", f"  Before: {_ellipsis(h['original'], 110)}\n")
                txt.insert("end", f"  After:  {_ellipsis(h['optimized'], 110)}\n\n")
            txt.config(state="disabled")

    App().mainloop()


# ══════════════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════════════

_SELFTESTS = [
    # (input, aggressive, predicate(output) -> bool, description)
    ("Write a file that lists all the users.", True,
     lambda o: "file lists" not in o, "no ungrammatical 'file that' collapse"),

    ("Line one.\nLine two.\n\n- bullet a\n- bullet b", True,
     lambda o: o.count("\n") >= 3, "newlines and bullets preserved"),

    ("Fix this:\n```python\ndef f():\n    return 1  # in order to test\n```\nGo.", True,
     lambda o: "def f():\n    return 1  # in order to test" in o,
     "fenced code block untouched"),

    ("I know that will work. Please confirm.", True,
     lambda o: "know that will work" in o, "'I know that will' not rewritten"),

    ("Just the headers, please.", True,
     lambda o: ",." not in o and o.endswith("."), "no orphan punctuation"),

    ("Use e.g. json. then stop.", True,
     lambda o: "e.g. json" in o, "abbreviation not capitalized"),

    ("Could you please provide a summary of this in order to save time?", True,
     lambda o: o.lower().startswith("summarize"), "opener + compression"),

    ("Therefore, use the API.", True,
     lambda o: o.startswith("So,"), "capitalization carried to replacement"),

    ("Send `please note that` verbatim.", True,
     lambda o: "`please note that`" in o, "inline code untouched"),

    ("Replace <please note that> exactly.", True,
     lambda o: "<please note that>" in o, "placeholder untouched"),

    ("Explain it, if possible.\nSummarize it, if possible.", True,
     lambda o: "if possible" not in o, "closers stripped on every line"),

    ("Fetch https://x.com/in%20order%20to/please here.", True,
     lambda o: "https://x.com/in%20order%20to/please" in o, "URL untouched"),

    ("Read the log file and the json file.", True,
     lambda o: "log file" not in o and "json file" not in o, "extension tautologies"),

    ("Just simply do it.", False,
     lambda o: "Just simply" in o, "safe mode keeps load-bearing adverbs"),

    ("A script that will parse logs.", True,
     lambda o: "script to parse" in o, "that-will rewrite still works"),

    ("Hi, can you please write a txt file that will have all of the "
     "good indicators of a good prompt? Thank you for your help.", True,
     lambda o: o.lower().startswith("write a txt with all the"),
     "end-to-end original failing prompt"),

    # ── quoted text is verbatim ──────────────────────────────────────────────

    ('Replace "please note that" exactly.', True,
     lambda o: '"please note that"' in o, "double-quoted text untouched"),

    ('Use “in order to” verbatim.', True,
     lambda o: "“in order to”" in o, "curly-quoted text untouched"),

    ("Replace 'in order to' exactly.", True,
     lambda o: "'in order to'" in o, "single-quoted text untouched"),

    ('Write "a script that will parse logs" and a script that will parse logs.', True,
     lambda o: '"a script that will parse logs"' in o and "and a script to parse logs" in o,
     "same phrase kept in quotes, compressed outside them"),

    ("Don't please note that it works.", True,
     lambda o: "please note that" not in o, "an apostrophe cannot open a quote"),

    ("The developers' guide says to please note that it works.", True,
     lambda o: "please note that" not in o, "a possessive cannot open a quote"),

    ('Say "hello and please note that it works.', True,
     lambda o: "please note that" not in o, "an unbalanced quote protects nothing"),

    ('First "keep this" then\n"and this" too, in order to test.', True,
     lambda o: '"keep this"' in o and '"and this"' in o and "in order to" not in o,
     "quotes do not cross newlines"),

    ('Output the word "done".', True,
     lambda o: o.endswith('"done".'), "full stop survives after a quote"),

    ("Run the command `ls -la`.", True,
     lambda o: o.endswith("`ls -la`."), "full stop survives after inline code"),
]


# Telegraphic mode only. Same shape as above minus the aggressive flag, which
# these all imply: (input, predicate(output) -> bool, description).
_SELFTESTS_TELEGRAPHIC = [
    ("Hello, I want you to create a txt file on the desktop that will include "
     "the ingredients of a good prompt",
     lambda o: o == "Create desktop txt: ingredients of a good prompt",
     "note-style rewrite of the worked example"),

    ("Write a report on the economy for my boss.",
     lambda o: "on the economy" in o, "'on' as a topic is not a location"),

    ("Write a script with pandas.",
     lambda o: "with pandas" in o, "one-word 'with' phrase is not a clause"),

    ("Make a list with the following: apples, pears.",
     lambda o: o.count(":") == 1, "line that already has a colon is left alone"),

    ("The summary with the key points is attached.",
     lambda o: "with the key points" in o, "non-imperative line is left alone"),

    ("First read the docs. Create a txt on the desktop with the ingredients.",
     lambda o: o.endswith("Create desktop txt: ingredients."),
     "fires on a sentence that starts mid-line"),

    ('Create a file with "the exact words on the desktop" preserved.',
     lambda o: '"the exact words on the desktop"' in o, "quoted text untouched"),

    ("Fix this:\n```python\ndef f():\n    return 1\n```\n"
     "Create a txt on the desktop with the results of the run.",
     lambda o: "def f():\n    return 1" in o and "Create desktop txt: results" in o,
     "code fence untouched, instruction after it rewritten"),

    ("- Create a file with the results\n- Update the readme with the new steps",
     lambda o: o == "- Create file: results\n- Update readme: new steps",
     "every bullet rewritten, list structure kept"),

    ("Save the config in the repo.",
     lambda o: o == "Save repo config.", "location folded in front of the thing"),

    ("Create a pdf file in the downloads folder with the meeting summary.",
     lambda o: o == "Create downloads pdf: meeting summary.",
     "generic 'folder' tail travels with the location"),
]


# Representative real-world prompts. Deliberately mixed: some verbose, some
# already terse. An earlier benchmark repeated one filler-stuffed sentence 200x
# and reported 62% savings, which flattered the engine and made it useless as a
# signal for whether a new rule actually helps.
_BENCH_CORPUS = [
    "Could you please provide a summary of the following text? Please note that it "
    "is important that your response is as clear and concise as possible.",
    "I would like you to write a python script that will parse the log file and "
    "provide a summary of all of the errors.",
    "Rewrite the parse_config function so that it handles all 42 of the edge cases "
    "described in the documentation.",
    "You are a helpful AI assistant. Your task is to take the data provided below "
    "and produce a report with a title, an introduction, three body sections and a "
    "conclusion.",
    "Summarize this article in 3 bullet points.",
    "Explain how a binary search tree works, in order to help me understand the time "
    "complexity. Please make your answer as detailed as possible.",
    "Refactor this code:\n```python\ndef f(a,b):\n    return a+b\n```\n"
    "Keep the behaviour identical and add type hints.",
    "Given the CSV at data/sales.csv, compute the monthly totals for 2025 and output "
    "them as a markdown table sorted by revenue descending.",
    "Act as a senior code reviewer. Review the diff below for correctness bugs. For "
    "each issue, give the file, the line, and a one sentence explanation.",
    "Translate the following paragraph into French, preserving the formal register "
    "and keeping all proper nouns unchanged.",
    "Hi there! I was wondering if you could possibly help me out with something. I "
    "need to understand the difference between a process and a thread.",
    "Write unit tests for the retry_with_backoff helper. Cover the success path, the "
    "exhausted-retries path, and the case where the callable raises a non-retryable "
    "error.",
    "Due to the fact that our API is rate limited, please make use of exponential "
    "backoff in order to avoid hitting the limit. Take into account that the limit "
    "is 100 requests per minute.",
    "extract all email addresses from the text below and return them as a json array",
    "Please note that the deadline is Friday. Keep in mind that the client has "
    "requested a draft in advance. Feel free to ask me if you need any clarification.",
]


def _benchmark():
    """Report honest savings on realistic prompts, plus a throughput figure."""
    import time

    total_a = total_b = total_t = 0
    pcts = []
    for src in _BENCH_CORPUS:
        _, _, a, b = optimize_prompt(src, aggressive=True)
        _, _, _, t = optimize_prompt(src, aggressive=True, telegraphic=True)
        total_a += a
        total_b += b
        total_t += t
        pcts.append((a - b) / a * 100 if a else 0.0)
    pcts.sort()
    median = pcts[len(pcts) // 2]
    zeros = sum(1 for x in pcts if x < 0.5)

    print(f"\ncorpus ({len(_BENCH_CORPUS)} realistic prompts):")
    print(f"  total   {total_a:,} -> {total_b:,} tokens "
          f"({(total_a - total_b) / total_a * 100:.1f}% saved)")
    print(f"  median  {median:.0f}%      best {pcts[-1]:.0f}%      "
          f"unchanged {zeros}/{len(pcts)}")
    print(f"  +tele   {total_a:,} -> {total_t:,} tokens "
          f"({(total_a - total_t) / total_a * 100:.1f}% saved)")

    # Throughput on a large input, so a regression in the regex path is visible.
    big = "\n\n".join(_BENCH_CORPUS) * 40
    t0 = time.perf_counter()
    optimize_prompt(big, aggressive=True)
    dt = time.perf_counter() - t0
    print(f"  speed   {len(big):,} chars in {dt * 1000:.0f} ms")


def _selftest() -> int:
    failed = 0
    checks = 0
    for src, aggr, check, desc in _SELFTESTS:
        out, changes, a, b = optimize_prompt(src, aggressive=aggr)
        ok = False
        try:
            ok = bool(check(out))
        except Exception as exc:
            out = f"{out!r}  <check raised {exc}>"
        checks += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {desc}")
        if not ok:
            failed += 1
            print(f"        in : {src!r}")
            print(f"        out: {out!r}")

    for src, check, desc in _SELFTESTS_TELEGRAPHIC:
        out, _, _, _ = optimize_prompt(src, aggressive=True, telegraphic=True)
        ok = False
        try:
            ok = bool(check(out))
        except Exception as exc:
            out = f"{out!r}  <check raised {exc}>"
        checks += 1
        print(f"[{'PASS' if ok else 'FAIL'}] telegraphic: {desc}")
        if not ok:
            failed += 1
            print(f"        in : {src!r}")
            print(f"        out: {out!r}")

    # Telegraphic mode is opt-in, and a second pass must not drift.
    _tele_src = "Create a txt file on the desktop that will include the steps."
    once, _, _, _ = optimize_prompt(_tele_src, aggressive=True, telegraphic=True)
    twice, _, _, _ = optimize_prompt(once, aggressive=True, telegraphic=True)
    for ok, desc in ((":" not in optimize_prompt(_tele_src, aggressive=True)[0],
                      "telegraphic: off unless asked for"),
                     (once == twice, "telegraphic: stable on a second pass")):
        checks += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {desc}")
        if not ok:
            failed += 1

    # No-crash fuzz over structural edge cases.
    edge = ["", "   ", "\n\n\n", ".", "please", "```\n```", "<>", "a" * 5000,
            "please " * 500, "in order to " * 300, "create ", "write a with the ",
            "create a file on the desktop with " * 200]
    for e in edge:
        try:
            optimize_prompt(e, aggressive=True, telegraphic=True)
            optimize_prompt(e, aggressive=False)
        except Exception as exc:
            failed += 1
            print(f"[FAIL] edge case {e[:20]!r} raised {type(exc).__name__}: {exc}")

    # Diff-span correctness. Offsets must index the ORIGINAL strings exactly,
    # because the GUI feeds them straight to tkinter as character indices.
    diff_cases = [
        ("Please write the report.", "Write the report.", ["Please"], []),
        ("in order to test", "to test", ["in order"], []),
        ("Summarize this.", "Summarize this.", [], []),
        ("the log file here", "the log here", ["file"], []),
    ]
    for before, after, want_cut, want_new in diff_cases:
        removed, added = _diff_spans(before, after)
        got_cut = [before[s:e].strip() for s, e in removed]
        got_new = [after[s:e].strip() for s, e in added]
        ok = got_cut == want_cut and got_new == want_new
        checks += 1
        print(f"[{'PASS' if ok else 'FAIL'}] diff {before!r} -> {after!r}")
        if not ok:
            failed += 1
            print(f"        cut={got_cut!r} (want {want_cut!r})")
            print(f"        new={got_new!r} (want {want_new!r})")

    # Spans must never fall outside the strings they index.
    for src in _BENCH_CORPUS + [c[0] for c in _SELFTESTS_TELEGRAPHIC]:
        out, _, _, _ = optimize_prompt(src, telegraphic=True)
        removed, added = _diff_spans(src, out)
        bad = ([1 for s, e in removed if not (0 <= s < e <= len(src))]
               + [1 for s, e in added if not (0 <= s < e <= len(out))])
        if bad:
            failed += 1
            print(f"[FAIL] diff span out of range for {src[:40]!r}")
    checks += 1
    print("[PASS] diff spans in range across corpus")

    _benchmark()

    print(f"\n{checks - failed}/{checks} checks passed, {failed} failure(s)")
    return 1 if failed else 0


# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        return _selftest()
    if args and args[0] == "--cli":
        rest = args[1:]
        aggressive = "--safe" not in rest
        telegraphic = "--no-telegraphic" not in rest
        rest = [a for a in rest if a not in ("--safe", "--no-telegraphic")]
        text = " ".join(rest) or sys.stdin.read()
        out, changes, a, b = optimize_prompt(
            text, aggressive=aggressive, telegraphic=telegraphic)
        print(out)
        print(f"\n--- {a} -> {b} tokens ({(a - b) / a * 100:.1f}% saved)"
              if a else "", file=sys.stderr)
        return 0
    _run_gui()
    return 0


if __name__ == "__main__":
    sys.exit(main())
