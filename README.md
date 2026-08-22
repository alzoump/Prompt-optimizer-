# Prompt Optimizer

**Cut your prompt token count without cutting quality.**

Prompt Optimizer is a Windows desktop app that rewrites your prompts into leaner versions — stripping filler, collapsing phrasing, and restructuring sentences while preserving the instruction underneath. Fewer tokens per call means lower API costs and more room in your context window.

It runs entirely on your machine. No API key, no network calls, no sending your prompts to a third party.

<!-- Save your screenshot as docs/screenshot.png -->
<img width="1919" height="1015" alt="image" src="https://github.com/user-attachments/assets/2962d376-35e2-444d-b863-07b932a728af" />

---

## Why

Prompts written by hand tend to be wordy. Openers like "Hello, I want you to...", padding clauses, and natural-language connective tissue all cost tokens on every single call without changing what the model actually does. At scale, that adds up.

Prompt Optimizer applies deterministic rewrite rules to strip that overhead. Because it's rule-based rather than AI-powered, it's instant, free to run, and produces the same result every time for the same input — and it shows you every change it made.

## Example

**Before** — 22 tokens

> Hello, I want you to create a txt file on the desktop that will include the ingredients of a good prompt

**After** — 9 tokens

> Create desktop txt: ingredients of a good prompt

**13 tokens saved (59.1%).** Rules applied:

```
Stripped opener
"txt file" -> "txt"
"that will include" -> "with"
"<thing> on the <place>" -> "<place> <thing>"
Content clause -> colon
```

## Features

- **Three compression modes** — dial in how hard the optimizer squeezes
- **Side-by-side diff** — see exactly what was cut, struck through in the original
- **Changes applied log** — every rule that fired, listed by name, so nothing happens invisibly
- **Exact token counts** — measured with the `cl100k` tokenizer, not estimated
- **Editable output** — tweak the optimized prompt in place before you copy it
- **History** — recall prompts you've optimized before
- **Keyboard shortcuts** — `Ctrl+Enter` to optimize, `Ctrl+K` to clear
- **Fully offline** — no API key, no account, nothing leaves your computer

## Compression modes

Three toggles, stackable:

| Toggle | What it does |
|---|---|
| **Show diff** | Displays the original with removed text struck through, so you can see what changed |
| **Aggressive rules** | Enables the broader substitution set — cuts more, likelier to touch wording you meant to keep |
| **Telegraphic** | Restructures sentences into terse instruction form: drops connective words, reorders phrases, and converts content clauses to colon notation |

Telegraphic gives the biggest savings and the least natural-sounding output. It's built for prompts you send to a model, not prompts a human has to read.

## Requirements

- Windows
- Python 3.9+
- PySide6
- tiktoken

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/prompt-optimizer.git
cd prompt-optimizer
pip install -r requirements.txt
```

## Running the app

Double-click **`run.bat`**.

Or from a terminal:

```bash
python main.py
```

## How to use it

1. Paste your prompt into the **Original Prompt** pane.
2. Pick your modes with the toggles at the top right.
3. Press **Optimize** (or `Ctrl+Enter`).
4. Check the diff and the **Changes applied** list to confirm nothing important was cut.
5. Edit the result if you want, then hit **Copy Result**.

## How it works

Your text runs through a series of deterministic rewrite rules:

**Removal**
- Stripping openers — "Hello, I want you to...", "Could you please..."
- Filler removal — padding that adds tokens but no instruction
- Redundancy collapsing — repeated or restated requirements

**Substitution**
- Phrase shortening — `"that will include" → "with"`, `"txt file" → "txt"`

**Restructuring** (telegraphic mode)
- Phrase reordering — `"<thing> on the <place>" → "<place> <thing>"`
- Content clauses converted to colon notation — "that will include the X" → ": X"

Token counts are computed with `cl100k` before and after, and every applied rule is listed in the panel at the bottom.

## Limitations

- Rules are pattern-based, not semantic — the app can't tell when unusual phrasing is load-bearing. That's why the diff and the changes log are front and center: check them before using the output for anything important.
- Telegraphic output reads awkwardly by design. If a human needs to read the prompt too, leave that mode off.
- Token counts use `cl100k`. Other providers tokenize slightly differently, so treat the number as a close guide rather than an exact match for every model.
- Prompts that are already tight won't compress much. That's expected.

## Roadmap

- [ ] Custom user-defined rules
- [ ] Batch optimization for multiple prompts
- [ ] Additional tokenizers (`o200k`, Claude)
- [ ] macOS and Linux launchers
- [ ] Export and share rule sets

## Contributing

Issues and pull requests are welcome. For larger changes, please open an issue first to discuss the approach.

## License

MIT — see [LICENSE](LICENSE).
