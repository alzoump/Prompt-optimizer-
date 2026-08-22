# Prompt Optimizer

**Cut your prompt token count without cutting quality.**

Prompt Optimizer is a Windows desktop app that rewrites your prompts into leaner versions — stripping filler, politeness padding, and redundant phrasing while preserving the instruction underneath. Fewer tokens per call means lower API costs and more room in your context window.

It runs entirely on your machine. No API key, no network calls, no sending your prompts to a third party.

<img width="1918" height="1012" alt="image" src="https://github.com/user-attachments/assets/f1eb763d-476b-4ceb-961c-cabd9bf3ed55" />


---

## Why

Prompts written by hand tend to be wordy. Openers like "Hello, I want you to..." cost tokens on every single call without changing what the model actually does. At scale, that adds up.

Prompt Optimizer applies a set of deterministic rewrite rules to strip that overhead. Because it's rule-based rather than AI-powered, it's instant, free to run, and produces the same result every time for the same input — and it shows you every change it made.

## Example

**Before** — 22 tokens

> Hello, I want you to create a txt file on the desktop that will include the ingredients of a good prompt

**After** — 13 tokens

> Create a txt on the desktop with the ingredients of a good prompt

**9 tokens saved (40.9%).** Changes applied: stripped opener, `"txt file" → "txt"`, `"that will include" → "with"`.

## Features

- **Side-by-side diff** — see exactly what was cut, struck through in the original
- **Changes applied log** — every rule that fired, listed by name, so nothing happens invisibly
- **Exact token counts** — measured with the `cl100k` tokenizer, not estimated
- **Editable output** — tweak the optimized prompt in place before you copy it
- **Aggressive rules toggle** — switch between conservative and maximum compression
- **History** — recall prompts you've optimized before
- **Keyboard shortcuts** — `Ctrl+Enter` to optimize, `Ctrl+K` to clear
- **Fully offline** — no API key, no account, nothing leaves your computer

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
2. Press **Optimize** (or `Ctrl+Enter`).
3. Check the diff and the **Changes applied** list to confirm nothing important was cut.
4. Edit the result if you want, then hit **Copy Result**.

Turn on **Aggressive rules** to squeeze harder — it removes more, but is likelier to touch wording you meant to keep.

## How it works

Your text runs through a series of deterministic rewrite rules, including:

- **Stripping openers** — "Hello, I want you to...", "Could you please..."
- **Phrase substitution** — "that will include" → "with", "txt file" → "txt"
- **Filler removal** — padding that adds tokens but no instruction
- **Redundancy collapsing** — repeated or restated requirements

Token counts are computed with `cl100k` before and after, and every applied rule is listed in the panel at the bottom.

## Limitations

- Rules are pattern-based, not semantic — the app can't tell when unusual phrasing is load-bearing. That's why the diff and the changes log are front and center: check them before using the output for anything important.
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
