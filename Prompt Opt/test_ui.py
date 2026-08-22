"""Headless smoke test: builds the real Tk app, drives it, asserts on widgets.

Runs without pyautogui or screen coordinates. The window is created but never
shown (withdrawn), so this is safe to run in the background.

    python test_ui.py
"""
import sys
import tkinter as tk

import prompt_optimizer as po

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if not cond and detail else ""))
    if not cond:
        FAILS.append(name)


def main():
    # Build the real App, but keep it off-screen.
    app_holder = {}
    orig_mainloop = tk.Tk.mainloop
    tk.Tk.mainloop = lambda self: app_holder.setdefault("app", self)
    try:
        po._run_gui()
    except tk.TclError as exc:
        print(f"No display available: {exc}")
        return 1
    finally:
        tk.Tk.mainloop = orig_mainloop

    app = app_holder["app"]
    app.withdraw()

    def type_in(text):
        app.input_text.delete("1.0", "end")
        app.input_text.insert("1.0", text)
        app._update_counts()

    # ── 1. widgets exist ────────────────────────────────────────────────────
    for w in ("input_text", "output_text", "log", "lbl_savings", "lbl_in", "lbl_out"):
        check(f"widget {w} exists", hasattr(app, w))

    # ── 2. no AI surface remains ────────────────────────────────────────────
    for gone in ("btn_load", "btn_ai_run", "btn_install", "ai_check",
                 "ai_model_menu", "ai_status_var", "_apply_ai_now", "_run_ai_on"):
        check(f"{gone} removed", not hasattr(app, gone))
    for gone in ("AIEngine", "AI", "ai_deps_installed", "install_ai_deps"):
        check(f"module-level {gone} removed", not hasattr(po, gone))

    # ── 3. the original failing prompt ──────────────────────────────────────
    type_in("hi write a txt file that will have all of the good indicators of a good Prompt")
    app._optimize()
    out = app._get_out()
    check("greeting stripped", not out.lower().startswith("hi"), out)
    check("grammatical result", "file will have" not in out and "file lists" not in out, out)
    check("savings reported", "saved" in app.lbl_savings.cget("text").lower(),
          app.lbl_savings.cget("text"))
    print(f"       -> {out!r}")

    # ── 4. heavy filler ─────────────────────────────────────────────────────
    type_in("Could you please provide a summary of the following text? "
            "Please note that it is important that your response is as clear "
            "and concise as possible, to the best of your ability. "
            "Feel free to ask me if you need any clarification.")
    app._optimize()
    out = app._get_out()
    orig_t = po.count_tokens(app._get_in())
    new_t = po.count_tokens(out)
    check("heavy filler cut by >50%", new_t < orig_t * 0.5, f"{orig_t} -> {new_t}")
    print(f"       -> {out!r}  ({orig_t} -> {new_t} tok)")

    # ── 5. structure preserved ──────────────────────────────────────────────
    type_in("Please review this:\n"
            "```python\n"
            "def add(a, b):\n"
            "    return a + b  # in order to add\n"
            "```\n"
            "- check the log file\n"
            "- check the json file")
    app._optimize()
    out = app._get_out()
    check("code block intact", "def add(a, b):\n    return a + b  # in order to add" in out, out)
    check("bullets intact", out.count("\n- check") == 2, out)
    print(f"       -> {out!r}")

    # ── 6. already-concise prompt ───────────────────────────────────────────
    type_in("Summarize this article in 3 bullet points.")
    app._optimize()
    check("concise prompt unchanged", app._get_out() == "Summarize this article in 3 bullet points.",
          app._get_out())

    # ── 7. aggressive toggle changes behaviour ──────────────────────────────
    type_in("Just simply write a script that will parse the logs.")
    app.aggressive.set(True)
    app._optimize()
    aggr = app._get_out()
    app.aggressive.set(False)
    app._optimize()
    safe = app._get_out()
    check("aggressive is more aggressive", po.count_tokens(aggr) < po.count_tokens(safe),
          f"{aggr!r} vs {safe!r}")
    app.aggressive.set(True)

    # ── 8. clear / history ──────────────────────────────────────────────────
    check("history recorded", len(app.history) >= 5, str(len(app.history)))
    app._clear()
    check("clear empties input", app._get_in() == "")
    check("clear empties output", app._get_out() == "")
    check("clear resets counters", app.lbl_in.cget("text") == "0 tokens")

    # ── 9. diff highlighting ────────────────────────────────────────────────
    type_in("Hi there, could you please provide a summary of the log file, "
            "in order to help me debug? Thank you for your help.")
    app._optimize()
    cut = app.input_text.tag_ranges("cut")
    new = app.output_text.tag_ranges("new")
    check("removed text is struck through in the original", len(cut) > 0)
    check("the greeting is marked as cut",
          any("Hi there" in app.input_text.get(cut[i], cut[i + 1])
              for i in range(0, len(cut), 2)),
          [app.input_text.get(cut[i], cut[i + 1]) for i in range(0, len(cut), 2)])

    # Regression: Ctrl+Enter's KeyRelease must not wipe the highlighting that
    # the same keystroke just produced.
    app._schedule_count()
    check("diff survives the KeyRelease of Ctrl+Enter",
          len(app.input_text.tag_ranges("cut")) == len(cut))

    # Editing the text must drop the now-stale highlighting.
    app.input_text.insert("1.0", "extra words here ")
    app._schedule_count()
    check("editing clears stale highlighting",
          len(app.input_text.tag_ranges("cut")) == 0)

    # Toggling off clears, toggling on restores.
    app._optimize()
    app.show_diff.set(False)
    app._on_diff_toggle()
    check("toggle off clears highlighting",
          len(app.input_text.tag_ranges("cut")) == 0)
    app.show_diff.set(True)
    app._on_diff_toggle()
    check("toggle on restores highlighting",
          len(app.input_text.tag_ranges("cut")) > 0)

    # ── 10. the app runs with no optional dependencies at all ───────────────
    for mod in ("torch", "transformers"):
        check(f"{mod} never imported", mod not in sys.modules)

    app.destroy()

    print(f"\n{'ALL PASSED' if not FAILS else str(len(FAILS)) + ' FAILURE(S): ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
