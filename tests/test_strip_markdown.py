"""Markdown must never reach Piper.

Obrenna answers in markdown because it is the same brain the web UI talks to.
Piper pronounces the syntax: measured on aragon, "**Tuesday at 7 PM**" takes
4.98 s to speak versus 2.11 s clean — it says the asterisks out loud.

The stripper is deliberately conservative, so the "leaves alone" cases below are
as load-bearing as the "strips" ones: mangling arithmetic or a filename would be
worse than reading an asterisk.
"""
import pytest

from lexi.audio import strip_markdown


@pytest.mark.parametrize(
    "raw, expected",
    [
        # The case observed in a real answer.
        ("It's currently **Tuesday, September 01** now.",
         "It's currently Tuesday, September 01 now."),
        ("*emphasis* here", "emphasis here"),
        ("***all of it***", "all of it"),
        ("__bold__ and _italic_", "bold and italic"),
        ("~~gone~~ text", "gone text"),
        ("use the `run` command", "use the run command"),
        ("# Heading\nbody", "Heading\nbody"),
        ("> quoted line", "quoted line"),
        ("- first\n- second", "first\nsecond"),
        ("1. first\n2. second", "first\nsecond"),
        ("see [the docs](https://example.com) for more",
         "see the docs for more"),
        ("![alt](img.png) after", "after"),
    ],
)
def test_strips(raw, expected):
    assert strip_markdown(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "5 * 3 = 15",                    # arithmetic, not italics
        "a * b * c",                     # unpaired around spaces
        "file_name_here.txt",            # underscores inside a word
        "snake_case and more_words",
        "2 * 3 and 4 * 5",
        "the * character",               # lone asterisk
        "50% off, 3 - 2 = 1",            # a dash mid-line is not a bullet
    ],
)
def test_leaves_alone(raw):
    assert strip_markdown(raw) == raw


def test_fenced_code_block_is_dropped():
    out = strip_markdown("before\n```\nprint('hi')\n```\nafter")
    assert "print" not in out
    assert "before" in out and "after" in out


def test_collapses_runs_of_spaces_and_trims():
    assert strip_markdown("  **a**    **b**  ") == "a b"


def test_empty_and_syntax_only_input():
    assert strip_markdown("") == ""
    # A sentence that was nothing but markup must not be sent to TTS at all.
    assert strip_markdown("---") == ""
