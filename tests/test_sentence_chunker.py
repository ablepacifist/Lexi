from lexi.audio import SentenceChunker


def test_emits_on_sentence_boundary():
    c = SentenceChunker()
    assert c.feed("Hello") == []
    assert c.feed(" world.") == []  # boundary needs trailing whitespace/newline
    out = c.feed(" Next one!")
    assert out == ["Hello world."]
    assert c.flush() == "Next one!"


def test_multiple_sentences_in_one_feed():
    c = SentenceChunker()
    out = c.feed("One. Two? Three! ")
    assert out == ["One.", "Two?", "Three!"]
    assert c.flush() is None


def test_newline_is_a_boundary():
    c = SentenceChunker()
    out = c.feed("a line\nmore")
    assert out == ["a line"]
    assert c.flush() == "more"


def test_soft_flush_when_no_punctuation():
    c = SentenceChunker(soft_flush_chars=20)
    long_no_punct = "word " * 10  # 50 chars, no terminal punctuation
    out = c.feed(long_no_punct)
    assert out, "should soft-flush a long run with no sentence boundary"
    assert all(s.strip() for s in out)


def test_flush_returns_remainder_without_punctuation():
    c = SentenceChunker()
    c.feed("dangling text")
    assert c.flush() == "dangling text"
    assert c.flush() is None
