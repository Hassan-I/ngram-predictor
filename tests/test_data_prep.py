from src.data_prep.normalizer import Normalizer

def test_normalize_lowercases():
    n = Normalizer()
    assert n.normalize("HELLO WORLD") == "hello world"

def test_normalize_removes_punctuation():
    n = Normalizer()
    assert n.normalize("hello, world!") == "hello world"

def test_normalize_removes_numbers():
    n = Normalizer()
    assert n.normalize("chapter 123") == "chapter"

def test_normalize_strips_whitespace():
    n = Normalizer()
    assert n.normalize("  extra  spaces  ") == "extra spaces"

def test_strip_gutenberg():
    n = Normalizer()
    n.text = "*** START OF THIS PROJECT GUTENBERG EBOOK ***\nStory content\n*** END OF THIS PROJECT GUTENBERG EBOOK ***"
    n.strip_gutenberg()
    assert "Story content" in n.text
    assert "START OF" not in n.text
    assert "END OF" not in n.text