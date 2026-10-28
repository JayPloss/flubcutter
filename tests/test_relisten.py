from flubcutter.relisten import find_repeats, patches, shift_key, to_source


def heard(*spec):
    """Words one after another, 0.3 s each."""
    return [{"word": w, "start": 0.3 * k, "end": 0.3 * k + 0.25} for k, w in enumerate(spec)]


def test_finds_repeats_the_book_does_not_have():
    book = "And he told Kreacher to go home. It's all right, it's gone. I - I think so."
    spots = find_repeats(heard("he", "told", "Kreacher", "to", "to", "go", "home"), book)
    assert spots == [{"first": 3, "last": 4}]
    assert find_repeats(heard("I", "I", "think", "so"), book) == []  # the book says it twice
    assert find_repeats(heard("there", "was", "a", "b", "basin"), "there was a basin") == [{"first": 3, "last": 4}]
    assert find_repeats(heard("moaned", "the", "orph", "moaned", "the", "elf"), "moaned the elf") \
        == [{"first": 0, "last": 5}]


def test_writes_back_only_where_the_transcript_heard_fewer_words():
    chunks = [{"part": 0, "start": 10.0, "end": 20.0, "at": 0.0}]
    h = heard("told", "to", "to", "go")
    spots = [{"first": 1, "last": 2}]
    # The recording's transcript took "to to" for one long "to".
    words = [{"word": "told", "start": 10.0, "end": 10.25, "part": 0},
             {"word": "to", "start": 10.3, "end": 10.85, "part": 0},
             {"word": "go", "start": 10.9, "end": 11.15, "part": 0}]
    (fix,) = patches(h, spots, chunks, words)
    assert (fix["first"], fix["count"]) == (1, 1)
    assert [(w["word"], w["start"]) for w in fix["words"]] == [("to", 10.3), ("to", 10.6)]
    # Heard as "but but" where the transcript has "but well": not a stutter.
    words[1:2] = [{"word": "but", "start": 10.3, "end": 10.55, "part": 0},
                  {"word": "well", "start": 10.6, "end": 10.85, "part": 0}]
    assert patches(h, spots, chunks, words) == []
    # An aside stays as recorded.
    words[1:3] = [{"word": "to", "start": 10.3, "end": 10.85, "part": 0}]
    assert patches(h, spots, chunks, words, skip={1}) == []


def test_a_repeat_across_a_cut_is_left_alone():
    chunks = [{"part": 0, "start": 10.0, "end": 10.5, "at": 0.0},
              {"part": 0, "start": 30.0, "end": 40.0, "at": 0.5}]
    assert to_source(chunks, 0.7) == (0, 30.2)
    words = [{"word": "to", "start": 10.3, "end": 10.45, "part": 0},
             {"word": "go", "start": 30.1, "end": 30.4, "part": 0}]
    assert patches(heard("x", "to", "to"), [{"first": 1, "last": 2}], chunks, words) == []


def test_join_keys_follow_the_words():
    assert shift_key("0:5>0:9", 7, 1, 2) == "0:5>0:10"
    assert shift_key("0:3>0:4", 7, 1, 2) == "0:3>0:4"
    assert shift_key("0:7>1:20", 7, 1, 2) == "0:7>1:21"
