from flubcutter.player import chapter_numbers


def test_chapter_numbers_come_from_the_headings():
    assert chapter_numbers("Chapter Seven: The Slug Club.\n\nHarry spent...") == [7]
    two = "Chapter Twenty-eight: Flight of the Prince.\n\n...\n\nChapter Twenty-nine: The Phoenix Lament.\n\n..."
    assert chapter_numbers(two) == [28, 29]
    assert chapter_numbers("Chapter Thirty: The White Tomb.") == [30]
    assert chapter_numbers("In another moment down went Alice") == []
