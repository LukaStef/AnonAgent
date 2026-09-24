"""Asking the user about uncertain detections.

Every test here is really one question: when the answer is unclear, does the
code fall on the safe side?
"""

import io

from helpers import StubDetector, span

from anonagent.privacy.detector import DetectedEntity
from anonagent.privacy.masker import Masker
from anonagent.review import InteractiveReviewer, mask_everything


def entity(text="Vienna", entity_type="LOCATION", score=0.3):
    return DetectedEntity(entity_type, 0, len(text), score, text)


def reviewer(answers="", *, confirm_below=0.5):
    return InteractiveReviewer(
        confirm_below=confirm_below,
        input_stream=io.StringIO(answers),
        output_stream=io.StringIO(),
        interactive=True,
    )


def test_a_confident_detection_is_masked_without_asking():
    asked = io.StringIO()
    review = InteractiveReviewer(
        input_stream=io.StringIO(), output_stream=asked, interactive=True
    )
    approved, declined = review([entity(score=0.9)])

    assert (len(approved), declined) == (1, [])
    assert asked.getvalue() == "", "nothing should have been printed"


def test_an_uncertain_detection_is_put_to_the_user():
    review = reviewer("y\n")
    approved, declined = review([entity(score=0.3)])
    assert (len(approved), declined) == (1, [])


def test_answering_no_leaves_the_value_in_the_clear():
    review = reviewer("n\n")
    approved, declined = review([entity(score=0.3)])
    assert (approved, len(declined)) == ([], 1)


def test_an_empty_answer_masks():
    """Hitting enter must never be the way to leak something."""
    review = reviewer("\n")
    approved, _ = review([entity(score=0.3)])
    assert len(approved) == 1


def test_a_closed_stream_masks():
    review = reviewer("")
    approved, _ = review([entity(score=0.3)])
    assert len(approved) == 1


def test_a_non_interactive_session_masks_without_asking():
    review = InteractiveReviewer(
        input_stream=io.StringIO("n\n"), output_stream=io.StringIO(), interactive=False
    )
    approved, declined = review([entity(score=0.3)])
    assert (len(approved), declined) == (1, []), "must not honour input nobody typed"


def test_a_decision_is_remembered_for_the_same_value():
    review = reviewer("n\n")  # one answer, two mentions
    approved, declined = review([entity(score=0.3), entity(score=0.3)])
    assert (approved, len(declined)) == ([], 2)


def test_the_decision_is_remembered_across_spellings():
    review = reviewer("n\n")
    approved, declined = review([entity(text="Vienna", score=0.3)])
    approved2, declined2 = review([entity(text="VIENNA", score=0.3)])
    assert declined and declined2
    assert approved == approved2 == []


def test_separate_values_are_asked_about_separately():
    review = reviewer("n\ny\n")
    approved, declined = review(
        [entity(text="Vienna", score=0.3), entity(text="Kevin", entity_type="PERSON", score=0.3)]
    )
    assert [e.text for e in declined] == ["Vienna"]
    assert [e.text for e in approved] == ["Kevin"]


def test_the_threshold_decides_what_gets_asked():
    review = reviewer("n\n", confirm_below=0.95)
    approved, declined = review([entity(score=0.9)])
    assert (approved, len(declined)) == ([], 1)


def test_the_default_reviewer_masks_everything_silently():
    entities = [entity(score=0.1)]
    assert mask_everything(entities) == (entities, [])


def test_a_declined_value_is_not_masked_and_never_enters_the_vault():
    text = "Kevin met Sarah Chen in Vienna."
    detector = StubDetector(
        [
            span(text, "Kevin", "PERSON", score=0.3),
            span(text, "Sarah Chen", "PERSON", score=0.9),
            span(text, "Vienna", "LOCATION", score=0.3),
        ]
    )
    masker = Masker(detector=detector, reviewer=reviewer("n\nn\n"))
    result = masker.mask(text)

    assert "Kevin" in result.masked_text
    assert "Vienna" in result.masked_text
    assert "Sarah Chen" not in result.masked_text
    assert len(masker.vault) == 1
    assert [e.text for e in result.declined] == ["Kevin", "Vienna"]


def test_a_declined_value_is_not_swept_up_later():
    """Declining must hold for the whole text, not just the first mention."""
    text = "Kevin filed it. Ask Kevin for the date."
    detector = StubDetector([span(text, "Kevin", "PERSON", score=0.3)])
    masker = Masker(detector=detector, reviewer=reviewer("n\n"))

    result = masker.mask(text)
    assert result.masked_text == text
    assert result.sealed == []


def test_declining_does_not_trip_the_leak_guard():
    """The guard protects the vault's promises, not choices the user made."""
    text = "Kevin filed it."
    detector = StubDetector([span(text, "Kevin", "PERSON", score=0.3)])
    masker = Masker(detector=detector, reviewer=reviewer("n\n"))

    result = masker.mask(text)
    assert masker.leaks(result.masked_text) == []
