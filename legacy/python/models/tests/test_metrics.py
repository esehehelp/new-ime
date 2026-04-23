"""Tests for evaluation metrics."""

from models.src.eval.metrics import (
    EvalResult,
    character_accuracy,
    edit_distance,
    exact_match,
    top_k_character_accuracy,
    top_k_exact_match,
)


class TestEditDistance:
    def test_identical(self):
        assert edit_distance("abc", "abc") == 0

    def test_empty(self):
        assert edit_distance("", "abc") == 3
        assert edit_distance("abc", "") == 3

    def test_both_empty(self):
        assert edit_distance("", "") == 0

    def test_substitution(self):
        assert edit_distance("abc", "axc") == 1

    def test_insertion(self):
        assert edit_distance("abc", "abcd") == 1

    def test_deletion(self):
        assert edit_distance("abcd", "abc") == 1

    def test_japanese(self):
        assert edit_distance("漢字変換", "漢字返還") == 2  # 変→返, 換→還
        assert edit_distance("東京都", "東京") == 1


class TestCharacterAccuracy:
    def test_perfect(self):
        assert character_accuracy("漢字", "漢字") == 1.0

    def test_one_error(self):
        acc = character_accuracy("漢字変換", "漢字返還")
        assert abs(acc - 0.5) < 0.01  # 2 edits / 4 chars

    def test_completely_wrong(self):
        acc = character_accuracy("あ", "かきくけ")
        assert acc == 0.0  # edit_distance=4, len=1, clamped to 0

    def test_empty_reference(self):
        assert character_accuracy("", "") == 1.0
        assert character_accuracy("", "x") == 0.0


class TestTopKAccuracy:
    def test_top1(self):
        acc = top_k_character_accuracy("漢字", ["漢字", "感じ"], k=1)
        assert acc == 1.0

    def test_top1_wrong_but_top2_right(self):
        acc1 = top_k_character_accuracy("漢字", ["感じ", "漢字"], k=1)
        acc2 = top_k_character_accuracy("漢字", ["感じ", "漢字"], k=2)
        assert acc1 < 1.0
        assert acc2 == 1.0

    def test_empty_candidates(self):
        assert top_k_character_accuracy("漢字", [], k=1) == 0.0


class TestExactMatch:
    def test_match(self):
        assert exact_match("漢字", "漢字")

    def test_no_match(self):
        assert not exact_match("漢字", "感じ")

    def test_top_k(self):
        assert not top_k_exact_match("漢字", ["感じ", "幹事"], k=2)
        assert top_k_exact_match("漢字", ["感じ", "漢字"], k=2)


class TestEvalResult:
    def test_single_sample(self):
        result = EvalResult()
        result.add("漢字", ["漢字", "感じ"])
        s = result.summary()
        assert s["total"] == 1
        assert s["char_acc_top1"] == 1.0
        assert s["exact_match_top1"] == 1.0

    def test_multiple_samples(self):
        result = EvalResult()
        result.add("漢字", ["漢字"])
        result.add("変換", ["返還"])  # 2 edits / 2 chars = 0.0 acc
        s = result.summary()
        assert s["total"] == 2
        assert 0.4 < s["char_acc_top1"] < 0.6  # (1.0 + 0.0) / 2 = 0.5

    def test_report(self):
        result = EvalResult()
        result.add("テスト", ["テスト"])
        report = result.report()
        assert "1 samples" in report
        assert "CharAcc" in report
