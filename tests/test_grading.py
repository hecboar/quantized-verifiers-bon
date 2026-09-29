import pytest

from vq.grading import (
    extract_boxed,
    extract_final_answer,
    gsm8k_gold,
    grade,
    normalize_number,
)


class TestExtractBoxed:
    def test_simple(self):
        assert extract_boxed(r"so the answer is \boxed{42}.") == "42"

    def test_nested_braces(self):
        assert extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_last_box_wins(self):
        assert extract_boxed(r"\boxed{1} then \boxed{2}") == "2"

    def test_fbox(self):
        assert extract_boxed(r"\fbox{7}") == "7"

    def test_unbalanced_returns_none(self):
        assert extract_boxed(r"\boxed{12") is None

    def test_absent_returns_none(self):
        assert extract_boxed("no box here 12") is None


class TestNormalizeNumber:
    @pytest.mark.parametrize(
        "raw,expected",
        [("1,234", "1234"), ("$18", "18"), ("18.00", "18"), ("-3", "-3"),
         ("0.5", "0.5"), ("50%", "50"), (" 7 ", "7"), ("3/4", "0.75")],
    )
    def test_forms(self, raw, expected):
        assert normalize_number(raw) == expected

    def test_non_number(self):
        assert normalize_number("apple") is None


class TestExtractFinalAnswer:
    def test_boxed_preferred(self):
        assert extract_final_answer(r"The answer is 5. \boxed{6}") == "6"

    def test_answer_is_phrase(self):
        assert extract_final_answer("... so the final answer is 18.") == "18"

    def test_gsm8k_hash_format(self):
        assert extract_final_answer("work\n#### 72") == "72"

    def test_last_number_is_not_accepted(self):
        # Plan: "last number of the completion" is NOT an accepted extractor.
        assert extract_final_answer("I computed 3 + 4 = 7 and stopped") is None

    def test_empty(self):
        assert extract_final_answer("") is None


def test_gsm8k_gold():
    assert gsm8k_gold("Natalia sold...\n#### 1,072") == "1072"


class TestGrade:
    def test_numeric_equal(self):
        assert grade(r"\boxed{1,000}", "1000", kind="numeric") is True

    def test_numeric_wrong(self):
        assert grade(r"\boxed{999}", "1000", kind="numeric") is False

    def test_unparseable_is_incorrect(self):
        assert grade("I don't know", "1000", kind="numeric") is False

    def test_math_equivalence_fraction(self):
        assert grade(r"\boxed{\frac{1}{2}}", "0.5", kind="math") is True

    def test_math_equivalence_symbolic(self):
        assert grade(r"\boxed{2\sqrt{2}}", r"\sqrt{8}", kind="math") is True

    def test_math_wrong(self):
        assert grade(r"\boxed{\frac{1}{3}}", r"\frac{1}{2}", kind="math") is False

    def test_math_unparseable(self):
        assert grade("no answer", r"\frac{1}{2}", kind="math") is False


class TestLatexNumbers:
    @pytest.mark.parametrize("raw,expected", [
        (r"1{,}000", "1000"), (r"\frac{1}{2}", "0.5"), (r"\dfrac{3}{4}", "0.75"), (r"-\frac{1}{4}", "-0.25"),
        (r"\$18", "18"), (r"50\%", "50"), (r"18 \text{ dollars}", "18"), (r"\textbf{42}", "42"), (r"1\,000", "1000"),
    ])
    def test_latex_forms(self, raw, expected):
        assert normalize_number(raw) == expected

    def test_boxed_latex_fraction_correct_for_numeric_gold(self):
        assert grade(r"\boxed{\frac{1}{2}}", "0.5", kind="numeric") is True

    def test_boxed_thousands_separator(self):
        assert grade(r"\boxed{1{,}000}", "1000", kind="numeric") is True

    def test_symbolic_not_a_number(self):
        assert normalize_number(r"\sqrt{2}") is None


def test_lenient_grade_equals_strict_when_an_answer_is_extractable():
    from vq.grading import grade, grade_lenient

    for text, gold, kind in [(r"so \boxed{42}", "42", "numeric"), (r"\boxed{41}", "42", "numeric"),
                             (r"thus \boxed{\frac{1}{2}}", r"\frac{1}{2}", "math")]:
        assert grade_lenient(text, gold, kind) == grade(text, gold, kind)


def test_lenient_grade_reads_prose_answers():
    """Exploratory sensitivity analysis: prose answers without \boxed{} (the
    strict grader counts them wrong)."""
    from vq.grading import grade, grade_lenient

    prose = "Katy used 7 teaspoons per cup, so in total she used 42 teaspoons of sugar."
    assert not grade(prose, "42", "numeric") and grade_lenient(prose, "42", "numeric")
    assert not grade_lenient("She used 1,200 grams, not 42.", "1200", "numeric")  # last number wins
    assert grade_lenient("The total is 1,200 dollars.", "1200", "numeric")
    assert grade_lenient(r"Therefore the probability is $\frac{3}{8}$.", r"\frac{3}{8}", "math")
    assert not grade_lenient(r"Therefore the probability is $\frac{3}{7}$.", r"\frac{3}{8}", "math")
    assert not grade_lenient("I am not sure how to continue", "5", "numeric")


def test_lenient_grade_ignores_empty_math_delimiters():
    from vq.grading import grade_lenient

    assert grade_lenient("cost is $ $ and then 12 in total", "12", "math")
