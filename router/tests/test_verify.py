from router import verify


# --- prose: never verifiable, so it must never trigger a cascade

def test_prose_always_returns_none():
    assert verify.check("prose", "Paris is the capital of France.") is None
    assert verify.check("prose", "") is None


# --- structured

def test_valid_json_matching_schema_passes():
    schema = {"type": "object", "required": ["name", "age"]}
    assert verify.check("structured", '{"name": "Ada", "age": 36}', schema=schema) == 1


def test_json_in_a_fenced_block_is_still_found():
    schema = {"type": "object", "required": ["name"]}
    text = 'Here you go:\n```json\n{"name": "Ada"}\n```\n'
    assert verify.check("structured", text, schema=schema) == 1


def test_malformed_json_fails():
    schema = {"type": "object", "required": ["name"]}
    assert verify.check("structured", '{"name": ', schema=schema) == 0


def test_json_missing_a_required_key_fails():
    schema = {"type": "object", "required": ["name", "age"]}
    assert verify.check("structured", '{"name": "Ada"}', schema=schema) == 0


def test_structured_without_a_schema_only_checks_parseability():
    assert verify.check("structured", '{"anything": 1}') == 1
    assert verify.check("structured", "not json at all") == 0


def test_structured_accepts_a_top_level_array():
    assert verify.check("structured", "[1, 2, 3]") == 1


# --- code

def test_syntactically_valid_python_passes():
    text = "```python\ndef add(a, b):\n    return a + b\n```"
    assert verify.check("code", text) == 1


def test_syntactically_invalid_python_fails():
    text = "```python\ndef add(a, b)\n    return a + b\n```"
    assert verify.check("code", text) == 0


def test_python_without_a_fence_is_still_checked():
    assert verify.check("code", "x = 1\ny = x + 2\n") == 1


def test_code_with_no_extractable_block_returns_none():
    assert verify.check("code", "I would use a hash map for this.") is None


def test_unsupported_language_returns_none():
    text = "```rust\nfn main() { println!(\"hi\"); }\n```"
    assert verify.check("code", text) is None


def test_extract_code_block_reports_the_language():
    lang, code = verify.extract_code_block("```python\nx = 1\n```")
    assert lang == "python"
    assert code.strip() == "x = 1"


def test_extract_code_block_returns_none_none_when_absent():
    assert verify.extract_code_block("just prose") == (None, None)


# --- math

def test_matching_boxed_answer_passes():
    assert verify.check("math", r"So the answer is \boxed{42}.", expected="42") == 1


def test_wrong_boxed_answer_fails():
    assert verify.check("math", r"The answer is \boxed{41}.", expected="42") == 0


def test_algebraically_equivalent_forms_pass():
    assert verify.check("math", r"\boxed{1/2}", expected="0.5") == 1
    assert verify.check("math", r"\boxed{2/4}", expected="1/2") == 1


def test_missing_boxed_answer_returns_none():
    assert verify.check("math", "The answer is forty two.", expected="42") is None


def test_math_without_an_expected_answer_returns_none():
    assert verify.check("math", r"\boxed{42}") is None


def test_unparseable_expression_returns_none_not_zero():
    """Cannot tell is different from wrong: it must not trigger a cascade."""
    assert verify.check("math", r"\boxed{\text{see above}}", expected="42") is None


def test_extract_boxed_takes_the_last_box():
    assert verify.extract_boxed(r"first \boxed{1} then \boxed{2}") == "2"


def test_extract_boxed_handles_nested_braces():
    assert verify.extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"


# --- unknown task type

def test_unknown_task_type_returns_none():
    assert verify.check("poetry", "roses are red") is None
