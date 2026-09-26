from utils.human_input import detect_human_input_need


def test_detects_approval_prompt():
    text = "This will overwrite the existing file. Continue? [Y/n]"
    result = detect_human_input_need(text)
    assert result is not None
    assert result["kind"] == "approval"
    assert "overwrite" in result["prompt"].lower()


def test_detects_credential_prompt():
    text = "Password required for sudo operation:"
    result = detect_human_input_need(text)
    assert result is not None
    assert result["kind"] == "credential"
    assert "password" in result["prompt"].lower()
