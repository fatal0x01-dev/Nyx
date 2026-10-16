from utils.human_input import detect_human_input_need, request_human_input


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


def test_detects_command_not_found_package_install_prompt():
    text = (
        "Command 'docker-compose' not found, but can be installed with:\n"
        "sudo apt install docker-compose\n"
        "Do you want to install it? (N/y)"
    )
    result = detect_human_input_need(text)
    assert result is not None
    assert result["kind"] == "package_install_confirmation"


def test_detects_unsafe_ignore_cancel_prompt():
    for prompt in (
        "Error: Can't have a partition outside the disk!\nIgnore/Cancel?",
        "Fix/Ignore?",
        "(I)gnore/(C)ancel?",
    ):
        result = detect_human_input_need(prompt)
        assert result is not None
        assert result["kind"] == "unsafe_confirmation"


def test_approval_words_normalize_to_single_terminal_key(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    assert request_human_input("approval", "Continue? [y/n]") == "y"

    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    assert request_human_input("approval", "Continue? [y/n]") == "n"


def test_invalid_approval_reprompts_instead_of_typing_arbitrary_text(monkeypatch):
    answers = iter(["maybe", "yes"])
    prompts = []
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: prompts.append(prompt) or next(answers),
    )

    assert request_human_input("approval", "Continue? [y/n]") == "y"
    assert len(prompts) == 2
    assert all("[y/n]" in prompt for prompt in prompts)
