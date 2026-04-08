"""Tests for UserConfirmation."""
import pytest
from agent_privacy_layer.user_confirmation import UserConfirmation


def test_dry_run_approves():
    uc = UserConfirmation(dry_run=True)
    assert uc.confirm("Access dataset X?") is True


def test_auto_deny_denies():
    uc = UserConfirmation(auto_deny=True)
    assert uc.confirm("Access dataset X?") is False


def test_dry_run_and_auto_deny_raises():
    with pytest.raises(ValueError, match="dry_run and auto_deny"):
        UserConfirmation(dry_run=True, auto_deny=True)


def test_callback_approves():
    uc = UserConfirmation(callback=lambda msg: True)
    assert uc.confirm("Access?") is True


def test_callback_denies():
    uc = UserConfirmation(callback=lambda msg: False)
    assert uc.confirm("Access?") is False


def test_require_raises_on_denial():
    uc = UserConfirmation(auto_deny=True)
    with pytest.raises(PermissionError, match="User denied access"):
        uc.require("Access dataset X?")


def test_require_passes_on_approval():
    uc = UserConfirmation(dry_run=True)
    uc.require("Access dataset X?")  # should not raise


def test_callback_receives_message():
    messages = []
    uc = UserConfirmation(callback=lambda msg: messages.append(msg) or True)
    uc.confirm("my message")
    assert "my message" in messages[0]
