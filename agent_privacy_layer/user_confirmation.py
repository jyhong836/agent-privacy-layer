"""
User Confirmation
=================
Handles interactive user confirmation for sensitive data-access requests.

In a real deployment these prompts would be presented through whatever UI
the agent framework supports (chat, web, CLI …).  This module provides a
simple, extensible abstraction that:

* Blocks until the user approves or denies the request.
* Supports a ``dry_run`` mode that auto-approves all requests (useful for
  testing).
* Supports an ``auto_deny`` mode that auto-denies all requests (useful for
  unattended / sandboxed contexts).
* Allows a custom ``callback`` to route the confirmation to an external UI.
"""

from __future__ import annotations

from typing import Callable, Optional


class UserConfirmation:
    """
    Prompt the user for confirmation before performing a sensitive action.

    Parameters
    ----------
    dry_run:
        When ``True`` all requests are automatically approved and no prompt is
        shown.  Useful for unit tests.
    auto_deny:
        When ``True`` all requests are automatically denied without prompting.
    callback:
        A callable ``(message: str) -> bool`` that handles the confirmation UI.
        If ``None`` the built-in ``input()`` prompt is used.
    """

    def __init__(
        self,
        dry_run: bool = False,
        auto_deny: bool = False,
        callback: Optional[Callable[[str], bool]] = None,
    ) -> None:
        if dry_run and auto_deny:
            raise ValueError("dry_run and auto_deny cannot both be True")
        self.dry_run = dry_run
        self.auto_deny = auto_deny
        self._callback = callback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def confirm(self, message: str) -> bool:
        """
        Ask the user to confirm *message*.

        Returns
        -------
        bool
            ``True`` if the user approved, ``False`` otherwise.
        """
        if self.dry_run:
            return True
        if self.auto_deny:
            return False
        if self._callback is not None:
            return bool(self._callback(message))
        return self._cli_prompt(message)

    def require(self, message: str) -> None:
        """
        Like :meth:`confirm` but raise :class:`PermissionError` on denial.

        Parameters
        ----------
        message:
            Human-readable description of the access being requested.

        Raises
        ------
        PermissionError
            If the user denied the request.
        """
        if not self.confirm(message):
            raise PermissionError(f"User denied access: {message}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cli_prompt(message: str) -> bool:
        """Show a Y/N prompt on the terminal."""
        full_message = f"\n[PRIVACY LAYER] Access request:\n{message}\nApprove? [y/N] "
        try:
            answer = input(full_message).strip().lower()
        except EOFError:
            # Non-interactive environment
            return False
        return answer in ("y", "yes")
