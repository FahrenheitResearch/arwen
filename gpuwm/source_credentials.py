"""What a registry row needs CONFIGURED before its bytes can be acquired.

A source whose acquisition needs a personal account key is a fact about
that source, and it belongs in the same table as every other fact about
it.  Before this module the fact lived nowhere: the registry rows carried
no credential column, so each front end grew its own one-row exception
table ("this id needs a key, here is where to put it") -- a per-model
lookup, which is the thing the arbitrary acceptance test forbids.  A row
declares a :class:`SourceCredential` now and every consumer reads it off
the reply.

NOTHING HERE KNOWS A MODEL.  There is no id in this file, no branch on
one, and ``tests/test_source_registry_columns.py`` fails if one appears:
a declaration is data, and everything below is a derivation over the
declared fields.  Adding a source that needs a key stays one row.

PRESENCE, NEVER THE VALUE.  A credential's presence is a ``stat`` or an
environment lookup, and the answer that leaves this module is a boolean.
The secret itself is never read, never carried and never printed -- a
front door that echoed a personal key would hand it to every log,
receipt and screenshot the reply reaches.

PRESENCE, NEVER VALIDITY, either.  Whether a key is *accepted* is the
provider's verdict to give; guessing at it here would produce confident
wrong advice about a file this project does not own.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
from typing import Any, Iterable


class CredentialLocation(str, Enum):
    """WHERE a credential is configured, as a kind a check can act on.

    A small closed vocabulary on purpose: each member is one generic
    check, so a row declares its kind and its location string and needs
    no code of its own.  A kind added later gets one arm here, not one
    arm per source.
    """

    #: A file in the user's home directory, named by its leaf (the tool
    #: that reads it decides the format; this only asks whether it is
    #: there).
    HOME_FILE = "home_file"
    #: An environment variable, named by the variable.
    ENV_VAR = "env_var"


@dataclass(frozen=True)
class SourceCredential:
    """One declared credential prerequisite of a registry row."""

    #: Stable identifier for the credential itself, so two rows needing
    #: the same account are recognisably the same prerequisite.
    credential_id: str
    #: What a person calls it, e.g. the provider's own name for the key.
    display_name: str
    location_kind: CredentialLocation
    #: The leaf filename or the variable name, per ``location_kind``.
    location: str
    #: Which stage stops without it, in the project's own vocabulary
    #: ("acquisition", "rendering", ...).
    needed_for: str
    #: The concrete breakage its absence causes, in one sentence.  A
    #: prerequisite that cannot name what it prevents is a nag, and the
    #: gate law says it does not exist.
    breakage: str
    #: Where a person obtains one, when there is such a place.
    obtain_url: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.location_kind, CredentialLocation):
            raise TypeError(
                f"{self.credential_id}: location_kind must be a "
                "CredentialLocation, so the presence check has a defined "
                "meaning instead of a guess at a string")
        for field in ("credential_id", "display_name", "location",
                      "needed_for", "breakage"):
            if not str(getattr(self, field) or "").strip():
                raise ValueError(
                    f"{self.credential_id or '<unnamed credential>'}: "
                    f"{field} must be declared -- a credential row that "
                    "cannot say what it is, where it lives, what needs it "
                    "and what breaks without it tells a reader nothing "
                    "they can act on")


def credential_location_display(credential: SourceCredential) -> str:
    """The declared location as a person would look for it on this box."""

    if credential.location_kind is CredentialLocation.HOME_FILE:
        return str(Path.home() / credential.location)
    if credential.location_kind is CredentialLocation.ENV_VAR:
        return f"environment variable {credential.location}"
    return credential.location


def credential_present(credential: SourceCredential) -> bool | None:
    """Is the declared credential in place on THIS box?

    ``None`` when the declared kind has no check here -- a truthful "not
    answered" rather than a ``False`` a reader would act on.  Existence
    only; the value is never read.
    """

    try:
        if credential.location_kind is CredentialLocation.HOME_FILE:
            return (Path.home() / credential.location).is_file()
        if credential.location_kind is CredentialLocation.ENV_VAR:
            return bool(os.environ.get(credential.location, "").strip())
    except OSError:
        # A home directory that cannot be resolved is not evidence that
        # the key is missing.
        return None
    return None


def credential_facts(credential: SourceCredential) -> dict[str, Any]:
    """One credential as JSON: its declaration, plus this box's verdict.

    ``status_message`` is the one line a front end shows.  It is derived
    from the declared fields, so the sentence a user reads about a source
    registered tomorrow is as complete as the sentence about one
    registered today.
    """

    where = credential_location_display(credential)
    present = credential_present(credential)
    obtain = (f"  Obtain one at {credential.obtain_url}."
              if credential.obtain_url else "")
    present_message = (
        f"{credential.display_name} is present ({where}); "
        f"{credential.needed_for} can proceed.")
    absent_message = (
        f"No {credential.display_name} at {where}: {credential.breakage}."
        f"{obtain}")
    unknown_message = (
        f"Whether {credential.display_name} is present at {where} cannot "
        f"be checked from here; {credential.needed_for} needs it and "
        f"without it {credential.breakage}.{obtain}")
    status_message = (unknown_message if present is None
                      else present_message if present else absent_message)
    return {
        "credential_id": credential.credential_id,
        "display_name": credential.display_name,
        "location_kind": credential.location_kind.value,
        "location": credential.location,
        "location_display": where,
        "needed_for": credential.needed_for,
        "breakage": credential.breakage,
        "obtain_url": credential.obtain_url,
        # Existence on the box answering, never the value and never a
        # verdict on whether the provider would accept it.
        "present": present,
        "present_message": present_message,
        "absent_message": absent_message,
        "status_message": status_message,
    }


def credential_declaration(credential: SourceCredential) -> dict[str, Any]:
    """The declared fields alone, with no verdict about this box.

    The capability manifest is a provenance document -- two boxes must
    be able to compare theirs -- so it carries the DECLARATION, while
    :func:`credential_facts` carries the declaration plus what is true
    here.
    """

    return {
        "credential_id": credential.credential_id,
        "display_name": credential.display_name,
        "location_kind": credential.location_kind.value,
        "location": credential.location,
        "needed_for": credential.needed_for,
        "breakage": credential.breakage,
        "obtain_url": credential.obtain_url,
    }


def credential_short_note(credential: SourceCredential) -> str:
    """ONE terminal line: what is missing, what it blocks, where to get it.

    Deliberately shorter than ``status_message``.  A terminal caller
    prints this beside a numbered command, and the wizard's default
    output has a measured line budget it must fit inside -- the block a
    reader is meant to act on stops being one when it scrolls.  It is
    also one printed line, never a pre-wrapped paragraph: a path or a
    URL split across a break is a path a reader cannot copy.
    """

    remedy = (f"; get one at {credential.obtain_url}"
              if credential.obtain_url else "")
    return (f"{credential.display_name} needed first: no "
            f"{credential_location_display(credential)} yet -- "
            f"{credential.needed_for} cannot start without it{remedy}")


def absent_credential_notes(
        credentials: Iterable[SourceCredential]) -> list[str]:
    """One pointer line per declared credential that is NOT in place.

    Empty when everything declared is there, because a line printed
    whether or not it was true would be noise on every later run.  This
    is the terminal form of the same derivation ``--sources`` serves as
    JSON: one place decides what the sentence says, so the wizard and a
    front end cannot come to word it differently.
    """

    return [credential_short_note(credential) for credential in credentials
            if credential_present(credential) is not True]


def credentials_block(
        credentials: Iterable[SourceCredential]) -> dict[str, Any]:
    """A row's whole credential answer, including the empty one.

    An empty declaration reports exactly that: the table declares no
    prerequisite.  It deliberately does NOT say "this source is public"
    -- that is a promise about a provider, and the only thing the
    registry knows is what its own column says.
    """

    items = [credential_facts(credential) for credential in credentials]
    if not items:
        return {
            "required": False,
            "summary": ("this registry row declares no credential "
                        "requirement"),
            "items": [],
        }
    named = "; ".join(
        f"{item['display_name']} ({item['location_display']}), needed for "
        f"{item['needed_for']}" for item in items)
    return {
        "required": True,
        "summary": f"this registry row declares {len(items)} "
                   f"credential{'' if len(items) == 1 else 's'}: {named}",
        "items": items,
    }
