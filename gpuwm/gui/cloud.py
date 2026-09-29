"""Cloud machines: start, stop and terminate from the Machines list, as table data.

A provider is an entry of ``providers.json``: the command lines of its
own CLI, filled from the machine row with ``{{field}}``.  Nothing here
holds a credential: every call runs the provider's CLI with the row's
``profile``, which is the user's own login on this computer.

Money is bounded on the machine itself, so it holds even when this
computer is off: the image's first boot installs a guard service
(``arwen-cap-guard``) that counts the minutes the machine runs in a file
on its own disk, across stops and starts, and shuts it down when that
count reaches the cap (the dollars left turned into minutes at the row's
price) or after ``idle_stop_min`` minutes with no gpuwm process and
nothing on the card.  The provider is told to treat a shutdown as a
stop, so a stopped machine costs only its disk.

The guard's count only grows.  The row keeps the money this computer has
counted (``spent_usd``), how much of the guard's count that money already
covers (``guard_minutes_charged``) and, while minutes are not yet settled,
when they began (``started_utc``) and the most they can be
(``cap_minutes_granted``).  Every start, stop and terminate settles first:
it reads the guard's count and charges only what was not charged before,
so a session is charged once whichever of them sees it.  When the count
cannot be read, the row is counted at the most the machine could have run
(the smaller of the time since ``started_utc`` and that bound), and the
next read replaces the bound with the real minutes.

Every start checks the machine's SSH host key under the instance's own
name (``HostKeyAlias``) in a gpuwm-owned ``known_hosts``, taking the key
from the instance's console, where the first boot prints it.  A start
whose cap cannot be set on the machine is a failed start: the machine is
stopped again, because a machine without its cap is a bill with no
ceiling.

Once a cloud machine is up, its address becomes the row's ``host`` and
it is an SSH machine like any other: the same check, the same forecasts
and renders.

``dry_run`` prints every call a start, stop or terminate would make,
with nothing run and nothing spent.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any

PROVIDERS_FILE = Path(__file__).resolve().parent / "providers.json"
PLACEHOLDER = re.compile(r"\{\{([a-z_]+)\}\}")
ACTIONS = ("start", "stop", "terminate")
CALL_TIMEOUT_S = 900.0
#: How long a new machine may take to print its host keys and answer SSH.
HOST_KEY_WAIT_S = 600.0
HOST_KEY_TICK_S = 15.0
KEYS_BEGIN = "-----BEGIN SSH HOST KEY KEYS-----"
KEYS_END = "-----END SSH HOST KEY KEYS-----"
KEY_TYPE = re.compile(r"^(ssh-[a-z0-9]+|ecdsa-sha2-[a-z0-9]+)$")


def providers(path: Path | None = None) -> dict[str, Any]:
    document = json.loads((path or PROVIDERS_FILE).read_text(encoding="utf-8"))
    return document.get("providers") or {}


def provider(kind: str, table: dict[str, Any] | None = None) -> dict[str, Any]:
    from .machines import MachineError

    table = providers() if table is None else table
    if kind not in table:
        raise MachineError(f"There is no cloud provider called {kind!r}.",
                           f"Known providers: {', '.join(sorted(table)) or 'none'}.", 400)
    return table[kind]


def _utc(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(text: Any) -> datetime | None:
    try:
        value = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _price(row: dict[str, Any]) -> float:
    try:
        return float(row.get("price_per_hour_usd") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def unsettled_minutes(row: dict[str, Any], now: datetime | None = None) -> float:
    """The most minutes the machine can have run since its last start that are not yet counted.

    The guard stops it at its grant, so the grant bounds it even when this
    computer never heard of the stop.
    """

    began = _parse(row.get("started_utc"))
    if began is None:
        return 0.0
    minutes = max(0.0, ((now or datetime.now(timezone.utc)) - began).total_seconds() / 60.0)
    granted = row.get("cap_minutes_granted")
    if granted not in (None, ""):
        minutes = min(minutes, float(granted))
    return minutes


def spent_since(row: dict[str, Any], now: datetime | None = None) -> float:
    return round(unsettled_minutes(row, now) / 60.0 * _price(row), 4)


def fields(row: dict[str, Any], table: dict[str, Any] | None = None,
           now: datetime | None = None) -> dict[str, Any]:
    """The row with the provider's defaults filled in and the cap turned into minutes."""

    from .machines import MachineError

    spec = provider(str(row.get("provider") or row.get("kind")), table)
    values: dict[str, Any] = {**spec.get("defaults", {}), **{k: v for k, v in row.items() if v not in (None, "")}}
    missing = [key for key in spec.get("needs", []) if values.get(key) in (None, "")]
    if missing:
        raise MachineError(f"{row.get('name')} needs {', '.join(missing)} before it can start.",
                           "Set them on the machine's row.", 400)
    try:
        price = float(values["price_per_hour_usd"])
        cap = float(values["spend_cap_usd"])
        spent = float(values.get("spent_usd") or 0.0)
    except (TypeError, ValueError) as error:
        raise MachineError("price_per_hour_usd, spend_cap_usd and spent_usd must be numbers.", "", 400) from error
    if price <= 0:
        # Without a price the cap cannot become a time limit, and a machine
        # with no time limit is a bill with no ceiling.
        raise MachineError("price_per_hour_usd must be above zero, or the spend cap cannot be enforced.",
                           "Look up the instance type's hourly price for the region and set it.", 400)
    unsettled = round(unsettled_minutes(row, now) / 60.0 * price, 4)
    left = cap - spent - unsettled
    if math.floor(left / price * 60.0) < 1:  # less than one minute left is no time to start in
        if unsettled > 0:
            raise MachineError(
                f"{row.get('name')} may have used its spend cap: it was started "
                f"{row.get('started_utc')} and not stopped through ArWen, so up to ${unsettled:.2f} more "
                f"is counted on top of ${spent:.2f} (cap ${cap:.2f}).",
                "Raise spend_cap_usd on its row to start it; the start then counts the minutes its guard "
                "really saw, and the rest of the raise is not spent.", 409)
        raise MachineError(f"{row.get('name')} has used its spend cap (${spent:.2f} of ${cap:.2f}).",
                           "Raise spend_cap_usd on its row to start it again.", 409)
    values["cap_minutes"] = int(math.floor(left / price * 60.0))
    values["guard_cap_minutes"] = _charged(row) + values["cap_minutes"]
    values["remaining_usd"] = round(left, 2)
    values["unsettled_usd"] = round(unsettled, 2)
    return values


def _charged(row: dict[str, Any]) -> int:
    """How much of the guard's count ``spent_usd`` already covers."""

    try:
        return max(0, int(row.get("guard_minutes_charged") or 0))
    except (TypeError, ValueError):
        return 0


def fill(words: list[str], values: dict[str, Any]) -> list[str]:
    def one(word: str) -> str:
        return PLACEHOLDER.sub(lambda m: str(values.get(m.group(1), "<" + m.group(1) + ">")), word)

    return [one(str(word)) for word in words]


def plan(row: dict[str, Any], action: str, *, table: dict[str, Any] | None = None,
         now: datetime | None = None) -> dict[str, Any]:
    """Every call ``action`` makes, in order, as argv lists and command lines."""

    from .jobs import display
    from .machines import MachineError

    if action not in ACTIONS:
        raise MachineError(f"A cloud machine can start, stop or terminate, not {action}.", "", 400)
    spec = provider(str(row.get("provider") or row.get("kind")), table)
    values = fields(row, table, now) if action == "start" else {**spec.get("defaults", {}), **row}
    if action != "start" and not row.get("instance_id"):
        raise MachineError(f"{row.get('name')} has no instance to {action}.", "Start it first.", 409)
    if action == "start":
        sequence = spec["sequences"]["start_existing" if row.get("instance_id") else "start_new"]
    else:
        sequence = spec["sequences"][action]
    values.setdefault("instance_id", "<from launch>")
    values.setdefault("address", "<from address>")
    values["user_data_file"] = "<a temporary file holding user_data>"
    steps = []
    for step in sequence:
        if step == "check":
            steps.append({"step": "check", "where": "this computer",
                          "command": f"gpuwm machines check {row.get('name')}",
                          "words": "The same check as any SSH machine: card, VRAM, gpuwm version, disk."})
            continue
        if step == "host_key":
            argv = fill(spec["calls"]["host_keys"], values)
            steps.append({"step": "host_key", "where": "this computer", "argv": argv, "command": display(argv),
                          "words": f"Read the machine's SSH host keys from its console, keep them under "
                                   f"{values['instance_id']} in gpuwm's own known_hosts, then wait for SSH."})
            continue
        if step == "settle":
            argv = fill(spec["read_used"], values)
            steps.append({"step": "settle", "where": "the machine, over SSH", "argv": argv,
                          "command": display(argv),
                          "words": "Read the minutes its guard has counted and add the ones not yet "
                                   "charged to spent_usd, then work out the cap again."})
            continue
        if step == "set_cap":
            argv = fill(spec["set_cap"], values)
            steps.append({"step": "set_cap", "where": "the machine, over SSH", "argv": argv,
                          "command": display(argv),
                          "words": f"Tell the machine's guard it may run {values.get('cap_minutes')} more min "
                                   f"(${values.get('remaining_usd')} left at ${values.get('price_per_hour_usd')}/h) "
                                   f"and stop after {values.get('idle_stop_min')} idle min. If this fails, "
                                   f"the machine is stopped again."})
            continue
        argv = fill(spec["calls"][step], values)
        for key, extra in (spec.get("optional_args", {}).get(step) or {}).items():
            if values.get(key) not in (None, ""):
                argv += fill(extra, values)
        steps.append({"step": step, "where": "this computer", "argv": argv, "command": display(argv)})
    document = {"machine": row.get("name"), "provider": row.get("provider") or row.get("kind"),
                "action": action, "steps": steps}
    if action == "start":
        document["user_data"] = fill([spec["user_data"]], values)[0]
        document["cap_minutes"] = values["cap_minutes"]
        document["remaining_usd"] = values["remaining_usd"]
        document["unsettled_usd"] = values["unsettled_usd"]
    return document


def cloud_state(row: dict[str, Any]) -> dict[str, Any]:
    """What the row itself says (no call is made)."""

    state = "stopped" if row.get("instance_id") else "not started"
    if row.get("host") and row.get("started_utc"):
        state = "starting"
    return {"cloud": {"provider": row.get("provider") or row.get("kind"), "instance_id": row.get("instance_id"),
                      "instance_type": row.get("instance_type"), "region": row.get("region"),
                      "spend_cap_usd": row.get("spend_cap_usd"), "spent_usd": row.get("spent_usd") or 0.0,
                      "unsettled_usd": round(spent_since(row), 2),
                      "cap_minutes_granted": row.get("cap_minutes_granted"),
                      "idle_stop_min": row.get("idle_stop_min")},
            "state": state, "detail": f"cloud machine, {state}"}


# ------------------------------------------------------------------ host keys

def known_hosts_path(registry: Any) -> Path:
    return Path(registry.file()).with_name("known_hosts")


def host_keys_from_console(text: str) -> list[str]:
    """``type key`` pairs from the block cloud-init prints on the console at boot."""

    keys: list[str] = []
    inside = False
    for raw in str(text or "").splitlines():
        # Some images prefix console lines with a tag such as "ec2: ".
        line = re.sub(r"^\S+:\s+(?=-----|ssh-|ecdsa-)", "", raw.strip())
        if line.startswith(KEYS_BEGIN):
            inside, keys = True, []
            continue
        if line.startswith(KEYS_END):
            inside = False
            continue
        parts = line.split()
        if inside and len(parts) >= 2 and KEY_TYPE.match(parts[0]):
            keys.append(f"{parts[0]} {parts[1]}")
    return keys


def write_known_hosts(path: Path, alias: str, keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    kept = [line for line in lines if line.split(" ", 1)[0] != alias]
    kept.extend(f"{alias} {key}" for key in keys)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(kept) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def has_alias(path: Path, alias: str) -> bool:
    try:
        return any(line.split(" ", 1)[0] == alias for line in path.read_text(encoding="utf-8").splitlines())
    except OSError:
        return False


# ------------------------------------------------------------------ running it

def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=CALL_TIMEOUT_S)


def _row(registry: Any, name: str) -> dict[str, Any]:
    return next((r for r in registry.rows() if r.get("name") == name), None) or {}


def execute(registry: Any, name: str, action: str, *, runner: Any = None,
            sleep: Any = time.sleep) -> dict[str, Any]:
    """Run ``action`` for real with the provider's CLI; the row records what came back.

    ``runner`` runs one provider argv (the provider's CLI by default);
    ``sleep`` waits between host key and SSH polls.
    """

    from .machines import MachineError

    run = runner or _run
    row = _row(registry, name)
    if not row:
        raise MachineError(f"There is no machine called {name!r}.", "", 404)
    spec = provider(str(row.get("provider") or row.get("kind")))
    if runner is None and shutil.which(spec["cli"]) is None:
        raise MachineError(f"The {spec['cli']} command is not installed on this computer.",
                           f"Install it and log in with {spec.get('credentials', 'your own account')}.", 500)
    document = plan(row, action)
    values = {**spec.get("defaults", {}), **row}
    if action == "start":
        values.update(fields(row))
    ran: list[dict[str, Any]] = []
    booted: list[datetime] = []
    settled_at: list[datetime] = []

    def call(step: str) -> str:
        argv = fill(spec["calls"][step], values)
        for key, extra in (spec.get("optional_args", {}).get(step) or {}).items():
            if values.get(key) not in (None, ""):
                argv += fill(extra, values)
        done = run(argv)
        if done.returncode != 0:
            raise MachineError(f"{spec['cli']} refused {step}: {str(done.stderr).strip()[-400:]}", "")
        return str(done.stdout).strip()

    def stop_after_failure(error: MachineError) -> MachineError:
        # The machine is up without a cap it can enforce: stop it and count its minutes.
        stopped = "so it was stopped again"
        try:
            call("stop")
        except MachineError as stop_error:
            stopped = (f"and stopping it failed too ({stop_error.message}); "
                       "stop it from the provider's own console now")
        # The minutes since the last charge stay unsettled, bounded by the
        # time up to now: the next settle reads the guard's real count
        # instead, and a terminate before then charges the bound.
        current = _row(registry, name)
        if settled_at:
            minutes = (datetime.now(timezone.utc) - settled_at[0]).total_seconds() / 60.0
        else:
            minutes = unsettled_minutes(current) + (datetime.now(timezone.utc) - booted[0]).total_seconds() / 60.0
        _freeze_bound(registry, name, minutes)
        registry.update(name, host=None)
        return MachineError(f"{name} started, but {error.message}, {stopped}.",
                            error.fix or "Check the image against the cloud image list in docs/dev/GUI-API.md.",
                            error.status)

    with tempfile.TemporaryDirectory(prefix="gpuwm-cloud-") as scratch:
        user_data = Path(scratch) / "user-data.sh"
        user_data.write_text(document.get("user_data", ""), encoding="utf-8", newline="\n")
        values["user_data_file"] = str(user_data)
        for step in document["steps"]:
            name_ = step["step"]
            try:
                if name_ == "check":
                    ran.append({"step": "check", "result": registry.probe(name, fresh=True).get("state")})
                    continue
                if name_ == "host_key":
                    ran.append({"step": name_, **_host_key(registry, name, values, call, sleep)})
                    continue
                if name_ == "settle":
                    ran.append({"step": name_, **_settle(registry, name, spec, values, action,
                                                        booted[0] if booted else None)})
                    settled_at.append(datetime.now(timezone.utc))
                    continue
                if name_ == "set_cap":
                    ran.append({"step": name_, **_set_cap(registry, name, spec, values)})
                    continue
                output = call(name_)
            except MachineError as error:
                if booted and action == "start":
                    raise stop_after_failure(error) from error
                raise
            if name_ in ("launch", "start"):
                booted.append(datetime.now(timezone.utc))
            key = spec.get("outputs", {}).get(name_)
            if key:
                values[key] = output
                if key == "instance_id":
                    registry.update(name, instance_id=values["instance_id"])
                if key == "address":
                    registry.update(name, host=f"{values.get('ssh_user', 'ubuntu')}@{values['address']}")
            ran.append({"step": name_, "exit_code": 0})
    if action == "stop":
        registry.update(name, host=None)
    if action == "terminate":
        # No later settle reads this machine: what is still unsettled is charged at its bound.
        current = _row(registry, name)
        spent = float(current.get("spent_usd") or 0.0) + spent_since(current)
        registry.update(name, spent_usd=round(spent, 2), started_utc=None, host=None, cap_minutes_granted=None,
                        instance_id=None, host_key_alias=None, guard_minutes_charged=None)
    return {"ok": True, "machine": name, "action": action, "ran": ran}


def _host_key(registry: Any, name: str, values: dict[str, Any], call: Any, sleep: Any) -> dict[str, Any]:
    from .machines import MachineError

    alias = str(values["instance_id"])
    path = known_hosts_path(registry)
    fetched = False
    if not has_alias(path, alias):
        waited = 0.0
        while True:
            keys = host_keys_from_console(call("host_keys"))
            if keys or waited >= HOST_KEY_WAIT_S:
                break
            sleep(HOST_KEY_TICK_S)
            waited += HOST_KEY_TICK_S
        if not keys:
            raise MachineError(f"its console showed no SSH host keys within {int(HOST_KEY_WAIT_S)} s",
                               "The image must print its host keys to the console at boot (cloud-init's default).",
                               502)
        write_known_hosts(path, alias, keys)
        fetched = True
    registry.update(name, known_hosts=str(path), host_key_alias=alias)
    waited = 0.0
    while True:
        try:
            done = registry.get(name).run(["true"], timeout=30)
            if done.returncode == 0:
                break
            last = done.stderr.decode("utf-8", "replace").strip()[-300:]
        except MachineError as error:
            last = error.message
        lowered = last.lower()
        if "host key verification failed" in lowered or "identification has changed" in lowered:
            raise MachineError(f"its SSH host key is not the one its console printed ({last})",
                               "Another machine may be answering on that address; do not use it.", 502)
        if waited >= HOST_KEY_WAIT_S:
            raise MachineError(f"it did not answer SSH within {int(HOST_KEY_WAIT_S)} s ({last})",
                               "Check that the security group lets SSH in from this computer.", 502)
        sleep(HOST_KEY_TICK_S)
        waited += HOST_KEY_TICK_S
    return {"alias": alias, "known_hosts": str(path), "fetched": fetched}


def _freeze_bound(registry: Any, name: str, minutes: float) -> None:
    """Leave ``minutes`` unsettled, with a bound that stops growing one minute on (the stop itself)."""

    minutes = max(0.0, minutes)
    began = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    registry.update(name, started_utc=_utc(began), cap_minutes_granted=int(math.ceil(minutes)) + 1)


def _read_used(registry: Any, name: str, spec: dict[str, Any], values: dict[str, Any]) -> int | None:
    if not _row(registry, name).get("host"):
        return None  # stopped through ArWen already: nothing answers
    try:
        done = registry.get(name).run(fill(spec["read_used"], values), timeout=60)
    except Exception:  # noqa: BLE001 - an unreadable count falls back to the bound
        return None
    text = done.stdout.decode("utf-8", "replace").strip() if done.returncode == 0 else ""
    return int(text) if text.isdigit() else None


def _settle(registry: Any, name: str, spec: dict[str, Any], values: dict[str, Any], action: str,
            booted: datetime | None) -> dict[str, Any]:
    """Charge the minutes the guard counted that ``spent_usd`` does not cover yet.

    The guard's count only grows and ``guard_minutes_charged`` is how much
    of it is charged, so whichever of a start, stop or terminate reads it
    charges each minute once.
    """

    row = _row(registry, name)
    price = _price(values)
    used = _read_used(registry, name, spec, values)
    changes: dict[str, Any] = {}
    if used is not None:
        charged = _charged(row)
        # A count below the charged one is a guard that started over (a new
        # disk): all of it is new.
        minutes, source = float(used - charged if used >= charged else used), "guard"
        changes.update(guard_minutes_charged=used, started_utc=None, cap_minutes_granted=None)
    elif action == "start":
        # A start that cannot read the count charges the most the machine can
        # have run: whatever was unsettled plus this boot.  Its set_cap then
        # fails the same way and the start is stopped.
        minutes, source = unsettled_minutes(row), "bound"
        if booted is not None:
            minutes += (datetime.now(timezone.utc) - booted).total_seconds() / 60.0
        changes.update(started_utc=None, cap_minutes_granted=None)
    else:
        # Stop or terminate: the minutes stay unsettled until a later read.  A
        # stop freezes the bound at now; a terminate charges it (execute does).
        minutes, source = 0.0, "none"
    spent = round(float(row.get("spent_usd") or 0.0) + minutes / 60.0 * price, 2)
    changes["spent_usd"] = spent
    registry.update(name, **changes)
    if action == "stop":
        # The guard keeps counting until the machine is down: up to one more minute.
        after = _row(registry, name)
        _freeze_bound(registry, name, unsettled_minutes(after) if after.get("started_utc") else 0.0)
    values["spent_usd"] = spent
    if action == "start":
        left = float(values["spend_cap_usd"]) - spent
        values["remaining_usd"] = round(left, 2)
        values["cap_minutes"] = max(0, int(math.floor(left / price * 60.0)))
        values["guard_cap_minutes"] = _charged(_row(registry, name)) + values["cap_minutes"]
    return {"minutes": round(minutes, 1), "from": source, "spent_usd": spent,
            "cap_minutes": values.get("cap_minutes")}


def _set_cap(registry: Any, name: str, spec: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    from .machines import MachineError

    if int(values["cap_minutes"]) <= 0:
        raise MachineError("its spend cap is used up, so there is no time left to give its guard",
                           "Raise spend_cap_usd on its row to start it again.", 409)
    done = registry.get(name).run(fill(spec["set_cap"], values), timeout=120)
    if done.returncode != 0:
        why = done.stderr.decode("utf-8", "replace").strip()[-300:] or f"exit {done.returncode}"
        raise MachineError(f"its spend cap could not be set on it ({why})",
                           "The image must have the cap guard installed and let the SSH user run sudo "
                           "without a password.", 502)
    # Settle charged the guard's count up to its read, so the unsettled
    # minutes start now; the guard's count keeps growing from where it is.
    registry.update(name, started_utc=_utc(), cap_minutes_granted=int(values["cap_minutes"]))
    return {"exit_code": 0, "cap_minutes": int(values["cap_minutes"])}


__all__ = ["ACTIONS", "cloud_state", "execute", "fields", "fill", "host_keys_from_console", "plan",
           "provider", "providers", "spent_since", "unsettled_minutes", "write_known_hosts"]
