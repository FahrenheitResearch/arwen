"""Atomic, local static-web delivery of rendered forecast hours.

Immutable generations keep a reader's manifest valid while another hour lands.
The manifest replacement is the visibility boundary; no server or upload is
started here. An explicitly supplied command receives the manifest on stdin.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import html
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import threading
import uuid

PRIORITY_PRODUCTS = (
    "composite_reflectivity", "2m_temperature",
    "10m_wind_speed_and_direction", "qpf_1h", "mslp_10m_winds",
)
SCHEMA = "gpuwm.publish.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def instant(value) -> datetime:
    stamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)


def add_arguments(parser):
    parser.add_argument("--publish-root", type=Path, default=None, metavar="DIR",
                        help="local static web root (default: run output/www)")
    parser.add_argument("--publish-cmd", default=None, metavar="COMMAND",
                        help="optional delivery command; manifest JSON on stdin, no shell (default: off)")


def _atomic_text(path: Path, text: str):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _manifest_lock(root):
    """Serialize independent forecast processes sharing one static root."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".publish.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.write(b"\0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


class LocalPublisher:
    def __init__(self, root: Path, *, cycle=None, command=None, hydro="auto"):
        self.hydro = hydro
        self.root = Path(root).absolute()
        self.cycle = None if cycle is None else instant(cycle)
        self.command = shlex.split(command) if command else None
        self._lock = threading.Lock()
        self._hours = {}

    def publish(self, source: Path, pictures, *, domain: int, valid_time,
                hour_closed_at, render_start, render_end, complete: bool,
                timing_source="frame_submitted") -> dict:
        """Link a finished batch, then atomically advertise its immutable paths.

        Call only after the renderer closes its PNGs. A failed/incomplete five
        product set may be browsed, but never gets an ``online_at`` claim.
        """
        valid = instant(valid_time)
        with self._lock, _manifest_lock(self.root):
            manifest_path = self.root / "manifest.json"
            if manifest_path.exists():
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                if existing.get("schema") != SCHEMA:
                    raise ValueError(f"unsupported publication manifest: {manifest_path}")
                self._hours = {(h["cycle"], h["domain"], h["forecast_hour"]): h
                               for h in existing["hours"]}
            if self.cycle is None:
                self.cycle = valid
            cycle = self.cycle.strftime("%Y%m%dT%H%M%SZ")
            lead = (valid - self.cycle).total_seconds() / 3600
            hour = f"f{int(lead):02d}" if lead.is_integer() else f"f{lead:.3f}"
            generation = self.root / cycle / hour / f"d{domain:02d}" / uuid.uuid4().hex
            staging = generation.with_name("." + generation.name)
            staging.mkdir(parents=True)
            products = []
            try:
                for picture in pictures:
                    relative = Path(picture).relative_to(source)
                    target = staging / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        # A priority pass's scratch may be reused by the full
                        # render. Its public bytes must not share that inode.
                        if complete:
                            os.link(picture, target)
                        else:
                            shutil.copyfile(picture, target)
                    except OSError as error:
                        if error.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP):
                            raise
                        shutil.copyfile(picture, target)
                    products.append({"name": relative.as_posix(),
                                     "url": (generation.relative_to(self.root) / relative).as_posix(),
                                     "size_bytes": target.stat().st_size})
                _atomic_text(staging / "index.html", "<!doctype html><meta charset=utf-8>" +
                             "<title>Forecast products</title>" + "".join(
                                 f'<p><a href="{html.escape(p["name"], quote=True)}">{html.escape(p["name"])}</a></p>'
                                 for p in products))
                os.replace(staging, generation)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
            present = [slug for slug in PRIORITY_PRODUCTS if any(
                slug in Path(p["name"]).parts for p in products)]
            key = (cycle, int(domain), lead)
            previous = self._hours.get(key, {})
            # f00 has no preceding hour, hence no 1 h precipitation claim.
            required = [slug for slug in PRIORITY_PRODUCTS if lead > 0 or slug != "qpf_1h"]
            ready = all(slug in present for slug in required)
            _atomic_text(self.root / "index.html", _INDEX)
            now = utc_now()
            online = previous.get("online_at")
            row = dict(cycle=cycle, domain=int(domain), forecast_hour=lead,
                       valid_time=valid.isoformat(), hour_closed_at=hour_closed_at,
                       timing_source=timing_source, hydro=self.hydro, render_start=render_start,
                       render_end=render_end, online_at=online,
                       priority_render_start=previous.get("priority_render_start") or (render_start if ready else None),
                       priority_render_end=previous.get("priority_render_end") or (render_end if ready else None),
                       online_lag_seconds=(None if online is None or hour_closed_at is None else
                           (instant(online) - instant(hour_closed_at)).total_seconds()),
                       all_products_online_at=None,
                       products_count=len(products), priority_products=list(PRIORITY_PRODUCTS),
                       priority_present=present, priority_required=required,
                       complete=complete, products=products,
                       index=(generation.relative_to(self.root) / "index.html").as_posix())
            self._hours[key] = row
            manifest = dict(schema=SCHEMA, updated_at=now,
                            online_at_definition="UTC observed immediately after the atomic manifest replacement naming the products; a second atomic replacement records that observation",
                            hours=[self._hours[k] for k in sorted(self._hours)])
            # The static page always reads the authoritative manifest, so an
            # HTML update cannot reveal an hour ahead of its manifest.
            _atomic_text(self.root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
            # Observe visibility AFTER replacement, never charge a timestamp
            # sampled before a potentially slow filesystem write as online.
            # The first manifest already names every readable product; this
            # second replacement records when that publication was observed.
            committed = utc_now()
            if ready and online is None:
                row["online_at"] = committed
            if complete:
                row["all_products_online_at"] = committed
            if row["online_at"] is not None and hour_closed_at is not None:
                row["online_lag_seconds"] = (instant(row["online_at"]) - instant(hour_closed_at)).total_seconds()
            manifest["updated_at"] = committed
            _atomic_text(self.root / "manifest.json", json.dumps(manifest, indent=2) + "\n")
        if self.command:
            result = subprocess.run(self.command, input=json.dumps(row), encoding="utf-8",  # text-pipe-ok: a one-shot JSON value, no line protocol
                                    capture_output=True, timeout=120, check=False,
                                    env={**os.environ, "GPUWM_PUBLISH_ROOT": str(self.root),
                                         "GPUWM_PUBLISH_MANIFEST": str(self.root / "manifest.json")})
            if result.returncode:
                raise RuntimeError(f"publish command exited {result.returncode}: {result.stderr[-2000:]}")
        return row


_INDEX = '''<!doctype html><meta charset="utf-8"><title>Forecast products</title>
<h1>Forecast products</h1><main id="hours">Loading forecast hours…</main>
<script>
async function refresh() {
  const m = await (await fetch('manifest.json', {cache:'no-store'})).json();
  const root = document.getElementById('hours'); root.replaceChildren();
  for (const h of m.hours) {
    const section = document.createElement('section');
    const title = document.createElement('h2');
    title.textContent = `${h.cycle} · f${h.forecast_hour} · domain ${h.domain}`;
    section.append(title);
    for (const p of h.products) {
      const a = document.createElement('a'); a.href = p.url;
      const img = document.createElement('img'); img.src = p.url;
      img.alt = p.name; img.width = 360; img.loading = 'lazy'; a.append(img); section.append(a);
    }
    root.append(section);
  }
}
refresh(); setInterval(refresh, 10000);
</script>'''
