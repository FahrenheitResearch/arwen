"""How many forcing leads a host-side decode holds at once, from HOST RAM.

A preparation decodes its forcing on the host whatever backend builds the
states afterwards, so the decode is budgeted against the RAM this process
can still take -- ``MemAvailable`` under every cgroup limit it runs in
(:func:`gpuwm.ingest.preparation_workers.host_available_bytes`) -- never
against the card.

Named breakage (the gate law): the 2026-10-04T12 and 2026-10-05T00 runs of
a 240 h, 81-lead GFS 0.25-degree template were SIGKILLed in their prepare
stage on a 64 GiB host beside a 96 GB card.  Their leads were whole-globe
``pgrb2.0p25`` objects (about 540 MB each); all 81 were ready at once, so
one bridge run parsed every one of them into memory (about 44 GB) and the
preparation then held all 81 decoded leads as float64 (about 1.1 GB
each), while the planner had priced the run against the card's 93.93 GiB.

The price of a lead batch, with ``n`` leads decoded on ``t`` threads:

* every lead's GRIB object, parsed and held for the whole bridge run
  (``n x lead_bytes``);
* per thread, one object being read beside its parse and one lead's
  decoded float32 arrays being written (``t x (lead_bytes + decoded)``);
* the preparation's own working copies of decoded leads -- the cached
  float64 snapshots and the transient of loading one -- which this module
  reserves as :data:`PREPARATION_RESIDENT_LEADS` float64 leads.

The whole is held to :data:`DECODE_HOST_SHARE` of the RAM available when
the batch starts.  The share is not all of it because the forecast stage
of a chained run starts beside the preparation and takes host RAM of its
own after the batch was sized, and because the fetch running beside an
as-posted preparation keeps writing.
"""
from __future__ import annotations

from dataclasses import dataclass

GIB = 1024 ** 3

#: Fraction of available host RAM one lead batch may be priced at.
DECODE_HOST_SHARE = 0.5

#: Decoded float64 leads the preparation itself holds beside a batch: the
#: two-entry snapshot cache plus the transient copy of the one being read.
PREPARATION_RESIDENT_LEADS = 3

#: GFS 0.25-degree whole-globe mesh, the mesh a full-file object decodes to.
GFS_0P25_GLOBAL_POINTS = 1440 * 721


@dataclass(frozen=True)
class DecodeWindow:
    """One sized lead batch: how many leads, on how many threads, why."""

    leads: int
    threads: int
    lead_bytes: int
    decoded_lead_bytes: int
    available_bytes: int | None
    budget_bytes: int | None

    @property
    def batch_bytes(self) -> int:
        """The batch's priced peak, bridge and preparation together."""

        return batch_price(self.leads, self.threads, self.lead_bytes,
                           self.decoded_lead_bytes)

    def sentence(self, total: int | None = None) -> str:
        of = "" if total is None else f" of {total}"
        if self.available_bytes is None:
            return (f"decode window {self.leads}{of} lead(s) on "
                    f"{self.threads} thread(s): host RAM unreadable, "
                    "unbounded")
        return (f"decode window {self.leads}{of} lead(s) on "
                f"{self.threads} thread(s), about "
                f"{self.batch_bytes / GIB:.2f} GiB, within "
                f"{DECODE_HOST_SHARE:.0%} of {self.available_bytes / GIB:.2f} "
                "GiB of host RAM available")


def batch_price(leads: int, threads: int, lead_bytes: int,
                decoded_lead_bytes: int) -> int:
    """Priced host peak of decoding ``leads`` leads on ``threads`` threads."""

    leads = max(1, int(leads))
    threads = max(1, min(int(threads), leads))
    lead = max(0, int(lead_bytes))
    decoded = max(0, int(decoded_lead_bytes))
    return (leads * lead + threads * (lead + decoded)
            + PREPARATION_RESIDENT_LEADS * 2 * decoded)


def available_host_bytes() -> int | None:
    """RAM this process can still take, or ``None`` when it cannot be read.

    ``MemAvailable`` (Windows: available physical memory), lowered to the
    headroom of every memory cgroup this process runs under, so a
    ``MemoryMax`` on the unit binds exactly as it will at the kill.
    """

    from gpuwm.ingest.preparation_workers import host_available_bytes

    try:
        return host_available_bytes()
    except Exception:  # an unreadable probe is unknown RAM, never a refusal
        return None


def decode_window(*, leads: int, threads: int, lead_bytes: int,
                  decoded_lead_bytes: int,
                  available: int | None) -> DecodeWindow:
    """The widest lead batch, and its thread count, that fits ``available``.

    Threads are narrowed first (a thread costs more than a held lead), then
    leads.  At least one lead on one thread is always returned: a single
    lead is the least any decode needs, and refusing it here would only
    move the failure; the planner says so before the download instead
    (:func:`plan_window`).  ``available=None`` (unreadable RAM) leaves the
    batch as wide as asked.
    """

    leads = max(1, int(leads))
    threads = max(1, int(threads))
    if available is None:
        return DecodeWindow(leads, min(threads, leads), int(lead_bytes),
                            int(decoded_lead_bytes), None, None)
    budget = int(DECODE_HOST_SHARE * max(0, int(available)))
    lead = max(1, int(lead_bytes))
    decoded = max(0, int(decoded_lead_bytes))
    fixed = PREPARATION_RESIDENT_LEADS * 2 * decoded
    per_thread = lead + decoded
    # One lead per thread at least: narrow threads until that fits.
    fit_threads = max(1, (budget - fixed) // (per_thread + lead))
    threads = max(1, min(threads, leads, int(fit_threads)))
    fit_leads = max(1, (budget - fixed - threads * per_thread) // lead)
    count = max(1, min(leads, int(fit_leads)))
    threads = min(threads, count)
    return DecodeWindow(count, threads, lead, decoded, int(available), budget)


def gfs_decoded_lead_bytes(fields_per_time: int,
                           points: int = GFS_0P25_GLOBAL_POINTS) -> int:
    """Float32 bytes one decoded GFS lead writes (the bridge's arrays)."""

    return 4 * int(points) * int(fields_per_time)


def plan_window(*, source: str, leads: int, threads: int,
                p_top_pa: float | None, available: int | None,
                ) -> DecodeWindow | None:
    """The planner's view of the decode window, before anything is fetched.

    Priced on the source's largest objects: for GFS a whole-globe
    0.25-degree ``pgrb2.0p25`` lead, which is what the full-file transport
    delivers; a NOMADS subregion crop is far smaller and decodes in a wider
    window.  ``None`` for a source this module has no lead price for.
    """

    if str(source).strip().lower() != "gfs":
        return None
    from gpuwm.core.preflight import source_analysis_fields_per_time

    fields = source_analysis_fields_per_time("gfs", p_top_pa=p_top_pa)
    return decode_window(
        leads=leads, threads=threads, lead_bytes=GFS_FULL_FILE_LEAD_BYTES,
        decoded_lead_bytes=gfs_decoded_lead_bytes(fields),
        available=available)


#: One whole-globe GFS 0.25-degree lead object as the s3 archive serves it
#: (measured on the 2026-10-05T00 cycle: 43.7 GB over 81 leads).
GFS_FULL_FILE_LEAD_BYTES = 540 * 1000 ** 2


def threads_available() -> int:
    """Cores this process may use (affinity and CPU quota)."""

    from gpuwm.ingest.preparation_workers import cpu_budget

    return int(cpu_budget()["available_cpus"])


__all__ = [
    "DECODE_HOST_SHARE", "PREPARATION_RESIDENT_LEADS", "GFS_0P25_GLOBAL_POINTS",
    "GFS_FULL_FILE_LEAD_BYTES", "DecodeWindow", "batch_price",
    "available_host_bytes", "decode_window", "gfs_decoded_lead_bytes",
    "plan_window", "threads_available",
]
