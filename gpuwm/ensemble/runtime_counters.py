"""Rain-counter observations on the original frozen runtime schedule.

This observer owns no model clock. It reads the elapsed value carried by the
original state or streamed store and advertises the original history dates.
"""
from __future__ import annotations

from datetime import timedelta


class FixedRuntimeCounterObserver:
    def __init__(self, capture, prepared, stepper, *, start_time, domain_id,
                 run_seconds, outer_steps, output_outer_steps,
                 history_begin_step, history_end_step, write_final_output):
        from gpuwm.ensemble.output_counters import CounterDeadlineCalendar
        from gpuwm.runtime import history_output_due
        self.capture, self.prepared, self.stepper = capture, prepared, stepper
        self.start_time, self.domain_id = start_time, int(domain_id)
        self.collector = getattr(capture.callback, "__self__", capture.callback)
        cfg = prepared.cfg
        history = ([start_time] if history_begin_step == 0 else [])
        final_step = outer_steps if write_final_output else None
        for outer_step in range(outer_steps):
            if history_output_due(outer_step, output_outer_steps,
                    final_outer_step=final_step, history_begin_outer_step=history_begin_step,
                    history_end_outer_step=history_end_step):
                history.append(start_time + timedelta(seconds=(outer_step + 1) * cfg.dt))
        self.calendar = CounterDeadlineCalendar.from_valid_times(grid_id=self.domain_id,
            start_time=start_time, run_end_time=start_time + timedelta(seconds=run_seconds),
            history_valid_times=tuple(history))
        capture.counter_calendars[(self.domain_id, id(self))] = self.calendar

    def _metadata(self):
        from gpuwm.runtime import _metadata_frame
        return _metadata_frame(self.prepared.grid, self.prepared.static_fields)

    def observe(self):
        from gpuwm.core.streaming import is_streaming
        from gpuwm.ensemble.output_counters import _resident_counters, _streamed_counters
        streamed = is_streaming(self.stepper)
        elapsed = (self.stepper.scalars["elapsed_seconds"] if streamed
                   else self.prepared.initial_result.state.elapsed_seconds)
        valid_time = self.start_time + timedelta(seconds=float(elapsed))
        delta = valid_time - self.start_time
        ticks = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
        due, _ = self.calendar.observe(ticks, member_id=self.capture.member_id)
        if not due:
            return
        fields, zeros, reason = (_streamed_counters(self.stepper, self.prepared.cfg) if streamed
            else _resident_counters(self.prepared.initial_result.state, self.prepared.cfg))
        if reason is not None:
            self.calendar.mark_unavailable(ticks, reason, member_id=self.capture.member_id)
            return
        metadata = self._metadata()
        captured = self.collector.capture_rain_counters(fields, valid_time=valid_time,
            grid_id=self.domain_id, episode=0, member_id=self.capture.member_id,
            latitude=metadata["XLAT"], longitude=metadata["XLONG"],
            absent_zero_fields=tuple(sorted(zeros - fields.keys())))
        if captured:
            self.calendar.mark_captured(ticks, member_id=self.capture.member_id)
        else:
            self.calendar.mark_unavailable(ticks, "the actual precipitation provider supplied no counter snapshot",
                                           member_id=self.capture.member_id)

    def history_consumed(self, valid_time):
        """Account for the original synchronous history consumer's snapshot."""
        has_counter = getattr(self.collector, "has_rain_counter", None)
        if not callable(has_counter):
            return
        metadata = self._metadata()
        if has_counter(valid_time=valid_time, grid_id=self.domain_id, episode=0,
                member_id=self.capture.member_id, latitude=metadata["XLAT"], longitude=metadata["XLONG"]):
            delta = valid_time - self.start_time
            ticks = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
            self.calendar.mark_captured(ticks, member_id=self.capture.member_id)


def fixed_counter_observer_for_current(prepared, stepper, **options):
    """An ordinary run adds no calendar, metadata work or counter transfer."""
    from gpuwm.ensemble.runtime_context import current_capture
    capture = current_capture()
    if capture is None:
        return None
    owner = getattr(capture.callback, "__self__", capture.callback)
    if not callable(getattr(owner, "capture_rain_counters", None)):
        return None
    return FixedRuntimeCounterObserver(capture, prepared, stepper, **options)


__all__ = ["FixedRuntimeCounterObserver", "fixed_counter_observer_for_current"]
