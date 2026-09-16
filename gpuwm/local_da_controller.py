"""Durable indexed analysis windows over the regional cycle owner.

A continuous local DA case is a bounded sequence of windows. Window ``i``
analyses at ``epoch + (i + 1) * cadence``: one forecast leg of the cadence
restarted from the previous analysis, the analysis itself, then the short
forecast and its products. Each window commits its own immutable
decisions (inputs, analysis, products, completion) and the controller keeps
one atomic head naming the last completed window. Forcing is renewed by the
backend when a window's products reach past the forcing the case has, and
the renewal is part of that window's input decision.

Nothing here integrates, decodes or renders; the backend composes the
existing owners. The controller decides which window is due, waits on the
clock or on unpublished forcing, stops on a durable request, and writes
``status.json`` for readers.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from fractions import Fraction
import math
import json
import os
import time
import uuid

from gpuwm.ensemble.analysis_commit import digest, read_record, write_record
from gpuwm.ensemble.manifest import write_json_atomically
from gpuwm.output_identity import file_record

SCHEMA = 'arwen.continuous-local-da.v1'
STATUS_SCHEMA = 'arwen.local-da-continuous-status.v1'
#: Every ``status`` value ``status.json`` can carry, in one place so the
#: protocol page and a reader's switch cannot disagree with the writer.
STATUSES = ('NOT_STARTED', 'PREPARING', 'WAITING_TIME', 'WAITING_FORCING', 'OBSERVATIONS',
            'ANALYZING', 'FORECASTING', 'RENDERING', 'READY', 'STOPPING', 'STOPPED',
            'COMPLETE', 'INTERRUPTED', 'FAILED')
TERMINAL_STATUSES = ('STOPPED', 'COMPLETE', 'INTERRUPTED', 'FAILED')
STAGE_PHASES = {'preparation': 'PREPARING', 'observations': 'OBSERVATIONS', 'analysis': 'ANALYZING',
                'forecast': 'FORECASTING', 'render': 'RENDERING'}


def capability_contract():
    return dict(supported=True, request_field='continuous_windows', review_field='continuous',
                status_schema=STATUS_SCHEMA, statuses=list(STATUSES), terminal_statuses=list(TERMINAL_STATUSES),
                status_command=['local-da', '--status', '{plan_path}'],
                stop_command=['local-da', '--stop', '{plan_path}'],
                resume_command=['local-da', '--launch', '{plan_path}'])


def review_contract(windows):
    return dict(enabled=True, windows=windows, status_schema=STATUS_SCHEMA,
                status_relative_path='continuous/status.json', control_relative_path='continuous/stop',
                window_relative_path='continuous/window_{index:06d}',
                product_policy='every-window-in-order',
                renewal_policy='the same source cycle over a longer window; the initial condition, the analysed state and every forcing frame already run under are retained')


def _saved_identity(path):
    """Cheap plan identity read; input validation belongs to attach/resume."""
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding='utf-8'))
    from gpuwm.local_da import SCHEMA as PLAN_SCHEMA
    unsigned = {k: v for k, v in document.items() if k not in ('review_sha256', 'files')}
    if (document.get('schema') != PLAN_SCHEMA or digest(unsigned) != document.get('review_sha256')
            or not document.get('continuous', {}).get('enabled')):
        raise ValueError('The saved document is not an unchanged continuous review; restore its original plan or publish one with --continuous')
    return path, document


def _controller_alive(path, owner):
    """A matching held OS lock proves ownership; status age cannot."""
    from gpuwm.fetch_guard import _try_lock, _unlock
    from gpuwm.mcp.gpulock import _pid_alive
    if not isinstance(owner, dict) or type(owner.get('pid')) is not int:
        return False
    try:
        with Path(path).open('r+b') as stream:
            if _try_lock(stream):
                _unlock(stream)
                return False
            stream.seek(1)
            recorded = json.loads(stream.read())
        return (recorded.get('run_id') == owner.get('nonce')
                and recorded.get('pid') == owner['pid'] and _pid_alive(owner['pid']))
    except (OSError, ValueError):
        return False


def status_for_plan(path):
    """Read the durable status of a saved continuous review without hashing products."""
    path, plan = _saved_identity(path)
    root = path.parent / 'continuous'
    target = root / 'status.json'
    windows = plan['continuous']['windows']
    value = (json.loads(target.read_text(encoding='utf-8')) if target.exists() else
             dict(schema=STATUS_SCHEMA, status='NOT_STARTED', completed_windows=0, windows=windows,
                  remaining_windows=windows, latest_products=None, active_window=None))
    if value.get('schema') != STATUS_SCHEMA or value.get('review_sha256', plan['review_sha256']) != plan['review_sha256']:
        raise ValueError('The status document belongs to another saved review; restore its original controller record')
    alive = _controller_alive(root / '.controller.lock', value.get('controller_owner'))
    if alive and type(value.get('sample_monotonic')) in (int, float):
        delta = max(0., time.monotonic()-value['sample_monotonic'])
        value['elapsed_seconds'] = value.get('elapsed_seconds', 0.) + delta
        value['current_stage_elapsed_seconds'] = value.get('current_stage_elapsed_seconds', 0.) + delta
    if not alive and value['status'] not in TERMINAL_STATUSES + ('NOT_STARTED',):
        value['status'] = 'INTERRUPTED'
    # The score is read off the window receipts, not off whatever the
    # controller last wrote: a later window fills in an earlier window's
    # pending leads, and the receipts are where that lands.
    # Same rule as the controller's own refresh: a receipt another process
    # wrote is read through summarize(), and a fault reading it must not
    # take down the door that reports whether the run is alive.
    try:
        from gpuwm.local_da_score import status_scores
        scores, score_error = status_scores(path.parent), None
    except Exception as error:
        scores, score_error = None, f'{type(error).__name__}: {error}'
    value.update(review_sha256=plan['review_sha256'], plan_path=str(path), status_path=str(target),
                 control_path=str(root / 'stop'), controller_alive=alive,
                 nowcast_score=scores,
                 **({'nowcast_score_error': score_error} if score_error else {}),
                 stop_requested=any((root / 'stop').glob('*.json')),
                 liveness_basis='held OS lock with matching controller owner identity')
    return value


def stop_plan(path):
    """Write a durable stop request; the controller honours it between operations."""
    path, plan = _saved_identity(path)
    directory = path.parent / 'continuous' / 'stop'
    write_record(directory / (uuid.uuid4().hex + '.json'), dict(review_sha256=plan['review_sha256']))
    value = status_for_plan(path)
    value['stop_requested'] = True
    if value['controller_alive']:
        value['status'] = 'STOPPING'
    return value


class ForcingUnavailable(RuntimeError):
    """The required forcing frames are not published yet."""


class WindowFailure(RuntimeError):
    """A window's operation failed; the message names the window and the reason."""
    code = 'CONTINUOUS_WINDOW_FAILED'

    def __init__(self, index, stage, error, *, forecast_started, recovery):
        where = f'window {index}' if index is not None else 'before the first window'
        stage_text = f' during {stage}' if stage else ''
        super().__init__(f'Continuous local DA stopped at {where}{stage_text}: {error}')
        self.window = index
        self.stage = stage
        self.details = dict(window=index, stage=stage, reason=str(error))
        self.forecast_started = forecast_started
        self.recovery = recovery


def _utc(value):
    value = datetime.fromisoformat(value) if isinstance(value, str) else value
    if value.tzinfo is None:
        raise ValueError('Continuous time requires an explicit UTC offset')
    return value.astimezone(timezone.utc)


def _verify_files(records):
    for record in records:
        if file_record(record['path']) != record:
            raise ValueError(f"{record['path']}: committed window bytes changed; restore the original artifact before continuing")


def _verify_products(value):
    if not isinstance(value, dict) or value.get('status') != 'complete' or not value.get('artifacts'):
        raise ValueError('The window has no complete verified products; finish its ordinary forecast and rendering before advancing')
    _verify_files(value['artifacts'])
    execution = next((a for a in value['artifacts'] if a['path'] == value.get('execution_path')), None)
    if execution is None or execution['sha256'] != value.get('execution_sha256'):
        raise ValueError('The product decision does not bind its immutable execution receipt')


class Controller:
    """One writer, immutable window decisions, and an atomic current head.

    Processing is serial: every requested analysis and product runs. Actual
    lag is reported without dropping windows or reducing scientific settings.
    A stop is durable and takes effect between complete operations. A process
    interruption resumes the active operation through its ordinary owner.
    ``windows`` bounds the sequence; the controller is COMPLETE once that
    many windows are committed.
    """
    def __init__(self, root, *, binding, epoch, cadence_seconds,
                 forecast_seconds, members, windows, now=None, monotonic=None, plan_path=None,
                 forcing_wait_seconds=3600.):
        from gpuwm.experiment import _check_whole_second_cadence
        for label, value in (('continuous analysis cadence', cadence_seconds),
                             ('continuous forecast duration', forecast_seconds)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{label} must be a positive finite number of seconds')
            _check_whole_second_cadence(label, Fraction(str(value)), 1, 'continuous cycling')
        if type(members) is not int or members < 1:
            raise ValueError('Continuous member count must be a positive integer')
        if type(windows) is not int or windows < 1:
            raise ValueError('A continuous run needs a positive whole number of windows; pass --continuous N with N at least 1')
        if type(forcing_wait_seconds) not in (int, float) or not math.isfinite(forcing_wait_seconds) or forcing_wait_seconds < 0:
            raise ValueError('The forcing wait must be a finite nonnegative number of seconds')
        self.root = Path(root).resolve()
        self.epoch = _utc(epoch)
        self.cadence = float(cadence_seconds)
        self.forecast = float(forecast_seconds)
        self.members = members
        self.windows = windows
        self.forcing_wait = float(forcing_wait_seconds)
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.monotonic = monotonic or time.monotonic
        self.binding = dict(schema=SCHEMA, inputs=binding, epoch=self.epoch.isoformat(),
                            cadence_seconds=self.cadence, forecast_seconds=self.forecast,
                            members=members, windows=windows, product_policy='every-window-in-order')
        self.identity = digest(self.binding)
        self.plan_path = Path(plan_path).resolve() if plan_path else self.root.parent / 'local-da.json'
        self.review_sha256 = binding.get('review_sha256', self.identity)
        self.owner = None
        self.started = self.monotonic()
        self.stage_started = self.started
        self.stage_seconds = {}
        self.current_stage = None
        self.last_head = dict(completed=0, latest=None)
        self.head_validated = None
        self.latest_products = None
        self.committed_analysis_time = None
        self.active_window = None
        self.elapsed_before = 0.
        self.total_stage_seconds = {}
        self.stage_details = {}
        self.timing_loaded = False
        self.forecast_started = False
        self.phase = None
        self.nowcast_score = None
        self.nowcast_error = None

    def _restore_timing(self):
        """Carry measured work across launches without inventing downtime."""
        if self.timing_loaded:
            return
        path = self.root / 'status.json'
        if path.exists():
            value = json.loads(path.read_text(encoding='utf-8'))
            if (value.get('schema') != STATUS_SCHEMA
                    or value.get('binding_sha256') != self.identity):
                raise ValueError('The saved timing status belongs to another continuous run')
            self.elapsed_before = float(value.get('elapsed_seconds', 0.))
            self.total_stage_seconds = dict(value.get('total_stage_seconds', {}))
            self.stage_seconds = dict(value.get('stage_seconds', {}))
            self.active_window = value.get('active_window')
            self.forecast_started = bool(value.get('forecast_started', False))
            readings = [self.elapsed_before, *self.total_stage_seconds.values(), *self.stage_seconds.values()]
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in readings):
                raise ValueError('Saved continuous timing is not finite and nonnegative')
        self.started = self.stage_started = self.monotonic()
        self.timing_loaded = True

    def _stage_totals(self, totals):
        result = dict(totals)
        if self.current_stage is not None:
            result[self.current_stage] = result.get(self.current_stage, 0.) + max(0., self.monotonic()-self.stage_started)
        return result

    def _finish_stage(self):
        if self.current_stage is not None:
            delta = max(0., self.monotonic()-self.stage_started)
            for totals in (self.stage_seconds, self.total_stage_seconds):
                totals[self.current_stage] = totals.get(self.current_stage, 0.) + delta
        self.current_stage = None

    def _claim(self):
        return write_record(self.root / 'binding.json', self.binding)

    def _head(self):
        path = self.root / 'head.json'
        if not path.exists():
            return dict(schema=SCHEMA, binding_sha256=self.identity, completed=0, latest=None)
        head = json.loads(path.read_text(encoding='utf-8'))
        if (head.get('schema') != SCHEMA or head.get('binding_sha256') != self.identity
                or type(head.get('completed')) is not int or head['completed'] < 1):
            raise ValueError('The continuous head belongs to another run or clock; restore its original record')
        expected = self.root / f"window_{head['completed'] - 1:06d}" / 'complete.json'
        if file_record(expected) != head.get('latest'):
            raise ValueError('The continuous head no longer names its committed window; restore the original receipt')
        result = read_record(expected)
        if (result.get('index') != head['completed'] - 1
                or result.get('binding_sha256') != self.identity):
            raise ValueError('The completed window contradicts its continuous head')
        if self.head_validated != head['latest']:
            self._validate_completion(result, result['index'], result['prior'])
            self.head_validated = head['latest']
        self.latest_products = {k:v for k,v in result['products'].items() if k != 'artifacts'}
        self.committed_analysis_time = result['analysis_utc']
        self.forecast_started = True
        return head

    def _analysis_time(self, index):
        return self.epoch + timedelta(seconds=(index + 1) * self.cadence)

    def _status(self, phase, head, **detail):
        if phase not in STATUSES:
            raise ValueError(f'unknown continuous status {phase!r}')
        self.last_head = head
        self.phase = phase
        value = dict(schema=STATUS_SCHEMA, binding_sha256=self.identity, status=phase,
                     windows=self.windows, completed_windows=head['completed'],
                     remaining_windows=max(0, self.windows - head['completed']), latest=head['latest'],
                     forecast_started=self.forecast_started,
                     review_sha256=self.review_sha256, plan_path=str(self.plan_path),
                     status_path=str(self.root / 'status.json'), control_path=str(self.root / 'stop'),
                     controller_owner=self.owner, controller_alive=self.owner is not None,
                     elapsed_seconds=self.elapsed_before+max(0., self.monotonic()-self.started),
                     session_elapsed_seconds=max(0., self.monotonic()-self.started),
                     timing_scope='measured controller time across launches; stopped process downtime excluded',
                     current_stage=self.current_stage,
                     current_stage_elapsed_seconds=max(0., self.monotonic()-self.stage_started),
                     stage_seconds=self._stage_totals(self.stage_seconds),
                     total_stage_seconds=self._stage_totals(self.total_stage_seconds),
                     sample_monotonic=self.monotonic(),
                     latest_products=self.latest_products,
                     nowcast_score=self.nowcast_score,
                     nowcast_score_error=self.nowcast_error,
                     committed_analysis_time=self.committed_analysis_time,
                     updated_utc=_utc(self.now()).isoformat())
        value.update(self.stage_details)
        value.update(detail)
        value.setdefault('active_window', self.active_window)
        value.setdefault('analysis_time', None if self.active_window is None else
                         self._analysis_time(self.active_window).isoformat())
        write_json_atomically(self.root / 'status.json', value)
        return value

    def stage(self, name, **detail):
        now = self.monotonic()
        if name != self.current_stage:
            self._finish_stage()
            self.current_stage, self.stage_started = name, now
            self.stage_details = {}
        self.stage_details.update(detail)
        return self._status(STAGE_PHASES[name], self.last_head, **detail)

    def run(self, backend, *, sleeper=time.sleep):
        """Serial production loop; a stop is durable and acknowledged once."""
        from gpuwm.supervisor import GPUFileLock
        self.root.mkdir(parents=True, exist_ok=True)
        nonce = uuid.uuid4().hex
        lock = GPUFileLock(self.identity, path=self.root / '.controller.lock', run_id=nonce)
        with lock:
            self.owner = dict(pid=os.getpid(), nonce=nonce)
            self._claim()
            try:
                self._restore_timing()
                acknowledged = self._stop_requests()
                backend.preflight()
                while True:
                    value = self.step(backend, acknowledged=acknowledged)
                    acknowledged = ()
                    if value['status'] in TERMINAL_STATUSES:
                        break
                    if value['status'] == 'WAITING_TIME':
                        remaining = (_utc(value['next_analysis_utc']) - _utc(self.now())).total_seconds()
                        sleeper(min(60., max(1., remaining)))
                    elif value['status'] == 'WAITING_FORCING':
                        sleeper(60.)
            except BaseException as error:
                interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
                # The backend names fine stages; the controller's own phase
                # stands in when the failure came before the backend did.
                stage = self.current_stage or {v: k for k, v in STAGE_PHASES.items()}.get(self.phase)
                self._status('INTERRUPTED' if interrupted else 'FAILED', self.last_head,
                             reason=str(error), failed_window=self.active_window, failed_stage=stage)
                if isinstance(error, Exception):
                    raise WindowFailure(self.active_window, stage, error,
                        forecast_started=self.forecast_started,
                        recovery='Prior completed windows, outputs and analysis checkpoints are retained. '
                                 'Launch this saved plan again to resume its unfinished window.') from error
                raise
            finally:
                backend.close()
                self.owner = None
                self._release_status()
        value['controller_alive'] = False
        return value

    def _release_status(self):
        """The document this process leaves behind says no controller holds it.

        Every status written while the lock was held says ``controller_alive``
        because it was true then; a reader of the raw file after the process
        exited would otherwise see a completed or failed run still claimed.
        ``--status`` decides liveness from the lock either way; this keeps the
        file itself from contradicting it. Written under the held lock, so it
        can never overwrite another controller's status.
        """
        path = self.root / 'status.json'
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return
        if value.get('schema') == STATUS_SCHEMA and value.get('binding_sha256') == self.identity:
            value['controller_alive'] = False
            write_json_atomically(path, value)

    def request_stop(self):
        self._claim()
        write_record(self.root / 'stop' / (uuid.uuid4().hex + '.json'), dict(binding_sha256=self.identity))

    def _stop_requests(self):
        return tuple((self.root / 'stop').glob('*.json'))

    def step(self, backend, *, resume=False, acknowledged=()):
        """Finish at most one durable window, or report its actual wait."""
        from gpuwm.supervisor import GPUFileLock
        acknowledged = self._stop_requests() if resume else acknowledged
        lock = GPUFileLock(self.identity, path=self.root / '.writer.lock', run_id=self.identity)
        self.root.mkdir(parents=True, exist_ok=True)
        with lock:
            self._claim()
            self._restore_timing()
            for path in acknowledged:
                path.unlink(missing_ok=True)
            head = self._head()
            self.last_head = head
            if head['completed'] >= self.windows:
                self._finish_stage()
                return self._status('COMPLETE', head, active_window=None,
                                    last_analysis_utc=self.committed_analysis_time)
            if self._stop_requests():
                return self._status('STOPPED', head)
            index = head['completed']
            if self.active_window != index:
                self._finish_stage()
                self.stage_seconds = {}
                self.stage_details = {}
            self.active_window = index
            when = self._analysis_time(index)
            if _utc(self.now()) < when:
                return self._status('WAITING_TIME', head, next_analysis_utc=when.isoformat())
            directory = self.root / f'window_{index:06d}'
            directory.mkdir(exist_ok=True)
            started = self.monotonic()
            prior = None if not head['latest'] else read_record(head['latest']['path'])
            if prior is not None:
                # Every new operation binds the current checkpoint bytes,
                # even when a previous read in this process was valid.
                _verify_files(prior['analysis'])
            complete_path = directory / 'complete.json'
            if complete_path.exists():
                completed = read_record(complete_path)
                self._validate_completion(completed, index, head['latest'])
            else:
                inputs_path = directory / 'inputs.json'
                try:
                    if inputs_path.exists():
                        inputs = read_record(inputs_path)
                        if (inputs.get('binding_sha256') != self.identity or inputs.get('index') != index
                                or inputs.get('prior') != head['latest']):
                            raise ValueError('The active window input decision contradicts its original lineage')
                        backend.restore_window(inputs['prepared'], directory)
                    else:
                        self._status('PREPARING', head, active_window=index)
                        prepared = backend.prepare_window(index, when, self.forecast, directory, prior=prior)
                        inputs = write_record(inputs_path, dict(schema=SCHEMA,
                            binding_sha256=self.identity, index=index, prior=head['latest'], prepared=prepared))
                except ForcingUnavailable as error:
                    lag = max(0., (_utc(self.now()) - when).total_seconds())
                    if lag > self.forcing_wait:
                        raise RuntimeError(f'the forcing for the analysis at {when.isoformat()} was still unpublished '
                                           f'{lag:.0f} s after that time, past the {self.forcing_wait:.0f} s wait '
                                           f'one forcing interval allows: {error}') from error
                    return self._status('WAITING_FORCING', head, active_window=index,
                        reason=str(error), next_analysis_utc=when.isoformat(), lag_seconds=lag,
                        wait_remaining_seconds=self.forcing_wait - lag)
                _verify_files(inputs['prepared']['assets'])
                if self._stop_requests():
                    return self._status('STOPPED', head, active_window=index)
                decision_path = directory / 'analysis.json'
                if decision_path.exists():
                    decision = read_record(decision_path)
                    if (decision.get('binding_sha256') != self.identity or decision.get('index') != index
                            or decision.get('prior') != head['latest']
                            or decision.get('inputs') != file_record(inputs_path)):
                        raise ValueError('The committed analysis contradicts this window or its input decision')
                    analysis = decision['analysis']
                else:
                    self._status('ANALYZING', head, active_window=index)
                    outcome = backend.analyze_window(index, directory, prior, inputs)
                    analysis = outcome['analysis']
                    if len(analysis) != self.members:
                        raise ValueError('The completed analysis has an incomplete member roster; recover every member before continuing')
                    # Product retries reuse this exact analysis decision.
                    decision = write_record(decision_path, dict(schema=SCHEMA,
                        binding_sha256=self.identity, index=index, prior=head['latest'],
                        inputs=file_record(inputs_path), analysis=analysis, outcome=outcome))
                _verify_files(analysis)
                if self._stop_requests():
                    return self._status('STOPPED', head, active_window=index)
                self._status('FORECASTING', head, active_window=index)
                products = backend.produce_window(index, directory, decision)
                _verify_products(products)
                self._finish_stage()
                completed = write_record(complete_path, dict(schema=SCHEMA,
                    binding_sha256=self.identity, index=index, prior=head['latest'],
                    analysis=analysis, analysis_decision=file_record(directory / 'analysis.json'),
                    products=products, analysis_utc=when.isoformat(),
                    nowcast_score=products.get('nowcast_score'),
                    forcing_generation=inputs['prepared'].get('forcing_generation', 0),
                    observation_usage=decision['outcome'].get('observation_usage'),
                    stage_seconds=dict(self.stage_seconds),
                    total_stage_seconds=dict(self.total_stage_seconds),
                    completed_utc=_utc(self.now()).isoformat(),
                    processing_seconds=max(0., self.monotonic() - started),
                    lag_seconds=max(0., (_utc(self.now()) - when).total_seconds()),
                    timing_scope='this attempt including preparation and products; interrupted work and downtime excluded'))
                self._validate_completion(completed, index, head['latest'])
            head = dict(schema=SCHEMA, binding_sha256=self.identity,
                        completed=index + 1, latest=file_record(complete_path))
            write_json_atomically(self.root / 'head.json', head)
            self.latest_products = {k:v for k,v in completed['products'].items() if k != 'artifacts'}
            self.committed_analysis_time = completed['analysis_utc']
            self.head_validated = head['latest']
            self._refresh_scores(backend)
            if head['completed'] >= self.windows:
                phase = 'COMPLETE'
            else:
                phase = 'STOPPED' if self._stop_requests() else 'READY'
            return self._status(phase, head,
                last_analysis_utc=completed['analysis_utc'],
                lag_seconds=completed['lag_seconds'], processing_seconds=completed['processing_seconds'],
                observation_usage=completed.get('observation_usage'))

    def _refresh_scores(self, backend):
        """Score what has become scoreable, then republish the live numbers.

        A nowcast lead cannot be scored before its own valid time, so a
        window's later leads are pending when that window commits and become
        scoreable while a later window runs. The backend fills them in and
        this reads the receipts back, so the status document carries the
        current model score, the persistence baseline and their difference
        rather than the snapshot taken at completion. Scoring never fails a
        run, so a failure here is recorded on the status and nothing else.
        """
        rescore = getattr(backend, 'rescore_pending', None)
        if rescore is not None:
            try:
                rescore()
            # Deliberately everything, the same rule score_window itself
            # keeps: a forecast that ran is a forecast that ran, and the
            # score is a number about the run rather than a gate on it.
            # The narrow tuple this used to name did not cover the walk:
            # rescore_pending and status_scores both read receipts through
            # summarize(), which indexes seven keys of a document another
            # process wrote, and score_case_root indexes the plan outside
            # score_window's own guard. A KeyError from either failed the
            # window commit over a number that is explicitly not a gate.
            except Exception as error:
                self.nowcast_error = f'{type(error).__name__}: {error}'
        try:
            from gpuwm.local_da_score import status_scores
            self.nowcast_score = status_scores(self.root.parent)
        except Exception as error:
            self.nowcast_error = f'{type(error).__name__}: {error}'

    def _validate_completion(self, value, index, prior):
        if value.get('binding_sha256') != self.identity or value.get('index') != index or value.get('prior') != prior:
            raise ValueError('The completed window belongs to another lineage or prior analysis')
        if len(value.get('analysis', ())) != self.members:
            raise ValueError('The completed window is missing analyzed members')
        _verify_files(value['analysis'] + [value['analysis_decision']])
        _verify_products(value['products'])
