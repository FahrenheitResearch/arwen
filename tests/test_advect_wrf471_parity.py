"""Advection against the compiled, unmodified WRF v4.7.1 routines.

Every defined output word is measured.  The CUDA words, distances and counts
are pinned to their receipt for exact equality, rather than accepted under
an arbitrary tolerance.  Optional native h/z channels have independent
diagnostic stores; production total tendencies use ordinary launchers.
"""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.verify.advect_oracle import (
    ADVECT_ORACLE_DIR, ROUTINES, SUPPORTED_ROUTINES, SUPPORTED_OUTPUTS,
    SENTINEL, load_advect_cases, validate_case, write_fortran_input,
    defined_output_mask, measure_words, measure_advect_parity,
    arithmetic_control, advect_port_outputs, w_lid_control_cases,
    mapped_radiation_control_cases, advect_gpu_identity,
    architecture_is_certified, check_receipt_format, receipt_staleness,
    check_uncertified_stale_records, uncertified_stale_report,
    UncertifiedReceiptStale, UNCERTIFIED_STALE_KEY)

#: The modules every receipt of this fixture pins by assembled source.
RECEIPT_MODULES = ("advection", "pd_advection", "openbc")


def _receipt_index(directory):
    """The fixture's gpu-receipts.json: card name and "major.minor" -> receipt.

    One file read by these tests and written by
    tools/advect_wrf471_oracle/recapture_receipt.py, so a capture installed
    for a new card is replayed here without a test edit.
    """
    index = json.loads((directory / "gpu-receipts.json").read_text(encoding="ascii"))
    assert index["schema_version"] == 1
    architectures = {tuple(int(part) for part in key.split(".")): name
                     for key, name in index["architectures"].items()}
    return dict(index["cards"]), architectures


GPU_RECEIPTS, ARCH_RECEIPTS = _receipt_index(ADVECT_ORACLE_DIR)
RECEIPT_INDEX = json.loads((ADVECT_ORACLE_DIR / "gpu-receipts.json").read_text(encoding="ascii"))


def _receipt(name="gpu-receipt.json"):
    return json.loads((ADVECT_ORACLE_DIR / name).read_text(encoding="ascii"))


def _receipt_for_gpu(gpu_name, compute_capability=None):
    """Use measured card or architecture words; runtime equality remains exact."""
    name = GPU_RECEIPTS.get(gpu_name)
    card_receipt = name is not None
    architecture = tuple(compute_capability) if compute_capability is not None else None
    if name is None:
        name = ARCH_RECEIPTS.get(architecture)
    assert name is not None, (
        f"Compiled WRF advection tests have no recorded measurements for GPU {gpu_name!r} "
        f"or architecture {architecture}; receipt mappings cover {sorted(GPU_RECEIPTS)} "
        f"and {sorted(ARCH_RECEIPTS)}. This is a test coverage gap."
    )
    assert (ADVECT_ORACLE_DIR / name).is_file(), (
        f"Compiled WRF advection receipt {name!r} is missing for GPU {gpu_name!r}. "
        "Record the output words before replaying this measurement."
    )
    receipt = _receipt(name)
    if card_receipt:
        assert receipt["gpu"] == gpu_name, (name, receipt["gpu"], gpu_name)
    if architecture is not None:
        assert tuple(receipt["compute_capability"]) == architecture, (name, architecture)
    return receipt


def _device_receipt():
    """The receipt this card's runtime words are compared with.

    A Blackwell or newer card gates exactly as before: no receipt is a
    coverage gap, a stale receipt fails.  An older card is not certified
    (ruling 2026-10-04): with no receipt, or with one that trails the tree,
    the word comparison is skipped and says why; a current receipt still
    compares every word.
    """
    gpu, capability = advect_gpu_identity()
    if not architecture_is_certified(capability):
        if GPU_RECEIPTS.get(gpu) is None and ARCH_RECEIPTS.get(tuple(capability)) is None:
            pytest.skip(f"{gpu} (sm_{capability[0]}{capability[1]}) has no advection receipt and is "
                        "not certified: byte identity is certified on Blackwell and newer only "
                        "(ruling 2026-10-04)")
        receipt = _receipt_for_gpu(gpu, capability)
        name = GPU_RECEIPTS.get(gpu) or ARCH_RECEIPTS[tuple(capability)]
        moved = receipt_staleness(receipt, ADVECT_ORACLE_DIR, RECEIPT_MODULES)
        if moved:
            pytest.skip(uncertified_stale_report(
                name, receipt, moved, RECEIPT_INDEX.get(UNCERTIFIED_STALE_KEY, {}).get(name)))
        return receipt
    return _receipt_for_gpu(gpu, capability)


def _available_receipts():
    """The primary receipt is required; collected extra-card receipts are checked too."""
    for gpu_name, name in GPU_RECEIPTS.items():
        if name == "gpu-receipt.json" or (ADVECT_ORACLE_DIR / name).is_file():
            yield name, _receipt_for_gpu(gpu_name)


def test_advect_gpu_receipt_selection_rejects_unrecorded_architectures():
    with pytest.raises(AssertionError, match="no recorded measurements"):
        _receipt_for_gpu("unmeasured GPU")
    with pytest.raises(AssertionError, match="no recorded measurements"):
        _receipt_for_gpu("unmeasured GPU", (13, 0))


def test_advect_sm120_selection_preserves_the_measured_card_provenance():
    receipt = _receipt_for_gpu("sm_120 architecture replay", (12, 0))
    assert receipt["gpu"] == "NVIDIA GeForce RTX 5090"
    assert receipt["compute_capability"] == [12, 0]
    with pytest.raises(AssertionError):
        _receipt_for_gpu("NVIDIA GeForce RTX 5090", (13, 0))


def test_advect_receipts_certify_blackwell_and_report_older_cards_with_their_staling_commit(monkeypatch):
    """Byte identity is certified on Blackwell and newer only (ruling 2026-10-04).

    The breakage this prevents, both ways: an uncertified receipt (the H100,
    left behind by b70a94a48) holding the receipt-pin test red, and a
    staleness record quietly excusing a Blackwell pin or outliving the
    capture that brought its receipt current.  Nothing here depends on which
    uncertified receipt trails the tree today.
    """
    import gpuwm.verify.advect_oracle as oracle
    assert [architecture_is_certified(cc) for cc in ((8, 6), (8, 9), (9, 0), (10, 0), (12, 0), (13, 0))] == [
        False, False, False, True, True, True]
    assert not architecture_is_certified(None)
    check_uncertified_stale_records(RECEIPT_INDEX, ADVECT_ORACLE_DIR, RECEIPT_MODULES)
    h100 = _receipt("gpu-receipt-h100.json")
    moved = {"advection": (h100["kernels"]["advection"], "a" * 64), "openbc": (h100["kernels"]["openbc"], "c" * 64)}
    record = {"staled_by": "b70a94a4869e1dcf65149c0701e83b54208a6a07",
              "pins": {"advection": h100["kernels"]["advection"]}}
    report = uncertified_stale_report("gpu-receipt-h100.json", h100, moved, record)
    assert "sm_90" in report and "ruling 2026-10-04" in report and "informational" in report
    assert "advection e3fa6135 -> aaaaaaaa (staled by b70a94a48)" in report
    assert "openbc 5bf7babe -> cccccccc (staling commit not recorded" in report
    stale = {"staled_by": "b" * 40}
    certified = _receipt("gpu-receipt-sm120.json")
    index = dict(RECEIPT_INDEX, **{UNCERTIFIED_STALE_KEY: {"gpu-receipt-sm120.json": dict(
        stale, pins={"advection": certified["kernels"]["advection"]})}})
    with pytest.raises(AssertionError, match="certified"):
        check_uncertified_stale_records(index, ADVECT_ORACLE_DIR, RECEIPT_MODULES)
    index = dict(RECEIPT_INDEX, **{UNCERTIFIED_STALE_KEY: {"gpu-receipt-h100.json": dict(
        stale, pins={"advection": "0" * 64})}})
    with pytest.raises(AssertionError, match="drop or rewrite"):
        check_uncertified_stale_records(index, ADVECT_ORACLE_DIR, RECEIPT_MODULES)
    index = dict(RECEIPT_INDEX, **{UNCERTIFIED_STALE_KEY: {"gpu-receipt-h100.json": dict(
        stale, pins={"advection": h100["kernels"]["advection"]})}})
    monkeypatch.setattr(oracle, "receipt_staleness", lambda *args: {})
    with pytest.raises(AssertionError, match="current again"):
        check_uncertified_stale_records(index, ADVECT_ORACLE_DIR, RECEIPT_MODULES)


def test_advect_device_receipt_gates_blackwell_and_skips_older_cards_only_when_stale(monkeypatch):
    """The device half of the 2026-10-04 ruling, without a card.

    The breakage this prevents: the device word tests failing on an
    uncertified card whose receipt trails the tree (or that has none), and
    the converse, a Blackwell card losing its coverage-gap failure or its
    word comparison.  A current uncertified receipt still compares words.
    """
    module = sys.modules[__name__]

    def card(name, capability, moved=None):
        monkeypatch.setattr(module, "advect_gpu_identity", lambda: (name, capability))
        monkeypatch.setattr(module, "receipt_staleness", lambda *args: dict(moved or {}))

    stale = {"advection": ("e" * 64, "a" * 64)}
    card("NVIDIA GeForce RTX 5090", (12, 0), stale)
    assert _device_receipt()["gpu"] == "NVIDIA GeForce RTX 5090"
    card("unmeasured Blackwell", (10, 0))
    with pytest.raises(AssertionError, match="test coverage gap"):
        _device_receipt()
    card("unmeasured Ampere", (8, 6))
    with pytest.raises(pytest.skip.Exception, match="not certified"):
        _device_receipt()
    card("NVIDIA H100 80GB HBM3", (9, 0), stale)
    with pytest.raises(pytest.skip.Exception, match="informational, not a gate"):
        _device_receipt()
    card("NVIDIA H100 80GB HBM3", (9, 0))
    assert _device_receipt()["gpu"] == "NVIDIA H100 80GB HBM3"


def test_advect_fixture_has_real_source_and_exact_input_receipts():
    manifest = json.loads((ADVECT_ORACLE_DIR / "cases.json").read_text(encoding="ascii"))
    assert manifest["source_sha256"] == "67d6337044d673024e6a002a0d97a7876afe03d6e6cb62bd3061b55caf11287d"
    rows = {row["name"]: row for row in manifest["cases"]}
    for case in load_advect_cases():
        validate_case(case)
        row = rows[case.name]
        assert hashlib.sha256((ADVECT_ORACLE_DIR / row["file"]).read_bytes()).hexdigest() == row["input_sha256"]
        for key, values in case.inputs.items():
            assert hashlib.sha256(values.tobytes()).hexdigest() == row["array_sha256"][key], (case.name, key)
        assert case.shape == (49, 24, 24)
    cases = {case.name: case for case in load_advect_cases()}
    assert cases["real_west_boundary"].metadata["specified"]
    assert not cases["real_open_boundary"].metadata["specified"]
    assert np.max(cases["southern_reflection"].inputs["latitude"]) < 0
    assert np.min(cases["real_interior"].inputs["latitude"]) > 0
    zero = cases["zero_nearzero_tracers"].inputs["q0"]
    assert np.count_nonzero(zero == 0)
    assert np.count_nonzero((zero > 0) & (zero < np.finfo(np.float32).tiny))


def test_advect_fortran_abi_preserves_staggering_and_mass_split(tmp_path):
    case = load_advect_cases()[0]
    path = tmp_path / "pd.in.bin"
    initial = write_fortran_input(case, "advect_scalar_pd", path)
    words = np.fromfile(path, dtype="<i4", count=27)
    assert tuple(words[:8]) == (1, 5, 1, 25, 1, 25, 1, 50)
    assert tuple(words[8:14]) == (-3, 28, -3, 28, 1, 50)
    assert words[26] == 1
    shape = initial.shape
    count = int(np.prod(shape))
    raw = np.fromfile(path, dtype="<f4", offset=27 * 4 + 3 * 4)
    field = raw[:count].reshape(shape[1], shape[0], shape[2]).transpose(1, 0, 2)
    np.testing.assert_array_equal(field[:49, 4:28, 4:28].view(np.uint32), case.inputs["scalar_pd"].view(np.uint32))
    offset = 7 * count
    nplane = shape[1] * shape[2]
    mu_old_native = raw[offset + 2 * nplane:offset + 3 * nplane].reshape(shape[1:])
    np.testing.assert_array_equal(mu_old_native[4:28, 4:28].view(np.uint32), case.inputs["mu_perturbation"].view(np.uint32))
    assert np.all(initial[:, :4] == SENTINEL)


def test_advect_word_metric_does_not_hide_signed_zero():
    row = measure_words(np.array([-0.0], np.float32), np.array([0.0], np.float32))
    assert row["max_ulp"] == 0
    assert row["different_words"] == row["signed_zero_words"] == 1


def test_advect_native_reference_and_cuda_words_match_their_receipts():
    """Blackwell receipts gate their pins; older ones are reported when stale.

    Byte identity is certified on Blackwell and newer only (ruling
    2026-10-04).  Every receipt is checked for presence and format and its
    committed words replay against the reference; only a Blackwell
    receipt's kernel and fixture pins must equal the tree.
    """
    from gpuwm.core.kernels import module_source
    records = check_uncertified_stale_records(RECEIPT_INDEX, ADVECT_ORACLE_DIR, RECEIPT_MODULES)
    cases = load_advect_cases()
    for name, receipt in _available_receipts():
        check_receipt_format(receipt, ADVECT_ORACLE_DIR, RECEIPT_MODULES, cases)
        if not architecture_is_certified(receipt["compute_capability"]):
            moved = receipt_staleness(receipt, ADVECT_ORACLE_DIR, RECEIPT_MODULES)
            if moved:
                warnings.warn(UncertifiedReceiptStale(
                    uncertified_stale_report(name, receipt, moved, records.get(name))))
            if "fixture" in moved:
                # Its words were measured against another fixture's reference.
                continue
        else:
            for module in RECEIPT_MODULES:
                # Exactly the source the default loader compiles. A receipt
                # digest moves only with a PTX identity receipt for the kernels
                # this oracle runs, or a fresh capture on the receipt's card (or
                # architecture, for sm_120) reproducing every recorded word
                # (tools/advect_wrf471_oracle/README.md).
                digest = receipt["kernels"][module]
                assert hashlib.sha256(module_source(module).encode("utf-8")).hexdigest() == digest, (
                    f"{name} pins {module} source {digest[:8]}, not the source the loader compiles: "
                    f"the kernel moved and {receipt['gpu']} has not reproduced its words at it. "
                    "On that card: python tools/advect_wrf471_oracle/recapture_receipt.py "
                    "--fixture tests/data/wrf471_advect --scratch <empty dir> --install")
            assert receipt["fixture_manifest_sha256"] == hashlib.sha256((ADVECT_ORACLE_DIR / "cases.json").read_bytes()).hexdigest(), name
        words_directory = receipt.get("words_directory", "gpu-words")
        for case in cases:
            assert set(ROUTINES).issubset(case.reference)
            row = receipt["cases"][case.name]
            path = ADVECT_ORACLE_DIR / words_directory / row["words_file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["words_sha256"], (name, case.name)
            with np.load(path, allow_pickle=False) as data:
                outputs = {key: data[key] for key in data.files}
            got = measure_advect_parity(case, outputs)
            assert got == {key: row["measurements"][key] for key in SUPPORTED_OUTPUTS}, (name, case.name)


def test_advect_compiled_native_outputs_hold_the_pinned_source_receipts():
    directory = Path(__file__).resolve().parents[1] / "tools" / "advect_wrf471_oracle"
    pin = json.loads((directory / "PIN.json").read_text(encoding="ascii"))
    assert pin["wrf_version"] == "4.7.1"
    assert pin["wrf_commit"] == "f52c197ed39d12e087d02c50f412d90d418f6186"
    sources = (directory / "SOURCES.sha256").read_text(encoding="ascii")
    assert "58253bdbeb188dd47ed0579fcd2891086be1889b75c0c7d3696c9ad1d213559d  dyn_em/module_advect_em.F" in sources
    rows = (ADVECT_ORACLE_DIR / "oracle-sha256sums.txt").read_text(encoding="ascii").splitlines()
    recorded = {line.split(maxsplit=1)[1].rsplit("/", 1)[-1]: line.split(maxsplit=1)[0] for line in rows}
    for case in load_advect_cases():
        name = f"{case.name}-wrf.npz"
        assert hashlib.sha256((ADVECT_ORACLE_DIR / name).read_bytes()).hexdigest() == recorded[name]
    for name in ("module_advect_em.F", "module_model_constants.F", "module_wrf_error.F"):
        assert name in recorded


@pytest.mark.parametrize("module_name,constant,load", (
    ("gpuwm.verify.advect_oracle", "ADVECT_ORACLE_DIR", lambda m, d: m.load_advect_cases(d)),
    ("gpuwm.verify.smallstep_oracle", "ORACLE_DIR", lambda m, d: next(m.load_cases())),
    ("gpuwm.verify.smallstep_bookkeeping_oracle", "BOOKKEEPING_DIR",
     lambda m, d: m.load_bookkeeping(d / "real_initial-0-prep-1.npz")),
    ("gpuwm.verify.bigstep_coupling_oracle", "ORACLE_DIR", lambda m, d: m.load_coupling_oracle(d)),
    ("gpuwm.verify.bigstep_momentum_oracle", "MOMENTUM_ORACLE_DIR",
     lambda m, d: m.load_momentum_fixture("real", d)),
    ("gpuwm.verify.bigstep_prep_oracle", "ORACLE_DIR", lambda m, d: m.load_prep_fixture("real-periodic2")),
    ("gpuwm.verify.bigstep_rk_oracle", "RK_ORACLE_DIR", lambda m, d: m.load_rk_oracle()),
), ids=("advect", "smallstep", "bookkeeping", "coupling", "momentum", "prep", "rk"))
def test_oracle_harness_refuses_by_name_without_source_checkout_fixtures(
        module_name, constant, load, tmp_path, monkeypatch):
    """The wheel carries these harnesses but not tests/data.

    Without the refusal an installed harness dies on a bare missing-archive
    error beside site-packages, which reads as a damaged install rather than
    as an oracle that only runs from a source checkout.
    """
    import importlib
    module = importlib.import_module(module_name)
    missing = tmp_path / "tests" / "data" / "absent"
    monkeypatch.setattr(module, constant, missing)
    with pytest.raises(FileNotFoundError, match="runs from a source checkout") as refused:
        load(module, missing)
    assert str(missing) in str(refused.value)


def test_advect_mono_reference_distinguishes_the_unimplemented_option():
    case = next(case for case in load_advect_cases() if case.name == "limiter_moisture_front")
    row = measure_words(case.reference["advect_scalar_pd"], case.reference["advect_scalar_mono"])
    assert row["different_words"] > 0
    docs = Path(__file__).resolve().parents[1] / "docs" / "public" / "CONFIGURATION.md"
    text = docs.read_text(encoding="utf-8")
    assert "moist_adv_opt" in text and "positive" in text.lower()


@pytest.fixture(scope="module")
def device_receipt():
    """This card's receipt, resolved before ``cuda_cases`` launches anything.

    A test that compares receipt words asks for this fixture ahead of
    ``cuda_cases``, so on an uncertified card whose receipt is missing or
    stale it skips before every case's kernels run for nothing.
    """
    return _device_receipt()


@pytest.fixture(scope="module")
def cuda_cases():
    gpu, capability = advect_gpu_identity()
    if architecture_is_certified(capability):
        # A certified card with no receipt is a coverage gap: fail before the launches.
        _receipt_for_gpu(gpu, capability)
    return {case.name: (case, advect_port_outputs(case)) for case in load_advect_cases()}


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("routine", SUPPORTED_ROUTINES)
def test_advect_cuda_routine_holds_every_measured_wrf_word(routine, device_receipt, cuda_cases):
    receipt = device_receipt
    for name, (case, outputs) in cuda_cases.items():
        row = measure_words(outputs[routine], case.reference[routine])
        assert row == receipt["cases"][name]["measurements"][routine], (name, routine, row)


@pytest.mark.gpu
@requires_gpu
def test_advect_cuda_pd_optional_channels_hold_every_defined_word(device_receipt, cuda_cases):
    receipt = device_receipt
    for name, (case, outputs) in cuda_cases.items():
        for key in ("advect_scalar_pd.h_tendency", "advect_scalar_pd.z_tendency"):
            row = measure_words(outputs[key], case.reference[key], defined=defined_output_mask(case, key))
            assert row == receipt["cases"][name]["measurements"][key], (name, key, row)


@pytest.mark.gpu
@requires_gpu
def test_advect_cuda_gate_rejects_a_flux5_transcription_mutation(cuda_cases):
    name = "real_interior"
    case, original = cuda_cases[name]
    with arithmetic_control("mutation"):
        changed = advect_port_outputs(case, variant="mutation")
    baseline = measure_words(original["advect_scalar"], case.reference["advect_scalar"])
    mutated = measure_words(changed["advect_scalar"], case.reference["advect_scalar"])
    with pytest.raises(AssertionError):
        assert mutated == baseline


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("control_name", ("w_lid_horizontal", "w_lid_vertical"))
def test_advect_cuda_w_lid_has_each_compiled_wrf_term_default_on(control_name):
    import cupy as cp
    from types import SimpleNamespace
    from gpuwm.core.advection import launch_flux_div_w
    from gpuwm.core.fp32_ulp import assert_bit_exact
    base = next(case for case in load_advect_cases() if case.name == "real_interior")
    case = next(case for case in w_lid_control_cases(base) if case.name == control_name)
    a = {key: cp.asarray(case.inputs[key]) for key in ("w", "ru", "rv", "rw", "rdn", "fnm", "fnp", "msftx")}
    coord = SimpleNamespace(rdn=a["rdn"], fnm=a["fnm"], fnp=a["fnp"])
    tendency = cp.zeros_like(a["w"])
    launch_flux_div_w(a["w"], a["ru"], a["rv"], a["rw"], tendency, coord,
                      case.metadata["dx"], case.metadata["dy"], msf=a["msftx"], has_msf=True)
    with np.load(ADVECT_ORACLE_DIR / "w-lid-controls-wrf.npz", allow_pickle=False) as native:
        expected = native[control_name][-1, 4:28, 4:28]
    assert np.count_nonzero(expected) == 24 * 24
    assert_bit_exact(cp.asnumpy(tendency[-1]), expected, control_name)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("control_name", ("mapped_open_u", "mapped_open_v"))
def test_advect_cuda_mapped_radiation_matches_compiled_wrf_default_on(control_name):
    import cupy as cp
    from types import SimpleNamespace
    from gpuwm.core.dycore import apply_open_radiative_bc
    from gpuwm.core.fp32_ulp import assert_bit_exact
    base = next(case for case in load_advect_cases() if case.name == "real_interior")
    case = next(case for case in mapped_radiation_control_cases(base) if case.name == control_name)
    a = {key: cp.asarray(case.inputs[key]) for key in
         ("u", "v", "mu_perturbation", "mub", "c1h", "c2h", "msfux", "msfvx")}
    state = SimpleNamespace(
        u=a["u"], v=a["v"], mup=a["mu_perturbation"], mub2d=a["mub"],
        c1h=a["c1h"], c2h=a["c2h"], msfu=a["msfux"], msfv=a["msfvx"], has_msf=True,
        ru_t=cp.zeros_like(a["u"]), rv_t=cp.zeros_like(a["v"]))
    cfg = SimpleNamespace(nz=49, ny=24, nx=24, dx=1.0, dy=1.0, open_x=True, open_y=True)
    apply_open_radiative_bc(state, cfg)
    with np.load(ADVECT_ORACLE_DIR / "mapped-radiation-controls-wrf.npz", allow_pickle=False) as native:
        expected = native[control_name]
    if control_name == "mapped_open_u":
        want = expected[:49, 4:28][:, :, [4, 28]]
        got = cp.asnumpy(state.ru_t)[:, :, [0, 24]]
    else:
        want = expected[:49, [4, 28], 4:28]
        got = cp.asnumpy(state.rv_t)[:, [0, 24], :]
    assert np.count_nonzero(want) == 49 * 24 * 2
    assert_bit_exact(got, want, control_name)
