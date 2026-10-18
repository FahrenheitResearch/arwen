"""YSU source generations shared by configuration, registry and restart."""

from gpuwm.checkpoint_identity import PBL_ALGORITHM_IDENTITIES as _PBL_IDENTITIES

YSU_DEFAULT_GENERATION = "wrf_480"
YSU_GENERATIONS = {
    # Default arithmetic: the reproduction of 2.8.9 (kernels/ysu.cu), under
    # the restart identity the checkpoint table publishes for
    # bl_pbl_physics = 1.  At 2.8.9 that is "ysu-wrf-v4.6.1-v2-...": a
    # default wrf_461 run is byte-identical to 2.8.9 and continues its
    # checkpoints.  The identity is READ from the table, so a later change
    # to ysu.cu that bumps the table moves this row with it.  Under
    # GPUWM_WRF_EXACT=1 the row runs YSU_STRICT_ARMS["wrf_461"] instead.
    "wrf_461": {
        "kernel_module": "ysu",
        "restart_algorithm_identity": _PBL_IDENTITIES[1],
        "source_commit": "d66e442fccc04111067e29274c9f9eaccc3cef28",
        "source_sha256": "a695daa712591b30fa72b8f24eac003e8e7016531283d29443bcc04ce3ea77d8",
        "topdown_efficiency": "0.2 * 8. * tmp1 + 0.2",
    },
    "wrf_480": {
        "kernel_module": "ysu_480",
        "restart_algorithm_identity": "ysu-wrf-v4.8.0-entrainment-courant-v2",
        "strict_restart_algorithm_identity": "ysu-wrf-v4.8.0-topdown-cap-noftz-v1",
        "source_commit": "550b5b494758ad2756394e93cb08b5709b8feb83",
        "source_sha256": "15d76892fc6ffa5c9f8f7ebecfeb6eaa5c23d71b44e8c53a8832f49ff711ea7e",
        "topdown_efficiency": "min(0.2 * 8. * tmp1, 0.4)",
        "declared_corrections": [
            "Bound explicit nonlocal heat/vapor entrainment by its discrete donor-jump Courant number (default arithmetic; strict reproduction retains upstream).",
            "Bound the get_pblh searches and set the upstream unwritten final TKE word to zero in shallow domains.",
            "Remove the whole YSU surface drag under BEP because its source terms already include rural drag.",
        ],
    },
}


#: GPUWM_WRF_EXACT=1 arms, by generation.  wrf_461's matches WRF 4.6.1
#: bl_ysu_run word for word on the generations corpus, with the same two
#: declared exceptions as wrf_480 (YSU surface drag removed on mixed BEP
#: urban columns, and the bounded PBL-height search on four-level columns,
#: where unmodified WRF reads past the column top): the source-ordered unit
#: (ysu_480.cu with the 4.6.1 top-down term), under its own restart identity.  ysu.cu is not WRF
#: 4.6.1 under strict arithmetic: on the generations corpus it wrote 4,916 to
#: 9,898 different words per step on up to 164 of the 1,188 columns, mostly
#: cloud-topped stratocumulus (tests/test_ysu_generation_batch_gpu.py grades
#: this arm).
#: Kept out of YSU_GENERATIONS, which the registry embeds as the YSU option's
#: physics: the arithmetic mode is a process setting, not a selection, so a
#: preparation made under default arithmetic stays valid.
YSU_STRICT_ARMS = {
    "wrf_461": {
        "kernel_module": "ysu_480",
        "defines": (("YSU_LEGACY_TOPDOWN", 1),),
        "restart_algorithm_identity": "ysu-wrf-v4.6.1-source-order-noftz-v1",
    },
}


def ysu_generation(version: str) -> dict:
    try:
        return YSU_GENERATIONS[version]
    except (KeyError, TypeError):
        raise ValueError(
            f"ysu_version must be one of {tuple(YSU_GENERATIONS)}, got {version!r}") from None


def ysu_kernel_selection(version: str, *, strict: bool = False):
    row = ysu_generation(version)
    if strict and version in YSU_STRICT_ARMS:
        arm = YSU_STRICT_ARMS[version]
        return arm["kernel_module"], arm["defines"]
    return row["kernel_module"], ()


def ysu_algorithm_identity(version: str, *, strict: bool = False):
    row = ysu_generation(version)
    if strict and version in YSU_STRICT_ARMS:
        return YSU_STRICT_ARMS[version]["restart_algorithm_identity"]
    if strict and "strict_restart_algorithm_identity" in row:
        # A generation whose strict reproduction retains the upstream
        # arithmetic the corrected default departs from (wrf_480's
        # entrainment bound, 1edfbfe953b) restarts under its own identity.
        return row["strict_restart_algorithm_identity"]
    if version == "wrf_461":
        # The default reproduction arm IS 2.8.9's YSU: its identity is the
        # one the checkpoint table publishes for bl_pbl_physics = 1 (read
        # live, so the table stays the single owner of that string).
        from gpuwm.checkpoint_identity import PBL_ALGORITHM_IDENTITIES
        return PBL_ALGORITHM_IDENTITIES[1]
    return row["restart_algorithm_identity"]
