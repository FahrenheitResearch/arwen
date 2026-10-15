"""YSU source generations shared by configuration, registry and restart."""

from gpuwm.checkpoint_identity import PBL_ALGORITHM_IDENTITIES as _PBL_IDENTITIES

YSU_DEFAULT_GENERATION = "wrf_480"
YSU_GENERATIONS = {
    # Reproduction: the WRF 4.6.1 kernel (kernels/ysu.cu) in both arithmetic
    # modes, under the restart identity the checkpoint table publishes for
    # bl_pbl_physics = 1.  At 2.8.9 that is "ysu-wrf-v4.6.1-v2-...": a
    # wrf_461 run is byte-identical to 2.8.9 and continues its checkpoints.
    # The identity is READ from the table, so a later change to ysu.cu that
    # bumps the table moves this row with it instead of disagreeing with it.
    "wrf_461": {
        "kernel_module": "ysu",
        "restart_algorithm_identity": _PBL_IDENTITIES[1],
        "source_commit": "d66e442fccc04111067e29274c9f9eaccc3cef28",
        "source_sha256": "a695daa712591b30fa72b8f24eac003e8e7016531283d29443bcc04ce3ea77d8",
        "topdown_efficiency": "0.2 * 8. * tmp1 + 0.2",
    },
    "wrf_480": {
        "kernel_module": "ysu_480",
        "restart_algorithm_identity": "ysu-wrf-v4.8.0-topdown-cap-noftz-v1",
        "source_commit": "550b5b494758ad2756394e93cb08b5709b8feb83",
        "source_sha256": "15d76892fc6ffa5c9f8f7ebecfeb6eaa5c23d71b44e8c53a8832f49ff711ea7e",
        "topdown_efficiency": "min(0.2 * 8. * tmp1, 0.4)",
        "declared_corrections": [
            "Bound the get_pblh searches and set the upstream unwritten final TKE word to zero in shallow domains.",
            "Remove the whole YSU surface drag under BEP because its source terms already include rural drag.",
        ],
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
    if strict and "strict_kernel_module" in row:
        return row["strict_kernel_module"], tuple(tuple(d) for d in row["strict_defines"])
    return row["kernel_module"], ()


def ysu_algorithm_identity(version: str, *, strict: bool = False):
    row = ysu_generation(version)
    if version == "wrf_461":
        # The reproduction arm IS 2.8.9's YSU: its identity is the one the
        # checkpoint table publishes for bl_pbl_physics = 1 (read live, so
        # the table stays the single owner of that string).
        from gpuwm.checkpoint_identity import PBL_ALGORITHM_IDENTITIES
        return PBL_ALGORITHM_IDENTITIES[1]
    return (row["strict_restart_algorithm_identity"] if strict and
            "strict_restart_algorithm_identity" in row else row["restart_algorithm_identity"])
