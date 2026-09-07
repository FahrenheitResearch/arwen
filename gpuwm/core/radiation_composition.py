"""Independent radiation-spectrum construction and composition.

Existing paired adapters retain their construction and arithmetic. Mixed
selections execute only each selected spectrum and merge its own carriers.
"""
from __future__ import annotations


def radiation_adapters(adapter):
    """Yield distinct leaf adapters in execution order."""
    seen = set()
    def visit(current):
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        components = getattr(current, "spectrum_adapters", None)
        if components is not None:
            for component in components:
                yield from visit(component)
        else:
            yield current
    return tuple(visit(adapter))


def modern_radiation_adapters(adapter):
    return tuple(item for item in radiation_adapters(adapter)
                 if type(item).__module__ == "gpuwm.core.rrtmgp"
                 and type(item).__name__ == "RRTMGPRadiation")


def attach_modern_workspace(adapter, workspace):
    """Attach a shared CUDA workspace only to the engines that consume it."""
    if workspace is not None:
        for engine in modern_radiation_adapters(adapter):
            engine.column_chunk = workspace.column_chunk
            engine.chunk_workspace = workspace


def legacy_radiation_adapter(adapter):
    return next((item for item in radiation_adapters(adapter)
                 if type(item).__module__ == "gpuwm.core.rrtmg_legacy"
                 and type(item).__name__ == "RRTMGLegacyRadiation"), None)


class ComposedRadiation:
    """Keep each spectrum's own heating, flux and restart authority."""
    @property
    def spectrum_adapters(self):
        return (self.longwave_adapter, self.shortwave_adapter)

    def __init__(self, start_time, latitude_deg, longitude_deg, *,
                 longwave_adapter=None, shortwave_adapter=None):
        if longwave_adapter is None and shortwave_adapter is None:
            raise ValueError("radiation composition needs an active spectrum")
        self.start_time = start_time
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg
        self.longwave_adapter = longwave_adapter
        self.shortwave_adapter = shortwave_adapter
        self.publishes_olr = bool(getattr(longwave_adapter, "publishes_olr", False))
        self.glw_provenance = "scheme" if longwave_adapter is not None else "declared"

    @property
    def update_count(self):
        return max(getattr(item, "update_count", 0)
                   for item in radiation_adapters(self))

    @property
    def _o33d_grid(self):
        adapter = legacy_radiation_adapter(self)
        return None if adapter is None else adapter._o33d_grid

    @_o33d_grid.setter
    def _o33d_grid(self, value):
        adapter = legacy_radiation_adapter(self)
        if adapter is None:
            raise ValueError("this radiation composition has no retained ozone provider")
        adapter._o33d_grid = value

    @property
    def column_chunk(self):
        adapters = modern_radiation_adapters(self)
        return adapters[0].column_chunk if adapters else None

    @column_chunk.setter
    def column_chunk(self, value):
        for adapter in modern_radiation_adapters(self):
            adapter.column_chunk = value

    @property
    def chunk_workspace(self):
        adapters = modern_radiation_adapters(self)
        return getattr(adapters[0], "chunk_workspace", None) if adapters else None

    @chunk_workspace.setter
    def chunk_workspace(self, value):
        for adapter in modern_radiation_adapters(self):
            adapter.chunk_workspace = value

    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp
        from datetime import timedelta
        from gpuwm.core.physics import (RadiationResult, _model_clock_dt,
                                        _physics_interval_seconds)
        arguments = dict(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
        lw = (self.longwave_adapter(**arguments)
              if self.longwave_adapter is not None else None)
        sw = (self.shortwave_adapter(**arguments)
              if self.shortwave_adapter is not None else None)
        zero = None
        if lw is None or sw is None:
            zero = cp.zeros_like(atmosphere["pressure"])
        if sw is None:
            from gpuwm.core.dudhia import wrf_solar_geometry
            valid_time = self.start_time + timedelta(seconds=float(state.elapsed_seconds))
            minutes = cfg.radt if cfg.radt > 0.0 else cfg.radt_minutes
            interval = _physics_interval_seconds(minutes, _model_clock_dt(cfg))
            coszen, _ = wrf_solar_geometry(valid_time, self.latitude_deg,
                self.longitude_deg, hour_offset_seconds=0.5 * interval)
            swdown = cp.zeros_like(fields["glw"])
            gsw = swdown
        else:
            swdown, gsw, coszen = sw.swdown, sw.gsw, sw.coszen
        return RadiationResult(
            rthratenlw=lw.rthratenlw if lw is not None else zero,
            rthratensw=sw.rthratensw if sw is not None else zero,
            swdown=swdown, glw=lw.glw if lw is not None else fields["glw"],
            gsw=gsw, coszen=coszen, olr=lw.olr if lw is not None else None)

    def restart_identity(self):
        from gpuwm.io.restart import _array_setup_identity
        return {
            "start_time": self.start_time.isoformat(),
            "latitude": _array_setup_identity(self.latitude_deg),
            "longitude": _array_setup_identity(self.longitude_deg),
            "algorithm": "independent-radiation-spectra-v1",
            "above_atmosphere_policy": "each selected spectrum retains its own column policy",
            "longwave": adapter_restart_identity(self.longwave_adapter),
            "shortwave": adapter_restart_identity(self.shortwave_adapter),
        }


def adapter_restart_identity(adapter):
    """Bind selected engines' setup without treating nested state as disposable."""
    if adapter is None:
        return None
    from gpuwm.io.restart import (_array_setup_identity, _resolved_object_setup_identity,
                                  _rrtmgp_workspace_identity)
    identity = {
        "class": type(adapter).__module__ + "." + type(adapter).__qualname__,
        "start_time": adapter.start_time.isoformat(),
        "latitude": _array_setup_identity(adapter.latitude_deg),
        "longitude": _array_setup_identity(adapter.longitude_deg),
    }
    declared = getattr(adapter, "restart_identity", None)
    if declared is not None:
        identity["implementation"] = declared() if callable(declared) else declared
    elif modern_radiation_adapters(adapter):
        identity.update({
            "column_chunk": adapter.column_chunk,
            "validation_mode": adapter.validation_mode,
            "trace_gas_overrides": adapter.trace_gas_overrides,
            "trace_vmr": adapter.trace_vmr,
            "ozone_log_pressure": _array_setup_identity(adapter._ozone_logp),
            "ozone_vmr": _array_setup_identity(adapter._ozone_vmr),
            "tables": {name: _resolved_object_setup_identity(getattr(adapter, name), name)
                       for name in ("lw_tables", "sw_tables", "lw_cloud_tables", "sw_cloud_tables")},
            "chunk_workspace": _rrtmgp_workspace_identity(getattr(adapter, "chunk_workspace", None)),
        })
    elif (type(adapter).__module__ == "gpuwm.core.analytic_radiation"
          and type(adapter).__name__ == "AnalyticClearSkyRadiation"):
        from gpuwm.core.analytic_radiation import (
            CLEAR_SKY_TRANSMISSIVITY, SOLAR_CONSTANT_WM2, STEFAN_BOLTZMANN)
        identity["proxy_constants"] = [CLEAR_SKY_TRANSMISSIVITY,
                                       SOLAR_CONSTANT_WM2, STEFAN_BOLTZMANN]
    else:
        raise ValueError("a custom radiation spectrum must declare restart_identity")
    if hasattr(adapter, "p_top"):
        identity["p_top"] = None if adapter.p_top is None else float(adapter.p_top)
    if isinstance(getattr(adapter, "longwave", None), bool):
        identity["spectra"] = {"lw": bool(adapter.longwave), "sw": bool(adapter.shortwave)}
    return identity


def trace_gas_override_consumption(cfg, overrides):
    """Selected absorption spectra per declared gas; an empty tuple is inactive.

    Legacy SW accepts an N2O profile for WRF interface compatibility but its
    optical-depth operator does not consume it. Keep that distinction visible
    without changing the requested value or inventing a coefficient consumer.
    """
    from gpuwm.config import radiation_scheme_ids
    from gpuwm.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant
    from gpuwm.core.trace_gases import (
        CLASSIC_GASES, LEGACY_LW_GASES, LEGACY_SW_GASES, LEGACY_SW_ABSORPTION_GASES,
        validate_trace_gas_overrides)
    declared = validate_trace_gas_overrides(overrides)
    if not declared:
        return {}
    lw, sw = radiation_scheme_ids(cfg)
    supported = set(CLASSIC_GASES) if lw == 1 else set()
    consumers = {"lw": set(supported), "sw": set()}
    if 4 in (lw, sw):
        if rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY:
            if lw == 4:
                supported.update(LEGACY_LW_GASES)
                consumers["lw"].update(LEGACY_LW_GASES)
            if sw == 4:
                supported.update(LEGACY_SW_GASES)
                consumers["sw"].update(LEGACY_SW_ABSORPTION_GASES)
        else:
            from gpuwm.core.rrtmgp import coefficient_gas_names
            if lw == 4:
                supported.update(coefficient_gas_names("lw"))
                consumers["lw"].update(coefficient_gas_names("lw"))
            if sw == 4:
                supported.update(coefficient_gas_names("sw"))
                consumers["sw"].update(coefficient_gas_names("sw"))
    if not supported:
        return {gas: () for gas in declared}
    validate_trace_gas_overrides(declared, supported=supported,
                                 consumer="selected radiation spectra")
    return {gas: tuple(spectrum for spectrum, gases in consumers.items() if gas in gases)
            for gas in declared}


def trace_gas_override_status(cfg, overrides):
    """Summarize actual consumption without labelling inactive operands applied."""
    consumption = trace_gas_override_consumption(cfg, overrides)
    if not consumption:
        return "not_declared"
    applied = sum(bool(spectra) for spectra in consumption.values())
    return ("applied" if applied == len(consumption) else
            "partially_applied" if applied else "inactive")


def make_radiation(cfg, start_time, latitude_deg, longitude_deg, *,
                   p_top=None, column_chunk=None, trace_gas_overrides=None,
                   ozone_parent=None):
    """Construct the requested spectra without substituting another selector."""
    from gpuwm.config import radiation_scheme_ids
    from gpuwm.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant
    lw, sw = radiation_scheme_ids(cfg)
    trace_gas_override_status(cfg, trace_gas_overrides)
    from gpuwm.core.trace_gases import (
        CLASSIC_GASES, LEGACY_LW_GASES, LEGACY_SW_GASES, trace_gas_subset)
    if not (lw or sw):
        return None
    args = (start_time, latitude_deg, longitude_deg)
    def engine(selector, longwave, shortwave):
        if selector == 0:
            return None
        if selector == 1 and longwave:
            from gpuwm.core.rrtm_lw import RRTMLongwaveRadiation
            return RRTMLongwaveRadiation(*args, p_top=p_top, icloud=cfg.icloud,
                                        column_chunk=column_chunk,
                                        trace_gas_overrides=trace_gas_subset(
                                            trace_gas_overrides, CLASSIC_GASES) or None)
        if selector == 1:
            from gpuwm.core.dudhia import DudhiaShortwaveRadiation
            return DudhiaShortwaveRadiation(*args, swrad_scat=cfg.swrad_scat, icloud=cfg.icloud)
        if selector == 90:
            from gpuwm.core.analytic_radiation import AnalyticClearSkyRadiation
            return AnalyticClearSkyRadiation(*args)
        if selector != 4:
            raise ValueError(f"no radiation engine for selector {selector}")
        if rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY:
            from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation
            return RRTMGLegacyRadiation(*args, p_top=p_top, o3input=cfg.o3input,
                ozone_parent=ozone_parent, longwave=longwave, shortwave=shortwave,
                trace_gas_overrides=trace_gas_subset(trace_gas_overrides,
                    (LEGACY_LW_GASES if longwave else frozenset())
                    | (LEGACY_SW_GASES if shortwave else frozenset())) or None)
        from gpuwm.core.rrtmgp import RRTMGPRadiation, coefficient_gas_names
        selected_gases = trace_gas_overrides
        if trace_gas_overrides:
            supported = set()
            if longwave:
                supported.update(coefficient_gas_names("lw"))
            if shortwave:
                supported.update(coefficient_gas_names("sw"))
            selected_gases = trace_gas_subset(trace_gas_overrides, supported) or None
        options = dict(trace_gas_overrides=selected_gases,
                       longwave=longwave, shortwave=shortwave)
        if column_chunk is not None:
            options["column_chunk"] = column_chunk
        return RRTMGPRadiation(*args, **options)
    if (lw, sw) == (1, 1):
        from gpuwm.core.rrtm_lw import RRTMDudhiaRadiation
        return RRTMDudhiaRadiation(*args, p_top=p_top,
                                   icloud=cfg.icloud, swrad_scat=cfg.swrad_scat,
                                   column_chunk=column_chunk,
                                   trace_gas_overrides=trace_gas_overrides)
    if lw == sw:
        return engine(lw, True, True)
    if (lw, sw) == (0, 1):
        return engine(sw, False, True)
    # One device geography pair is shared by both CUDA leaves. Legacy
    # leaves retain their existing host geography conversion.
    import cupy as cp
    args = (start_time,
            cp.ascontiguousarray(cp.asarray(latitude_deg, dtype=cp.float32)),
            cp.ascontiguousarray(cp.asarray(longitude_deg, dtype=cp.float32)))
    return ComposedRadiation(*args, longwave_adapter=engine(lw, True, False),
                             shortwave_adapter=engine(sw, False, True))
