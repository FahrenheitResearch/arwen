from __future__ import annotations

def _frame_header(
    mapping: Mapping[str, object],
    *,
    valid_time: datetime,
    source_cycle: datetime,
    latitude: np.ndarray,
    longitude: np.ndarray,
    vertical_values: np.ndarray,
    fields: Mapping[str, CanonicalField],
    source_id: str,
    hybrid_a: np.ndarray | None = None,
    hybrid_b: np.ndarray | None = None,
) -> SourceFrameHeader:
    vertical = mapping["coordinates"]["vertical"]
    vertical_kind = str(vertical["kind"])
    source_vertical_name = "atmosphere"
    descriptors: dict[str, VerticalDescriptor] = {
        source_vertical_name: VerticalDescriptor(
            coordinate={
                "hybrid_sigma_pressure": "hybrid",
                "embedded_levels": "model_level",
            }.get(vertical_kind, vertical_kind),
            level_count=int(vertical_values.size),
            level_values=tuple(float(value) for value in vertical_values),
            a_coefficients=(
                () if hybrid_a is None
                else tuple(float(value) for value in hybrid_a)
            ),
            b_coefficients=(
                () if hybrid_b is None
                else tuple(float(value) for value in hybrid_b)
            ),
            positive=str(vertical.get("positive", "down")),
            units=str(vertical["units"]),
        )
    }
    soil_fields = [field for field in fields.values() if "soil" in field.axes]
    if soil_fields:
        soil_count = soil_fields[0].values.shape[soil_fields[0].axes.index("soil")]
        descriptors["soil"] = VerticalDescriptor(
            coordinate="soil_depth", level_count=soil_count,
            units="index", positive="down",
        )
    lead = int((valid_time - source_cycle).total_seconds())
    time = TimeDescriptor(
        reference_time=source_cycle.replace(tzinfo=timezone.utc).isoformat(),
        valid_time=valid_time.replace(tzinfo=timezone.utc).isoformat(),
        lead_seconds=lead,
    )
    field_descriptors = []
    for field in fields.values():
        field_descriptors.append(FieldDescriptor(
            canonical_name=field.name,
            units=field.units,
            dimensions=field.axes,
            grid_location=field.location,
            vertical_coordinate=(
                source_vertical_name if "vertical" in field.axes
                else "soil" if "soil" in field.axes else None
            ),
            time=time,
            data_reference="sha256:" + _array_sha256(field.values),
            dtype=field.values.dtype.str,
            shape=field.values.shape,
            missing_value_policy=(
                "explicit_missing" if field.missing_count else "reject_nonfinite"
            ),
            source_field=";".join(field.source_references),
        ))
    declaration = _mapping_grid_declaration(mapping)
    if declaration["family"] == GRID_FAMILY_LAMBERT:
        parameters = declaration["parameters"]
        grid_descriptor = GridDescriptor(
            projection=GRID_FAMILY_LAMBERT,
            nx=int(longitude.size), ny=int(latitude.size),
            earth_shape=f"grib_shape_of_earth:{parameters['shape_of_earth']}",
            scan_order="+x,+y",
            # Grid-relative sources were rotated to the earth basis at
            # decode time (_rotate_grid_relative_winds); the frame states
            # what its arrays ARE, not what the producer published.
            wind_basis="earth_relative",
            parameters={
                **{key: parameters[key] for key in sorted(_LAMBERT_PARAMETER_KEYS)},
                "axis_unit_m": PROJECTED_AXIS_UNIT_M,
                "source_wind_basis": declaration["wind_basis"],
            },
        )
    else:
        grid_descriptor = GridDescriptor(
            projection="regular_latitude_longitude",
            nx=int(longitude.size), ny=int(latitude.size),
            earth_shape="source_metadata_bound",
            scan_order=(
                ("+x" if longitude[-1] > longitude[0] else "-x") + ","
                + ("+y" if latitude[-1] > latitude[0] else "-y")
            ),
            wind_basis="earth_relative",
            parameters={
                "latitude_first": float(latitude[0]),
                "latitude_last": float(latitude[-1]),
                "longitude_first": float(longitude[0]),
                "longitude_last": float(longitude[-1]),
            },
        )
    return SourceFrameHeader(
        source_id=source_id,
        source_cycle=source_cycle.replace(tzinfo=timezone.utc).isoformat(),
        grid=grid_descriptor,
        vertical_coordinates=descriptors,
        fields=tuple(field_descriptors),
        initialization_policies=dict(
            mapping["target"].get("initialization_policies", {})
        ),
    )
