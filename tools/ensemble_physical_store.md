# Native physical field storage

`NativePhysicalStore` captures the physical fields on the target horizontal grid before the ordinary native real initializer reconstructs prognostic state and lateral boundaries. New stores use `gpuwm-ensemble-physical-store.v2`.

A writer must supply `grid_identity`, `source_identity` and `field_contract`. The field contract declares each array's units, dimensions, scalar or target-grid vector basis, source fields, and mapping operation. Source mapping files and implementation evidence are hashed. These declarations come from the source adapter's verified contract. An array name or plausible range does not establish its units.

The native NetCDF writer records the field contract digest and vertical coordinate in global attributes. Each variable also records its units, physical dimensions, vector basis and source operation. The reader verifies these attributes, shape, dtype, file hash, valid time and grid before returning arrays. The manifest digest is pinned when the reader opens it; changing the manifest after opening is refused.

The source adapters preserve their distinct physical representations:

- HRRR uses dimensionless hybrid model indices in the historically named `levels_hpa` member. Its actual pressure is the three-dimensional `PRES` field in Pa. Its specific humidity is retained without inventing a separately mapped RH field.
- Native GFS uses pressure levels in hPa and RH/RH2 in percent. Its native geopotential-height values are retained, with source `gpm` and the initializer's `m` interpretation recorded explicitly. No extra division is applied.
- Mapped compositions take units from the validated canonical mapping. For example, canonical geopotential height is already in metres and reaches `GHT` without a second conversion. The full pressure field remains the authority for column alignment. Hybrid mappings can expose a representative pressure ladder, explicitly distinguished from independent pressure levels.

The physical field stores retain soil, surface and provider metadata. Winds use the actual target C-grid faces: `UU` and `U10` have an x-face dimension, and `VV` and `V10` have a y-face dimension. Each native preparation consumer checks the incoming unit, dimension and basis declarations against its own supported physical representation before initialization.

## Existing stores

Version 1 manifests contain array dtype and shape, but lack units and coordinate authority. They are refused for ordinary use until explicitly qualified or recaptured. `allow_unqualified_legacy=True` exists for the qualification procedure and does not establish a usable field contract.

`qualify_legacy_store` requires the expected original manifest SHA-256, the exact captured source identity, an explicit field contract and all files cited by that contract's evidence hashes. The source adapter must first revalidate the original source mapping and receipts. The procedure verifies the source evidence, every native frame and all coordinate dimensions, then exclusively publishes `physical-qualification.json`. It preserves the original manifest and native array bytes.

The certificate binds the original manifest, individual frame hashes, explicit field contract and verified evidence. It records `native_attributes_present: false`; qualification does not pretend that legacy files contained native unit attributes. Prepared input bindings carry the complete certificate. Where the original implementation evidence is incomplete, a new pinned native capture plus exact array-word comparison can supply the qualification evidence.

Storage integrity, unit consistency and word-identical recapture are implementation checks. They do not establish calibrated ensemble spread or forecast skill.
