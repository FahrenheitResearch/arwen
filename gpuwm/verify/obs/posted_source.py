"""Table-declared posted GRIB2 forecasts through the native scoring reader.

This is a model-independent implementation of the existing observation seam.
Rust owns decoding, unit conversion, wind magnitude and geographic projection.
The wrapper only owns the handle and exposes its arrays to the scorer.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import json
from pathlib import Path

import numpy as np

from gpuwm.verify.obs.contracts import ModelGrid
from gpuwm.verify.obs.stations import StationPosition


class PostedGrib2Source:
    def __init__(self, run_directory, *, field_table, reader_library):
        self._directory = Path(run_directory)
        paths = sorted(self._directory.glob("*.grib2"))
        if not paths:
            raise ValueError("the posted forecast directory holds no GRIB2 files")
        library = Path(reader_library).resolve()
        self._library_sha256 = hashlib.sha256(library.read_bytes()).hexdigest()
        self._lib = ct.CDLL(str(library))
        self._lib.posted_reader_abi_version.restype = ct.c_uint32
        if self._lib.posted_reader_abi_version() != 1:
            raise ValueError("the posted scoring reader library has an incompatible ABI")
        self._lib.posted_reader_open.argtypes = [ct.c_char_p, ct.c_char_p]
        self._lib.posted_reader_open.restype = ct.c_void_p
        self._lib.posted_reader_close.argtypes = [ct.c_void_p]
        self._lib.posted_reader_close.restype = None
        self._lib.posted_reader_error.restype = ct.c_char_p
        self._lib.posted_reader_metadata.argtypes = [ct.c_void_p]
        self._lib.posted_reader_metadata.restype = ct.c_char_p
        self._lib.posted_reader_field.argtypes = [ct.c_void_p, ct.c_char_p,
                                                ct.c_char_p, ct.POINTER(ct.c_size_t)]
        self._lib.posted_reader_field.restype = ct.POINTER(ct.c_double)
        self._lib.posted_reader_land.argtypes = [ct.c_void_p]
        self._lib.posted_reader_land.restype = ct.POINTER(ct.c_uint8)
        self._lib.posted_reader_locate.argtypes = [ct.c_void_p, ct.c_double,
            ct.c_double, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
        self._lib.posted_reader_locate.restype = ct.c_int
        self._handle = None
        self._handle = self._lib.posted_reader_open(
            str(Path(field_table).resolve()).encode("utf-8"),
            json.dumps([str(p.resolve()) for p in paths]).encode("utf-8"))
        if not self._handle:
            raise ValueError(self._error())
        self._record = json.loads(self._lib.posted_reader_metadata(self._handle))
        self._record["reader_library_sha256"] = self._library_sha256
        self._shape = tuple(self._record["shape"])

    def _error(self):
        return self._lib.posted_reader_error().decode("utf-8")

    def __del__(self):
        if getattr(self, "_handle", None):
            self._lib.posted_reader_close(self._handle)
            self._handle = None

    def _array(self, pointer, length, ctype):
        buffer = (ctype * length).from_address(ct.addressof(pointer.contents))
        # A view keeps the reader alive, even when retained beyond the source
        # variable's lifetime. The native handle owns every pointed-to array.
        buffer._reader_owner = self
        result = np.ctypeslib.as_array(buffer).reshape(self._shape)
        result.flags.writeable = False
        return result

    def _field(self, time, name):
        length = ct.c_size_t()
        pointer = self._lib.posted_reader_field(
            self._handle, str(time).encode("utf-8"), name.encode("utf-8"),
            ct.byref(length))
        if not pointer:
            raise ValueError(self._error())
        return self._array(pointer, length.value, ct.c_double)

    def valid_times(self):
        return tuple(self._record["valid_times"])

    def grid(self):
        return ModelGrid(latitude=self._field("", "latitude"),
                         longitude=self._field("", "longitude"),
                         dx_m=float(self._record["dx_m"]),
                         terrain_m=self._field(self.valid_times()[0], "terrain_height"))

    def station_locator(self):
        def locate(station_id, latitude, longitude):
            x, y = ct.c_double(), ct.c_double()
            if self._lib.posted_reader_locate(self._handle, float(latitude),
                    float(longitude), ct.byref(x), ct.byref(y)):
                raise ValueError(self._error())
            return StationPosition(station_id=str(station_id), x=x.value, y=y.value)
        return locate

    def land_mask(self):
        return self._array(self._lib.posted_reader_land(self._handle),
                           self._shape[0] * self._shape[1], ct.c_bool)

    def composite_reflectivity(self, valid_time):
        return self._field(valid_time, "composite_reflectivity")

    def surface_field(self, valid_time, variable):
        return self._field(valid_time, variable)

    def precipitation_accumulation(self, valid_time):
        raise ValueError("this instantaneous-field reader does not decode run-total precipitation intervals; scoring it would silently change the accumulation window")

    def record(self):
        return self._record.copy()
