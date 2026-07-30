/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
 * Copyright (c) 2002-2014, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2014, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux
 * Copyright (c) 2003-2014, Antonin Descampe
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
 * Copyright (c) 2008, 2011-2012, Centre National d'Etudes Spatiales (CNES), FR
 * Copyright (c) 2012, CS Systemes d'Information, France
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

use std::io::{BufReader, BufWriter, Error as IoError, Read, Seek, SeekFrom, Write};

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

#[cfg(feature = "file-io")]
use std::{fs::File, path::Path};

use super::event::*;
use super::openjpeg::*;

extern "C" {
  fn memcpy(
    _: *mut core::ffi::c_void,
    _: *const core::ffi::c_void,
    _: usize,
  ) -> *mut core::ffi::c_void;
}

pub(crate) trait ReadSeek: Read + Seek {}

impl<R: Read + Seek> ReadSeek for R {}

pub(crate) trait WriteSeek: Write + Seek {}

impl<R: Write + Seek> WriteSeek for R {}

pub(crate) struct CustomStream {
  pub m_user_data: *mut core::ffi::c_void,
  pub m_free_user_data_fn: opj_stream_free_user_data_fn,
  pub m_read_fn: opj_stream_read_fn,
  pub m_write_fn: opj_stream_write_fn,
  pub m_skip_fn: opj_stream_skip_fn,
  pub m_seek_fn: opj_stream_seek_fn,
  pub m_byte_offset: OPJ_OFF_T,
}

impl CustomStream {
  pub fn set_read(&mut self, read: opj_stream_read_fn) {
    self.m_read_fn = read;
  }

  pub fn set_write(&mut self, write: opj_stream_write_fn) {
    self.m_write_fn = write;
  }

  pub fn set_skip(&mut self, skip: opj_stream_skip_fn) {
    self.m_skip_fn = skip;
  }

  pub fn set_seek(&mut self, seek: opj_stream_seek_fn) {
    self.m_seek_fn = seek;
  }

  pub fn set_user_data(
    &mut self,
    data: *mut core::ffi::c_void,
    free: opj_stream_free_user_data_fn,
  ) {
    self.m_user_data = data;
    self.m_free_user_data_fn = free;
  }

  pub fn has_seek(&self) -> bool {
    self.m_seek_fn.is_some()
  }
}

impl Drop for CustomStream {
  fn drop(&mut self) {
    if let Some(free) = self.m_free_user_data_fn {
      unsafe {
        free(self.m_user_data);
      }
    }
  }
}

impl Read for CustomStream {
  fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
    let len = buf.len();
    let res = if let Some(read) = &self.m_read_fn {
      unsafe {
        read(
          buf.as_mut_ptr() as *mut core::ffi::c_void,
          len,
          self.m_user_data,
        ) as isize
      }
    } else {
      return Err(IoError::other("Custom stream doesn't have a read function"));
    };
    if res >= 0 {
      self.m_byte_offset += res as i64;
      Ok(res as usize)
    } else {
      Err(IoError::other("read failed"))
    }
  }
}

impl Write for CustomStream {
  fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
    let len = buf.len();
    let res = if let Some(write) = &self.m_write_fn {
      unsafe {
        write(
          buf.as_ptr() as *mut core::ffi::c_void,
          len,
          self.m_user_data,
        ) as isize
      }
    } else {
      return Err(IoError::other(
        "Custom stream doesn't have a write function",
      ));
    };
    if res >= 0 {
      self.m_byte_offset += res as i64;
      Ok(res as usize)
    } else {
      Err(IoError::other("write failed"))
    }
  }

  fn flush(&mut self) -> std::io::Result<()> {
    Ok(())
  }
}

impl Seek for CustomStream {
  fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
    log::trace!("-- CustomStream.seek({pos:?})");
    match pos {
      SeekFrom::Start(offset) => {
        let res = if let Some(seek) = &self.m_seek_fn {
          unsafe { seek(offset as i64, self.m_user_data) }
        } else {
          return Err(IoError::other("Custom stream doesn't have a seek function"));
        };
        if res != 0 {
          self.m_byte_offset = offset as i64;
          Ok(offset)
        } else {
          Err(IoError::other("seek failed"))
        }
      }
      SeekFrom::Current(offset) => {
        let res = if let Some(skip) = &self.m_skip_fn {
          unsafe { skip(offset, self.m_user_data) }
        } else {
          return Err(IoError::other("Custom stream doesn't have a skip function"));
        };
        if res != -1 {
          self.m_byte_offset += offset;
          Ok(self.m_byte_offset as u64)
        } else {
          Err(IoError::other("skip failed"))
        }
      }
      SeekFrom::End(_offset) => Err(IoError::other("Unsupported seek from end")),
    }
  }
}

pub(crate) enum StreamInner {
  Reader(BufReader<Box<dyn ReadSeek>>),
  Writer(BufWriter<Box<dyn WriteSeek>>),
  CustomReader(BufReader<CustomStream>),
  CustomWriter(BufWriter<CustomStream>),
}

impl StreamInner {
  pub fn new_reader<R: Read + Seek + 'static>(capacity: usize, reader: R) -> Self {
    Self::Reader(BufReader::with_capacity(capacity, Box::new(reader)))
  }

  pub fn new_writer<R: Write + Seek + 'static>(capacity: usize, writer: R) -> Self {
    Self::Writer(BufWriter::with_capacity(capacity, Box::new(writer)))
  }

  pub fn new_custom_reader(capacity: usize, reader: CustomStream) -> Self {
    Self::CustomReader(BufReader::with_capacity(capacity, reader))
  }

  pub fn new_custom_writer(capacity: usize, writer: CustomStream) -> Self {
    Self::CustomWriter(BufWriter::with_capacity(capacity, writer))
  }

  pub fn seek_relative(&mut self, offset: i64) -> std::io::Result<()> {
    match self {
      StreamInner::Reader(reader) => reader.seek_relative(offset),
      StreamInner::Writer(writer) => {
        writer.seek(SeekFrom::Current(offset))?;
        Ok(())
      }
      StreamInner::CustomReader(reader) => reader.seek_relative(offset),
      StreamInner::CustomWriter(writer) => {
        writer.seek(SeekFrom::Current(offset))?;
        Ok(())
      }
    }
  }

  pub fn has_seek(&self) -> bool {
    match self {
      StreamInner::Reader(_) => true,
      StreamInner::Writer(_) => true,
      StreamInner::CustomReader(b) => b.get_ref().has_seek(),
      StreamInner::CustomWriter(b) => b.get_ref().has_seek(),
    }
  }
}

impl Read for StreamInner {
  fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
    match self {
      StreamInner::Reader(reader) => reader.read(buf),
      StreamInner::CustomReader(reader) => reader.read(buf),
      _ => Err(IoError::other("Can't read from output stream.")),
    }
  }
}

impl Write for StreamInner {
  fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
    match self {
      StreamInner::Writer(writer) => writer.write(buf),
      StreamInner::CustomWriter(writer) => writer.write(buf),
      _ => Err(IoError::other("Can't write to input stream.")),
    }
  }

  fn flush(&mut self) -> std::io::Result<()> {
    match self {
      StreamInner::Writer(writer) => writer.flush(),
      StreamInner::CustomWriter(writer) => writer.flush(),
      _ => Err(IoError::other("Can't flush input stream.")),
    }
  }
}

impl Seek for StreamInner {
  fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
    match self {
      StreamInner::Reader(reader) => reader.seek(pos),
      StreamInner::Writer(writer) => writer.seek(pos),
      StreamInner::CustomReader(reader) => reader.seek(pos),
      StreamInner::CustomWriter(writer) => writer.seek(pos),
    }
  }
}

impl Stream {
  #[cfg(feature = "file-io")]
  pub fn new_file<P: AsRef<Path>>(
    path: P,
    buffer_size: usize,
    is_input: bool,
  ) -> std::io::Result<Self> {
    if is_input {
      let file = File::open(&path)?;
      let m_stream_length = file.metadata().map(|m| m.len())?;
      Ok(Self {
        m_inner: super::stream::StreamInner::new_reader(buffer_size, file),
        m_stream_length,
        m_byte_offset: 0,
      })
    } else {
      let file = File::create(&path)?;
      Ok(Self {
        m_inner: super::stream::StreamInner::new_writer(buffer_size, file),
        m_stream_length: 0,
        m_byte_offset: 0,
      })
    }
  }

  pub fn new_custom(buffer_size: usize, is_input: bool) -> Self {
    let custom = CustomStream {
      m_user_data: core::ptr::null_mut(),
      m_free_user_data_fn: None,
      m_read_fn: None,
      m_write_fn: None,
      m_skip_fn: None,
      m_seek_fn: None,
      m_byte_offset: 0,
    };
    let mut l_stream = Self {
      m_inner: if is_input {
        StreamInner::CustomReader(BufReader::with_capacity(buffer_size, custom))
      } else {
        StreamInner::CustomWriter(BufWriter::with_capacity(buffer_size, custom))
      },
      m_stream_length: Default::default(),
      m_byte_offset: 0,
    };
    l_stream
  }

  pub fn is_input(&self) -> bool {
    match self.m_inner {
      StreamInner::Reader(_) => true,
      StreamInner::CustomReader(_) => true,
      StreamInner::Writer(_) => false,
      StreamInner::CustomWriter(_) => false,
    }
  }

  pub(crate) fn as_custom(&mut self) -> Option<&mut CustomStream> {
    match &mut self.m_inner {
      StreamInner::Reader(_) => None,
      StreamInner::Writer(_) => None,
      StreamInner::CustomReader(reader) => Some(reader.get_mut()),
      StreamInner::CustomWriter(writer) => Some(writer.get_mut()),
    }
  }

  pub fn set_stream_length(&mut self, len: u64) {
    self.m_stream_length = len;
  }

  pub fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
    let len = buf.len();
    match self.m_inner.read_exact(buf) {
      Ok(_) => {
        self.m_byte_offset += len as i64;
        Ok(len)
      }
      Err(_err) => {
        // Maybe EOF, do a partial read.
        match self.m_inner.read(buf) {
          Ok(nb) => {
            self.m_byte_offset += nb as i64;
            Ok(nb)
          }
          Err(err) => {
            log::trace!("Failed to read from stream: {err}");
            Err(err)
          }
        }
      }
    }
  }

  pub fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
    let len = buf.len();
    log::trace!("-- write({len}), offset={}", self.m_byte_offset);
    match self.m_inner.write_all(buf) {
      Ok(_) => {
        self.m_byte_offset += len as i64;
        Ok(len)
      }
      Err(err) => {
        log::trace!("Failed to write to stream: {err}");
        Err(err)
      }
    }
  }

  pub fn flush(&mut self) -> std::io::Result<()> {
    log::trace!("-- flush(), offset={}", self.m_byte_offset);
    self.m_inner.flush()
  }

  pub fn tell(&self) -> i64 {
    log::trace!("-- tell() = {}", self.m_byte_offset);
    self.m_byte_offset
  }

  pub fn get_bytes_left(&self) -> i64 {
    log::trace!("-- byte_left(), offset={}", self.m_byte_offset);
    assert!(self.m_byte_offset >= 0i64);
    assert!(self.m_stream_length >= self.m_byte_offset as OPJ_UINT64);
    let nb = if self.m_stream_length != 0 {
      (self.m_stream_length as OPJ_OFF_T) - self.m_byte_offset
    } else {
      0i64
    };
    log::trace!("-- get_number_byte_left() = {}", nb);
    nb
  }

  pub fn skip(&mut self, count: i64) -> std::io::Result<i64> {
    if count < 0 {
      log::trace!("Can't skip with count < 0: {count}");
      return Err(std::io::Error::other(format!(
        "Can't skip with count < 0: {count}"
      )));
    }
    if self.m_stream_length > 0 {
      let new_offset = self.m_byte_offset + count;
      if (new_offset as u64) > self.m_stream_length {
        log::trace!("Skip pass the end of the stream.");
        return Err(std::io::Error::other(format!(
          "Skip pass the end of the stream"
        )));
      }
    }
    let res = self
      .m_inner
      .seek_relative(count)
      .map(|_| (self.m_byte_offset + count) as u64);
    match res {
      Ok(offset) => {
        self.m_byte_offset = offset as i64;
        // TODO: return number of bytes skipped.
        log::trace!("-- skip({count}) = {offset}");
        Ok(count)
      }
      Err(err) => {
        log::trace!("Failed to skip stream: {err}");
        Err(err)
      }
    }
  }

  pub fn seek(&mut self, offset: i64) -> std::io::Result<()> {
    let res = self.m_inner.seek_relative(offset - self.m_byte_offset);
    match res {
      Ok(_) => {
        self.m_byte_offset = offset;
        log::trace!("-- seek({offset}) = Ok");
        Ok(())
      }
      Err(err) => {
        log::trace!("Failed to seek stream: {err}");
        Err(err)
      }
    }
  }

  pub fn has_seek(&self) -> bool {
    self.m_inner.has_seek()
  }
}

impl Read for Stream {
  fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
    self.read(buf)
  }
}

impl Write for Stream {
  fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
    self.write(buf)
  }

  fn flush(&mut self) -> std::io::Result<()> {
    self.flush()
  }
}

pub(crate) fn opj_stream_read_data(
  mut p_stream: *mut opj_stream_private_t,
  mut p_buffer: *mut OPJ_BYTE,
  mut p_size: OPJ_SIZE_T,
  mut _p_event_mgr: &mut opj_event_mgr,
) -> OPJ_SIZE_T {
  let p_stream = unsafe { &mut *p_stream };
  let buf = unsafe { core::slice::from_raw_parts_mut(p_buffer as *mut u8, p_size) };
  match p_stream.read(buf) {
    Ok(nb) => nb as OPJ_SIZE_T,
    Err(err) => {
      log::trace!("Failed to read from stream: {err}");
      -1i32 as OPJ_SIZE_T
    }
  }
}

pub(crate) fn opj_stream_write_data(
  mut p_stream: *mut opj_stream_private_t,
  mut p_buffer: *const OPJ_BYTE,
  mut p_size: OPJ_SIZE_T,
  mut _p_event_mgr: &mut opj_event_mgr,
) -> OPJ_SIZE_T {
  let p_stream = unsafe { &mut *p_stream };
  log::trace!("-- write({p_size}), offset={}", p_stream.m_byte_offset);
  let buf = unsafe { core::slice::from_raw_parts(p_buffer as *const u8, p_size) };
  match p_stream.write(buf) {
    Ok(nb) => nb,
    Err(err) => {
      log::trace!("Failed to write to stream: {err}");
      -1i32 as OPJ_SIZE_T
    }
  }
}

pub(crate) fn opj_stream_flush(
  mut p_stream: *mut opj_stream_private_t,
  mut _p_event_mgr: &mut opj_event_mgr,
) -> OPJ_BOOL {
  let p_stream = unsafe { &mut *p_stream };
  return match p_stream.flush() {
    Ok(_) => 1,
    Err(err) => {
      log::trace!("Failed to flush stream: {err}");
      0
    }
  };
}

pub(crate) fn opj_stream_tell(mut p_stream: *mut opj_stream_private_t) -> OPJ_OFF_T {
  let p_stream = unsafe { &mut *p_stream };
  p_stream.tell()
}

pub(crate) fn opj_stream_get_number_byte_left(
  mut p_stream: *mut opj_stream_private_t,
) -> OPJ_OFF_T {
  let p_stream = unsafe { &mut *p_stream };
  p_stream.get_bytes_left()
}

pub(crate) fn opj_stream_skip(
  mut p_stream: *mut opj_stream_private_t,
  mut p_size: OPJ_OFF_T,
  mut _p_event_mgr: &mut opj_event_mgr,
) -> OPJ_OFF_T {
  let p_stream = unsafe { &mut *p_stream };
  match p_stream.skip(p_size) {
    Ok(offset) => {
      log::trace!("-- skip({p_size}) = {}", offset);
      offset
    }
    Err(err) => {
      log::trace!("Failed to skip stream: {err}");
      -1i32 as _
    }
  }
}

pub(crate) fn opj_stream_seek(
  mut p_stream: *mut opj_stream_private_t,
  mut p_size: OPJ_OFF_T,
  mut _p_event_mgr: &mut opj_event_mgr,
) -> OPJ_BOOL {
  let p_stream = unsafe { &mut *p_stream };
  match p_stream.seek(p_size) {
    Ok(_) => {
      log::trace!("-- seek({p_size}) = {}", 1);
      1
    }
    Err(err) => {
      log::trace!("Failed to seek stream: {err}");
      0
    }
  }
}

pub(crate) fn opj_stream_has_seek(mut p_stream: *const opj_stream_private_t) -> OPJ_BOOL {
  let p_stream = unsafe { &*p_stream };
  p_stream.has_seek() as _
}
