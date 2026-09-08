#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[derive(Clone)]
pub(crate) struct ProcedureList<P> {
  list: Vec<P>,
}

impl<P> ProcedureList<P> {
  pub fn new() -> Self {
    Self {
      list: Default::default(),
    }
  }

  pub fn add(&mut self, procedure: P) {
    self.list.push(procedure)
  }

  pub fn execute<F: FnMut(P) -> bool>(&mut self, mut eval: F) -> bool {
    for p in self.list.drain(..) {
      if !eval(p) {
        return false;
      }
    }
    true
  }
}
