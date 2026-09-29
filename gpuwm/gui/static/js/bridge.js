// What the assistant hands the Create form, and what the form hands back. The assistant never writes a run:
// it fills the form here, where the person sees every field change and can change it back.

let last = null;
const subs = new Set();
let reader = null;

// {fields, reasons, fit}: fields the form sets, one-line reasons shown under them, the fit to draw
export function fillCreate(detail) {
  last = detail;
  for (const cb of subs) cb(detail);
}

// the fill waiting for a Create form that is not on screen yet
export function takeFill() {
  const value = last;
  last = null;
  return value;
}

export function onFill(cb) {
  subs.add(cb);
  return () => subs.delete(cb);
}

// the Create form registers how to read itself; the assistant sends what the person sees
export function setFormReader(fn) {
  reader = fn;
  return () => { if (reader === fn) reader = null; };
}

export function readForm() {
  try { return reader ? reader() : null; } catch (_) { return null; }
}
