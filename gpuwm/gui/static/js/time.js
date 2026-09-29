// Times as the page says them: always UTC, the way forecasters write them.

import { fill } from "./core.js";

const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const pad = (n) => String(n).padStart(2, "0");

// "2026-09-24 12:00Z", "2026-09-24T12:00:00Z", "2026-09-24T12" -> Date (UTC), or null
export function parseTime(text) {
  if (!text) return null;
  const m = /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2})(?::(\d{2}))?(?::(\d{2}))?/.exec(String(text));
  if (!m) return null;
  return new Date(Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +(m[5] || 0), +(m[6] || 0)));
}

// "2026-09-24 12:00 UTC": the one way a moment is written in lists, facts and choices (the clock the viewer shows)
export function utcText(text) {
  const d = text instanceof Date ? text : parseTime(text);
  if (!d) return "";
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())} UTC`;
}

// "Thu 24 Sep, 13:00 UTC"
export function longTime(text) {
  const d = text instanceof Date ? text : parseTime(text);
  if (!d) return "";
  return `${DAYS[d.getUTCDay()]} ${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}, ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())} UTC`;
}

// "1 h", "36 h", "30 min"
export function spanWords(seconds) {
  if (seconds === null || seconds === undefined) return "";
  const h = seconds / 3600;
  return h >= 1 ? `${+h.toFixed(1)} h` : `${Math.round(seconds / 60)} min`;
}

export function hourWords(modelSeconds, runSeconds, template) {
  const now = Math.floor((modelSeconds || 0) / 3600);
  const total = Math.round((runSeconds || 0) / 3600);
  return fill(template, { now, total });
}

// "2 h 05 min", "4 min", "40 s"
export function leftWords(seconds) {
  const s = Math.max(0, Math.round(seconds || 0));
  if (s >= 3600) return `${Math.floor(s / 3600)} h ${pad(Math.floor((s % 3600) / 60))} min`;
  if (s >= 60) return `${Math.round(s / 60)} min`;
  return `${s} s`;
}
