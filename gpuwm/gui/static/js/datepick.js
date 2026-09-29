// A calendar for one UTC day: a month grid with month and year jumps. Days that have not happened yet cannot be
// picked; days outside the chosen source's archive are drawn faint but can still be picked, because another source
// may hold them. Everything is UTC: a forecast starts at a UTC hour.

import { h } from "./core.js";

const pad = (n) => String(n).padStart(2, "0");
export const dayText = (d) => `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`;
export const todayText = () => dayText(new Date());

// "1999-05-03", "1999/5/3", "1999-05-03 18", "1999-05-03T18Z", "19990503" -> {day: "1999-05-03", hour: 18 | null}
export function parseTyped(text) {
  const t = String(text || "").trim().toUpperCase();
  const m = /^(\d{4})[-/. ]?(\d{1,2})[-/. ]?(\d{1,2})(?:[T ]+(\d{1,2})(?::00)?Z?)?(?:\s*UTC)?$/.exec(t);
  if (!m) return null;
  const [y, mo, d] = [Number(m[1]), Number(m[2]), Number(m[3])];
  const date = new Date(Date.UTC(y, mo - 1, d));
  if (date.getUTCFullYear() !== y || date.getUTCMonth() !== mo - 1 || date.getUTCDate() !== d) return null;
  const hour = m[4] === undefined ? null : Number(m[4]);
  if (hour !== null && hour > 23) return null;
  return { day: dayText(date), hour };
}

// calendar({value, min, words, faint(day) -> bool, onpick(day)}) -> {el, set(value), paint(), shade()}
export function calendar(opts) {
  const w = opts.words;
  let value = opts.value;
  let view = new Date(`${value}T00:00:00Z`);
  view.setUTCDate(1);
  const el = h("div", { class: "cal", role: "group", "aria-label": w.calendar });

  function move(months) {
    view = new Date(Date.UTC(view.getUTCFullYear(), view.getUTCMonth() + months, 1));
    paint();
  }

  function paint() {
    const today = todayText();
    const first = Number(opts.min.slice(0, 4));
    const last = Number(today.slice(0, 4));
    const year = h("select", { class: "input", "aria-label": w.year },
      Array.from({ length: last - first + 1 }, (_, i) => last - i).map((y) => h("option", { value: String(y) }, String(y))));
    year.value = String(view.getUTCFullYear());
    year.addEventListener("change", () => { view = new Date(Date.UTC(Number(year.value), view.getUTCMonth(), 1)); paint(); });
    const month = h("select", { class: "input", "aria-label": w.month },
      w.months.map((name, i) => h("option", { value: String(i) }, name)));
    month.value = String(view.getUTCMonth());
    month.addEventListener("change", () => { view = new Date(Date.UTC(view.getUTCFullYear(), Number(month.value), 1)); paint(); });
    const prev = h("button", { type: "button", class: "btn small", title: w.prev_month, "aria-label": w.prev_month }, "‹");
    prev.addEventListener("click", () => move(-1));
    const next = h("button", { type: "button", class: "btn small", title: w.next_month, "aria-label": w.next_month }, "›");
    next.addEventListener("click", () => move(1));
    const earliest = `${opts.min}`;
    const monthEnd = new Date(Date.UTC(view.getUTCFullYear(), view.getUTCMonth() + 1, 0));
    prev.disabled = dayText(new Date(view.getTime() - 86400000)) < earliest;
    next.disabled = dayText(new Date(monthEnd.getTime() + 86400000)) > today;

    // Monday first; blank cells before the 1st.
    const lead = (view.getUTCDay() + 6) % 7;
    const cells = [];
    for (let i = 0; i < lead; i += 1) cells.push(h("span", { class: "blank" }));
    for (let d = 1; d <= monthEnd.getUTCDate(); d += 1) {
      const text = dayText(new Date(Date.UTC(view.getUTCFullYear(), view.getUTCMonth(), d)));
      const future = text > today;
      const before = text < earliest;
      const cls = ["day", text === value ? "on" : "", text === today ? "today" : "",
        !future && !before && opts.faint && opts.faint(text) ? "faint" : ""].filter(Boolean).join(" ");
      const b = h("button", { type: "button", class: cls, disabled: future || before || null,
        "aria-pressed": text === value ? "true" : "false", title: text }, String(d));
      b.addEventListener("click", () => { value = text; opts.onpick(text); paint(); });
      cells.push(b);
    }
    el.replaceChildren(
      h("div", { class: "calhead" }, prev, month, year, next),
      h("div", { class: "calgrid" }, w.weekdays.map((d) => h("span", { class: "wd" }, d)), cells));
  }

  // Only which days are faint, changed in place: the month and year boxes and the day buttons stay as they are, so
  // an open box stays open and a click on a day is never lost to a redraw.
  function shade() {
    for (const b of el.querySelectorAll(".calgrid button.day")) {
      b.classList.toggle("faint", !b.disabled && !!(opts.faint && opts.faint(b.title)));
    }
  }

  paint();
  return {
    el,
    paint,
    shade,
    set(next) {
      value = next;
      view = new Date(`${next}T00:00:00Z`);
      view.setUTCDate(1);
      paint();
    },
  };
}
