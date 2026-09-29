// Plain names and headings for the renderer's product folders, from copy/looks.json. A product without an
// entry gets its folder name made readable, so a new product shows up named without code.

const UPPER = new Set(["cape", "cin", "srh", "uh", "ehi", "stp", "qpf", "mslp", "rh", "ir", "lcl", "lfc", "pw",
  "scp", "tts", "vtp", "hdw", "olr", "pblh", "wrf", "tehi", "bri"]);
const LEVELS = [
  [/_height_winds$/, "_height_and_wind"],
  [/_winds$/, "_and_wind"],
  [/^(\d+)mb_/, "$1 mb "],
  [/(^|_)(\d+)m_/g, "$1$2 m "],
  [/_(\d+)hpa/g, " $1 hPa"],
  [/_(\d+)_(\d+)km/g, " $1 to $2 km"],
  [/_(\d+)to(\d+)km/g, " $1 to $2 km"],
];

export function productName(product, looks) {
  const names = (looks && looks.names) || {};
  if (names[product]) return names[product];
  let text = String(product);
  const raw = text.startsWith("var_");
  if (raw) text = text.slice(4).replace(/_[0-9a-f]{16}$/, "").replace(/^wrf_/, "");
  for (const [re, to] of LEVELS) text = text.replace(re, to);
  const words = text.split(/[_\s]+/).filter(Boolean).map((w) => (UPPER.has(w.toLowerCase()) ? w.toUpperCase() : w));
  const out = words.join(" ").replace(/\bminus(\d+)c\b/, "minus $1 C").replace(/\b(\d+)c\b/, "$1 C");
  return out.charAt(0).toUpperCase() + out.slice(1);
}

export function productGroup(product, looks) {
  const low = String(product).toLowerCase();
  for (const group of (looks && looks.groups) || []) {
    if ((group.match || []).some((m) => low.includes(m))) return group.name;
  }
  return null;
}

// [{name, products: [...]}], favourites first, then each heading in the order looks.json lists them.
export function groupProducts(products, favourites, looks, words) {
  const out = [];
  const favs = products.filter((p) => favourites.includes(p));
  if (favs.length) out.push({ name: words.favourites, products: favs });
  const byGroup = new Map();
  for (const g of (looks && looks.groups) || []) byGroup.set(g.name, []);
  const other = [];
  for (const p of products) {
    if (favs.includes(p)) continue;
    const g = productGroup(p, looks);
    if (g && byGroup.has(g)) byGroup.get(g).push(p);
    else other.push(p);
  }
  // a heading marked "last" (the raw model fields: many, and rarely the first thing wanted) goes at the end
  const sorted = (list) => list.sort((a, b) => productName(a, looks).localeCompare(productName(b, looks)));
  const last = [];
  for (const g of (looks && looks.groups) || []) {
    const list = byGroup.get(g.name);
    if (!list || !list.length) continue;
    (g.last ? last : out).push({ name: g.name, products: sorted(list) });
    byGroup.delete(g.name);
  }
  if (other.length) out.push({ name: words.other, products: sorted(other) });
  out.push(...last);
  return out;
}

// The heading a product sorts under on the Files page: the first kind in looks.json whose pattern it holds, the
// model's own fields under their own heading, anything else under "Other".
export function kindOf(product, looks) {
  if (looks.raw_prefix && product.startsWith(looks.raw_prefix)) return looks.raw_kind;
  for (const kind of looks.kinds || []) {
    if (kind.patterns.some((p) => product.includes(p))) return kind.name;
  }
  return looks.other_kind;
}
