// Keep this detector ASCII-only so the repair layer cannot itself be damaged
// by an editor or terminal choosing the wrong code page. These are the common
// leading sequences produced when Vietnamese UTF-8 bytes are decoded as
// Windows-1252 one or more times (for example: "triá»‡u" -> "triệu").
const MOJIBAKE_MARKERS = /(?:\u00c3|\u00c2|\u00e1[\u00ba\u00bb]|\u00e2[\u0080\u2013\u2014\u2018\u2019\u201a\u201c\u201d\u201e\u2020\u2021\u2022\u2026\u20ac\u2122])/;

const WINDOWS_1252_BYTES = new Map<number, number>([
  [0x20ac, 0x80], [0x201a, 0x82], [0x0192, 0x83], [0x201e, 0x84],
  [0x2026, 0x85], [0x2020, 0x86], [0x2021, 0x87], [0x02c6, 0x88],
  [0x2030, 0x89], [0x0160, 0x8a], [0x2039, 0x8b], [0x0152, 0x8c],
  [0x017d, 0x8e], [0x2018, 0x91], [0x2019, 0x92], [0x201c, 0x93],
  [0x201d, 0x94], [0x2022, 0x95], [0x2013, 0x96], [0x2014, 0x97],
  [0x02dc, 0x98], [0x2122, 0x99], [0x0161, 0x9a], [0x203a, 0x9b],
  [0x0153, 0x9c], [0x017e, 0x9e], [0x0178, 0x9f],
]);

function decodeWindows1252Utf8(value: string) {
  if (!MOJIBAKE_MARKERS.test(value)) return value;
  const bytes: number[] = [];
  for (const character of value) {
    const point = character.codePointAt(0) ?? 0;
    if (point <= 0xff) bytes.push(point);
    else if (WINDOWS_1252_BYTES.has(point)) bytes.push(WINDOWS_1252_BYTES.get(point)!);
    else return value;
  }
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(Uint8Array.from(bytes));
  } catch {
    return value;
  }
}

export function repairVietnameseText(value: string) {
  let repaired = value;
  for (let pass = 0; pass < 3; pass += 1) {
    const next = decodeWindows1252Utf8(repaired);
    if (next === repaired) break;
    repaired = next;
  }
  return repaired;
}

export function repairTextDeep<T>(value: T): T {
  if (typeof value === "string") return repairVietnameseText(value) as T;
  if (Array.isArray(value)) return value.map((item) => repairTextDeep(item)) as T;
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, item]) => [key, repairTextDeep(item)]),
    ) as T;
  }
  return value;
}
