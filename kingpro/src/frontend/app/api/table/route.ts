import { NextRequest, NextResponse } from "next/server";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { repairTextDeep } from "@/lib/text";

export const dynamic = "force-dynamic";

type CatalogTable = {
  table_ref: string;
  csv_path: string;
  n_rows?: number;
  n_cols?: number;
};

let tableCatalog: Promise<Map<string, CatalogTable>> | null = null;

function loadTableCatalog() {
  if (!tableCatalog) {
    tableCatalog = (async () => {
      const root = path.resolve(process.env.KINGPRO_DATA_ROOT ?? path.resolve(process.cwd(), ".."));
      const source = await readFile(path.join(root, "build", "catalog.jsonl"), "utf8");
      const rows = new Map<string, CatalogTable>();
      for (const line of source.split(/\r?\n/)) {
        if (!line.trim()) continue;
        const row = JSON.parse(line) as CatalogTable;
        if (row.table_ref && row.csv_path) rows.set(row.table_ref, row);
      }
      return rows;
    })();
  }
  return tableCatalog;
}

function parseCsv(source: string) {
  const records: string[][] = [];
  let record: string[] = [];
  let field = "";
  let quoted = false;

  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    if (char === '"') {
      if (quoted && source[index + 1] === '"') {
        field += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === "," && !quoted) {
      record.push(field);
      field = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && source[index + 1] === "\n") index += 1;
      record.push(field);
      if (record.some((cell) => cell.length > 0)) records.push(record);
      record = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field.length > 0 || record.length > 0) {
    record.push(field);
    records.push(record);
  }
  return records;
}

export async function GET(request: NextRequest) {
  const tableRef = request.nextUrl.searchParams.get("ref")?.trim() ?? "";
  if (!tableRef || tableRef.length > 240) {
    return NextResponse.json({ status: "error", error: "Thiếu table reference hợp lệ." }, { status: 400 });
  }

  try {
    const catalog = await loadTableCatalog();
    const table = catalog.get(tableRef);
    if (!table) {
      return NextResponse.json({ status: "error", error: "Table reference không tồn tại trong catalog." }, { status: 404 });
    }

    const root = path.resolve(process.env.KINGPRO_DATA_ROOT ?? path.resolve(process.cwd(), ".."));
    const tableRoot = path.resolve(root, "build", "tables");
    const csvPath = path.resolve(tableRoot, table.csv_path);
    if (csvPath !== tableRoot && !csvPath.startsWith(`${tableRoot}${path.sep}`)) {
      return NextResponse.json({ status: "error", error: "Đường dẫn bảng không hợp lệ." }, { status: 400 });
    }

    const source = (await readFile(csvPath, "utf8")).replace(/^\uFEFF/, "");
    const parsed = parseCsv(source);
    const [columns = [], ...allRows] = parsed;
    // The evidence dialog is a full table inspector.  A 14-row teaser hid
    // audited cells such as row 17 and made exact lineage impossible to see.
    // Keep a bounded safety cap for pathological OCR tables while returning
    // every row for normal financial statements.
    const previewRows = allRows.slice(0, 500).map((row) => columns.map((_, index) => (row[index] ?? "").slice(0, 220)));
    return NextResponse.json(repairTextDeep({
      status: "ok",
      table_ref: tableRef,
      columns,
      rows: previewRows,
      row_count: allRows.length,
      column_count: columns.length,
      truncated: allRows.length > previewRows.length,
    }));
  } catch (error) {
    return NextResponse.json(
      { status: "error", error: error instanceof Error ? error.message : "Không thể đọc bảng nguồn." },
      { status: 500 },
    );
  }
}
