import { NextResponse } from "next/server";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { repairTextDeep } from "@/lib/text";
import { backendHeaders } from "@/lib/backend-auth";

export const dynamic = "force-dynamic";

type CatalogRow = { table_ref: string; report_id: string; ticker: string; year: string; scope: string };

let localSummary: Promise<Record<string, unknown>> | null = null;

async function readLocalCatalog() {
  if (!localSummary) {
    localSummary = (async () => {
      const root = path.resolve(process.env.KINGPRO_DATA_ROOT ?? path.resolve(process.cwd(), ".."));
      const [catalogText, companyText] = await Promise.all([
        readFile(path.join(root, "build", "catalog.jsonl"), "utf8"),
        readFile(path.join(root, "data", "code_stock.csv"), "utf8"),
      ]);
      const names = new Map(companyText.trim().split(/\r?\n/).slice(1).map((line) => {
        const comma = line.indexOf(",");
        return [line.slice(0, comma).trim(), line.slice(comma + 1).trim()] as const;
      }));
      const reports = new Set<string>();
      const years = new Set<string>();
      const scopeCounts: Record<string, number> = {};
      const companies = new Map<string, { ticker: string; company: string; years: Set<string>; reports: Set<string>; tables: number; scopes: Set<string> }>();
      for (const line of catalogText.trim().split(/\r?\n/)) {
        const row = JSON.parse(line) as CatalogRow;
        reports.add(row.report_id); years.add(row.year); scopeCounts[row.scope] = (scopeCounts[row.scope] ?? 0) + 1;
        const company = companies.get(row.ticker) ?? { ticker: row.ticker, company: names.get(row.ticker) ?? row.ticker, years: new Set<string>(), reports: new Set<string>(), tables: 0, scopes: new Set<string>() };
        company.years.add(row.year); company.reports.add(row.report_id); company.scopes.add(row.scope); company.tables += 1; companies.set(row.ticker, company);
      }
      const companyRows = [...companies.values()].map((company) => {
        const companyYears = [...company.years].sort();
        return { ticker: company.ticker, company: company.company, year_from: companyYears[0] ?? "", year_to: companyYears.at(-1) ?? "", year_count: companyYears.length, report_count: company.reports.size, table_count: company.tables, scopes: [...company.scopes].sort() };
      }).sort((a, b) => a.ticker.localeCompare(b.ticker));
      return repairTextDeep({ status: "ok", company_count: companyRows.length, report_count: reports.size, table_count: catalogText.trim().split(/\r?\n/).length, years: [...years].sort(), scope_counts: scopeCounts, companies: companyRows });
    })();
  }
  return localSummary;
}

export async function GET(request: Request) {
  const upstream = process.env.KINGPRO_API_URL ?? "http://127.0.0.1:8080";
  try {
    const response = await fetch(`${upstream}/catalog`, {
      headers: backendHeaders(request),
      cache: "no-store",
      signal: AbortSignal.timeout(15_000),
    });
    if (response.ok) return NextResponse.json(repairTextDeep(await response.json()), { status: response.status });
    return NextResponse.json(await readLocalCatalog());
  } catch {
    try {
      return NextResponse.json(await readLocalCatalog());
    } catch {
    return NextResponse.json(
      { status: "offline", companies: [], years: [], scope_counts: {} },
      { status: 503 },
    );
    }
  }
}
