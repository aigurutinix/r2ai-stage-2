/**
 * Bản đồ mã cổ phiếu ↔ tên doanh nghiệp (seed từ các mã HOSE phổ biến trong benchmark).
 * Dùng cho Query Understanding + Table Retrieval (nhận diện công ty trong câu hỏi tiếng Việt).
 * Khi có data thật của BTC, bổ sung/nạp thêm từ metadata kho.
 */
export interface TickerInfo {
  ticker: string;
  companyName: string;
  aliases: string[];
}

export const TICKERS: TickerInfo[] = [
  { ticker: "HPG", companyName: "Tập đoàn Hòa Phát", aliases: ["hòa phát", "tập đoàn hòa phát"] },
  { ticker: "HSG", companyName: "Tập đoàn Hoa Sen", aliases: ["hoa sen", "tôn hoa sen"] },
  { ticker: "NKG", companyName: "Thép Nam Kim", aliases: ["nam kim", "thép nam kim"] },
  { ticker: "TLH", companyName: "Thép Tiến Lên", aliases: ["tiến lên", "thép tiến lên"] },
  { ticker: "SMC", companyName: "Đầu tư Thương mại SMC", aliases: ["đầu tư thương mại smc"] },
  { ticker: "VCB", companyName: "Vietcombank", aliases: ["vietcombank", "ngoại thương"] },
  { ticker: "TCB", companyName: "Techcombank", aliases: ["techcombank", "kỹ thương"] },
  { ticker: "MBB", companyName: "MBBank", aliases: ["mbbank", "quân đội", "ngân hàng quân đội"] },
  { ticker: "ACB", companyName: "Ngân hàng Á Châu", aliases: ["á châu", "ngân hàng á châu"] },
  { ticker: "VPB", companyName: "VPBank", aliases: ["vpbank", "việt nam thịnh vượng"] },
  { ticker: "BID", companyName: "BIDV", aliases: ["bidv", "đầu tư và phát triển"] },
  { ticker: "CTG", companyName: "VietinBank", aliases: ["vietinbank", "công thương"] },
  { ticker: "STB", companyName: "Sacombank", aliases: ["sacombank", "sài gòn thương tín"] },
  { ticker: "HDB", companyName: "HDBank", aliases: ["hdbank"] },
  { ticker: "VIB", companyName: "Ngân hàng Quốc Tế", aliases: ["quốc tế", "ngân hàng quốc tế"] },
  { ticker: "VNM", companyName: "Vinamilk", aliases: ["vinamilk", "sữa việt nam"] },
  { ticker: "MWG", companyName: "Thế Giới Di Động", aliases: ["thế giới di động", "tgdđ"] },
  { ticker: "MSN", companyName: "Tập đoàn Masan", aliases: ["masan", "tập đoàn masan"] },
  { ticker: "PNJ", companyName: "Vàng bạc Đá quý Phú Nhuận", aliases: ["phú nhuận", "vàng bạc đá quý phú nhuận"] },
  { ticker: "FRT", companyName: "FPT Retail", aliases: ["fpt retail", "fpt shop"] },
  { ticker: "SAB", companyName: "Sabeco", aliases: ["sabeco", "bia sài gòn"] },
  { ticker: "BHN", companyName: "Habeco", aliases: ["habeco", "bia hà nội"] },
  { ticker: "KDC", companyName: "KIDO Group", aliases: ["kido", "kinh đô"] },
  { ticker: "FPT", companyName: "Tập đoàn FPT", aliases: ["fpt", "tập đoàn fpt"] },
  { ticker: "CTR", companyName: "Viettel Construction", aliases: ["viettel construction", "công trình viettel"] },
  { ticker: "ELC", companyName: "Elcom", aliases: ["elcom"] },
  { ticker: "CMG", companyName: "CMC Corporation", aliases: ["cmc", "cmc corporation"] },
  { ticker: "VHM", companyName: "Vinhomes", aliases: ["vinhomes"] },
  { ticker: "VIC", companyName: "Vingroup", aliases: ["vingroup"] },
  { ticker: "VRE", companyName: "Vincom Retail", aliases: ["vincom retail", "vincom"] },
  { ticker: "NVL", companyName: "No Va Land", aliases: ["novaland", "no va land"] },
  { ticker: "KDH", companyName: "Nhà Khang Điền", aliases: ["khang điền", "nhà khang điền"] },
  { ticker: "NLG", companyName: "Nam Long", aliases: ["nam long"] },
  { ticker: "DIG", companyName: "DIC Corp", aliases: ["dic corp", "dic"] },
  { ticker: "PDR", companyName: "Phát Đạt", aliases: ["phát đạt"] },
  { ticker: "DXG", companyName: "Đất Xanh Group", aliases: ["đất xanh", "đất xanh group"] },
  { ticker: "CEO", companyName: "CEO Group", aliases: ["ceo group"] },
  { ticker: "GAS", companyName: "PV GAS", aliases: ["pv gas", "khí việt nam"] },
  { ticker: "PVD", companyName: "PV Drilling", aliases: ["pv drilling", "khoan dầu khí"] },
  { ticker: "PVS", companyName: "PTSC", aliases: ["ptsc", "dịch vụ kỹ thuật dầu khí"] },
  { ticker: "POW", companyName: "PV Power", aliases: ["pv power", "điện lực dầu khí"] },
  { ticker: "REE", companyName: "Cơ điện REE", aliases: ["cơ điện lạnh", "cơ điện ree"] },
  { ticker: "GEG", companyName: "Điện Gia Lai", aliases: ["điện gia lai"] },
  { ticker: "GMD", companyName: "Gemadept", aliases: ["gemadept"] },
  { ticker: "HAH", companyName: "Vận tải Hai An", aliases: ["hải an", "vận tải hải an"] },
  { ticker: "VSC", companyName: "Container Việt Nam", aliases: ["container việt nam", "viconship"] },
  { ticker: "DHG", companyName: "Dược Hậu Giang", aliases: ["dược hậu giang", "hậu giang"] },
  { ticker: "IMP", companyName: "Dược Imexpharm", aliases: ["imexpharm", "dược imexpharm"] },
  { ticker: "DBD", companyName: "Dược Bình Định", aliases: ["dược bình định", "bidiphar"] },
];

const BY_TICKER = new Map(TICKERS.map((t) => [t.ticker, t]));
export function tickerInfo(code: string): TickerInfo | undefined {
  return BY_TICKER.get(code.toUpperCase());
}

/** Bỏ dấu tiếng Việt + lowercase (đ→d). */
export function stripVN(s: string): string {
  return s
    .toLowerCase()
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/đ/g, "d");
}

/** Dò các mã CK xuất hiện trong câu hỏi (qua mã hoặc tên/alias). */
export function resolveTickers(text: string): string[] {
  const t = " " + stripVN(text) + " ";
  const found = new Set<string>();
  for (const info of TICKERS) {
    const code = info.ticker.toLowerCase();
    if (new RegExp(`(?:^|[^a-z0-9])${code}(?:[^a-z0-9]|$)`).test(t)) {
      found.add(info.ticker);
      continue;
    }
    for (const a of [info.companyName, ...info.aliases]) {
      const na = stripVN(a);
      if (na.length >= 4 && t.includes(na)) {
        found.add(info.ticker);
        break;
      }
    }
  }
  return [...found];
}
