---
version: 1.0
name: R2AI
description: >
  Design system cho R2AI — AI Financial Data Assistant. Light-first, canvas trắng
  với một "voltage" duy nhất là brand azure #2F7DF6, gradient azure→cyan dùng rất
  tiết chế. Kế thừa triết lý restraint của Airbnb (một accent, nhiều whitespace, bo
  góc mềm, type-weight vừa phải, để dữ liệu và một chút illustration gánh visual)
  nhưng chuyển sang tông xanh tài chính đáng tin. Số liệu và code Pandas dùng Geist
  Mono. Mục tiêu: sạch, thân thiện với người không rành kỹ thuật, KHÔNG có cảm giác
  "AI công nghiệp" (không glassmorphism phủ khắp, không neon-on-black, không gradient
  loạn xạ).

principles:
  - "Một accent duy nhất: brand azure #2F7DF6 mang mọi CTA chính, link, trạng thái active. Không thêm accent thứ hai."
  - "Light-first: canvas trắng #ffffff, các surface bước qua vài sắc xanh rất nhạt. Không dark mode ở MVP."
  - "Gradient azure→cyan chỉ dùng có chủ đích: overlay video login, viền/nền nhấn của KPI card, đôi khi nút primary. Không phủ tràn lan."
  - "Type-weight vừa phải (500-700), heading không quá to. Số liệu tài chính và code Pandas dùng Geist Mono để tạo điểm nhấn 'chính xác'."
  - "Bo góc mềm (base 12px), một tầng shadow nhẹ duy nhất. Chiều sâu đến từ surface-on-surface và hairline, không phải đổ bóng nhiều tầng."
  - "Semantic tài chính rõ ràng: tăng = xanh lá, giảm = đỏ, cảnh báo = cam. Tách bạch khỏi brand azure."

colors:
  # Bộ màu đã làm ĐẬM HƠN (deeper/more saturated) — light-first vẫn giữ
  primary: "#1C64E8"
  primary-hover: "#1550C4"
  primary-active: "#1145AE"
  primary-soft: "#DBE8FE"
  primary-disabled: "#A9C6F8"
  accent-cyan: "#0FA6C9"
  gradient-from: "#1C64E8"
  gradient-to: "#10B4D6"
  ink: "#0B1B2E"
  body: "#263849"
  muted: "#52637A"
  muted-soft: "#8394A6"
  canvas: "#FFFFFF"
  surface-soft: "#EEF4FC"
  surface-strong: "#DFEAFB"
  hairline: "#DBE5F0"
  hairline-soft: "#EAF0F8"
  border-strong: "#B9C7D8"
  on-primary: "#FFFFFF"
  on-dark: "#FFFFFF"
  positive: "#0E9E5C"
  positive-soft: "#DCF5E8"
  negative: "#E23A2E"
  negative-soft: "#FCE6E4"
  warning: "#E07C00"
  warning-soft: "#FDEFDC"
  scrim: "#0B1B2E"

typography:
  fontSans: "var(--font-geist-sans), Inter, -apple-system, system-ui, 'Segoe UI', Roboto, sans-serif"
  fontMono: "var(--font-geist-mono), 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace"
  # Đã tăng +4px mỗi bậc theo yêu cầu
  display-xl: { size: "36px", weight: 700, lineHeight: 1.2, tracking: "-0.5px" }   # hero login
  display-lg: { size: "28px", weight: 700, lineHeight: 1.25, tracking: "-0.3px" }  # tiêu đề trang
  title-md:   { size: "22px", weight: 600, lineHeight: 1.3 }                       # tiêu đề card / panel
  title-sm:   { size: "19px", weight: 600, lineHeight: 1.35 }                      # nhãn nhóm
  body-md:    { size: "19px", weight: 400, lineHeight: 1.55 }                      # câu trả lời, đoạn văn
  body-sm:    { size: "17px", weight: 400, lineHeight: 1.5 }                       # meta, phụ đề
  caption:    { size: "16px", weight: 500, lineHeight: 1.4 }                       # nhãn field, chip
  kpi-number: { size: "32px", weight: 700, lineHeight: 1.1, family: mono, tracking: "-0.5px" }  # số KPI lớn
  mono-code:  { size: "17px", weight: 400, lineHeight: 1.5, family: mono }         # code Pandas, ticker, mã số
  button:     { size: "19px", weight: 600, lineHeight: 1.2 }

rounded:
  none: "0px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
  full: "9999px"

spacing:
  xxs: "2px"
  xs: "4px"
  sm: "8px"
  md: "12px"
  base: "16px"
  lg: "24px"
  xl: "32px"
  xxl: "48px"

elevation:
  flat: "none"
  card: "0 1px 2px rgba(16,36,59,0.04), 0 2px 8px rgba(16,36,59,0.06)"
  raised: "0 4px 16px rgba(16,36,59,0.10)"
  focus-ring: "0 0 0 3px rgba(47,125,246,0.25)"
---

## Overview

R2AI là một trợ lý phân tích **báo cáo tài chính** cho người dùng bán chuyên (biết một
chút tài chính, không biết lập trình). Thẩm mỹ hướng tới cảm giác **một fintech đáng
tin nhưng thân thiện**: nền trắng thoáng, một sắc xanh azure làm điểm nhấn, số liệu
được tôn lên bằng font mono, và một chút hình minh hoạ brand (nhân vật R2AI, video
nền login) tạo sự ấm áp thay vì lạnh lùng kiểu "AI công nghiệp".

**Nguyên tắc xuyên suốt:** restraint. Một accent (azure `{colors.primary}`), một tầng
shadow, bo góc mềm 12px, whitespace rộng. Gradient azure→cyan là "gia vị" chỉ xuất
hiện ở vài điểm neo thị giác (overlay video login, nền nhấn KPI, đôi khi nút primary),
không bao giờ phủ toàn trang.

## Colors

**Brand.** `{colors.primary}` (#2F7DF6 — azure) là voltage duy nhất: nút CTA chính,
link, tab active, viền focus, con trỏ đang gõ. `{colors.accent-cyan}` (#19B8D8) chỉ
là điểm cuối gradient, không dùng độc lập như accent thứ hai. Gradient thương hiệu:
`linear-gradient(135deg, {colors.gradient-from}, {colors.gradient-to})`.

**Surface (light-first).** Canvas trắng `{colors.canvas}`. Panel/nền phụ bước qua
`{colors.surface-soft}` (#F4F8FD, xanh cực nhạt) và `{colors.surface-strong}`
(#E9F1FB). Card nền trắng, tách khỏi nền bằng hairline `{colors.hairline}` + shadow
`{elevation.card}`.

**Text.** `{colors.ink}` (#10243B — navy sâu, không đen tuyền) cho heading; `{colors.body}`
cho đoạn văn/câu trả lời; `{colors.muted}` cho phụ đề, meta; `{colors.muted-soft}` cho
placeholder/disabled.

**Semantic tài chính (tách khỏi brand).** Tăng trưởng/dương: `{colors.positive}` (xanh
lá) trên nền `{colors.positive-soft}`. Giảm/âm: `{colors.negative}` (đỏ) trên
`{colors.negative-soft}`. Cảnh báo/dữ liệu bẩn: `{colors.warning}` (cam). Ba màu này
chỉ dùng cho biến động số liệu và badge trạng thái, không lẫn với azure.

## Typography

Hệ chữ chính là **Geist Sans** (đã có sẵn trong Next.js), fallback Inter. Heading giữ
weight 600-700 nhưng cỡ vừa phải (24-32px) — để dữ liệu và card gánh hierarchy, không
để chữ hò hét. **Geist Mono** là điểm nhấn đặc trưng của R2AI: mọi **con số tài chính**,
**mã cổ phiếu (ticker)**, **Mã số BCTC**, và **code Pandas** đều dùng mono — vừa gợi
cảm giác "chính xác, kiểm toán được", vừa tạo tương phản với sans của phần diễn giải.

Điểm typographically "to nhất" trong hệ thống là **KPI number** (`{typography.kpi-number}`
— 28px mono/700): con số là tín hiệu tin cậy cao nhất với người dùng tài chính, nên
được xử lý nổi bật nhất.

## Layout

- **Login:** split 2 cột (`lg:grid-cols-2`). Trái: form đăng nhập canh giữa trên canvas
  trắng, tối đa ~400px. Phải: **video nền** full-bleed (`object-cover`) phủ gradient
  overlay tối nhẹ để chữ branding đè lên đọc được; ẩn trên mobile (`hidden lg:block`).
- **App chính:** split 2 cột resizable. Trái (~45-52%): hội thoại — danh sách message
  cuộn + composer có nút đính kèm file. Phải: panel dữ liệu — hàng **KPI cards** trên
  cùng, dưới là **bảng preview** sheet đang chọn + **chart trend**. Mobile: chuyển
  thành Tabs "Trò chuyện | Dữ liệu".
- **Whitespace:** padding panel `{spacing.lg}` (24px), gap giữa card `{spacing.base}`.
  Composer và KPI row bám mép, nội dung thở.

## Elevation

Một tầng shadow duy nhất `{elevation.card}` cho card/KPI/dropdown; `{elevation.raised}`
chỉ cho popover/modal. Focus dùng ring azure `{elevation.focus-ring}` (không glow, không
đổi màu nền). 95% bề mặt là phẳng.

## Components

### Buttons
- **button-primary:** nền `{colors.primary}`, chữ trắng, radius `{rounded.sm}` (8px),
  cao 44px, weight 600. Hover → `{colors.primary-hover}`. Disabled → `{colors.primary-disabled}`.
  Biến thể nhấn (login CTA) có thể dùng nền gradient brand.
- **button-secondary:** nền trắng, chữ `{colors.ink}`, viền 1px `{colors.border-strong}`,
  radius 8px. Cho hành động phụ.
- **button-ghost:** trong suốt, chữ `{colors.muted}`, hover nền `{colors.surface-soft}`.

### Inputs
- **text-input:** nền trắng, viền 1px `{colors.hairline}`, radius `{rounded.sm}`, cao
  44px, padding 12px. Nhãn phía trên dạng `{typography.caption}` muted. Focus: viền dày
  2px `{colors.primary}` + ring `{elevation.focus-ring}`.

### Chat
- **message-user:** bong bóng nền `{colors.primary}` chữ trắng, bo `{rounded.lg}`, canh
  phải, tối đa ~80% chiều rộng.
- **message-assistant:** nền trắng (hoặc `{colors.surface-soft}`), chữ `{colors.body}`,
  bo `{rounded.lg}`, canh trái. Bên trong có thể chứa: đoạn diễn giải (sans), **khối số
  liệu** (mono, lớn), **khối code Pandas** (mono trên nền `{colors.surface-strong}` bo
  8px), và **citation** dạng chip nhỏ.
- **file-chip:** thẻ file đính kèm — icon bảng tính, tên file (`{typography.title-sm}`),
  meta "N sheet · K KB" (`{typography.body-sm}` muted), dropdown chọn sheet active. Nền
  trắng, viền hairline, radius `{rounded.md}`. Hiện cả trong composer và trong bong bóng.
- **suggestion-chip:** pill (`{rounded.full}`) nền `{colors.primary-soft}` chữ
  `{colors.primary}`, câu hỏi gợi ý ngắn. Hover đậm hơn.

### Data
- **kpi-card:** nền trắng, radius `{rounded.md}`, shadow card, padding `{spacing.base}`.
  Trên: nhãn chỉ số (`{typography.caption}` muted). Giữa: **số lớn** (`{typography.kpi-number}`
  mono). Dưới: delta YoY dạng badge (mũi tên + %, màu positive/negative). Một dải viền
  trái mỏng gradient brand để nhấn.
- **data-table:** header nền `{colors.surface-soft}`, chữ `{typography.caption}` muted
  uppercase nhẹ; cell `{typography.body-sm}`, số căn phải dùng mono; hàng phân tách bằng
  hairline; hàng hover `{colors.surface-soft}`. Cắt preview ~50 dòng, có chỉ báo "còn N
  dòng".
- **citation-tag:** chip nhỏ `{typography.caption}` nền `{colors.surface-strong}` chữ
  `{colors.muted}`, dạng "Nguồn: Mã số 60 · 2024". Bấm được để nhảy tới ô trong bảng.
- **cleaning-badge:** badge cam `{colors.warning}` khi phát hiện dữ liệu bẩn (VND/USD
  dính, ngoặc âm, khoảng trắng) — báo cho người dùng biết đã tự làm sạch.

### Responsive
| Breakpoint | Hành vi |
|---|---|
| Mobile <768px | Login ẩn video; app chuyển 2 cột thành Tabs "Trò chuyện \| Dữ liệu"; KPI cuộn ngang. |
| Tablet 768-1128px | App 2 cột, panel phải hẹp lại; KPI 2 cột. |
| Desktop >1128px | Đầy đủ 2 cột resizable; KPI 4 cột. |

## Known Gaps / để sau MVP
- Dark mode (hiện light-only).
- Hover micro-interactions chi tiết cho từng card.
- Trạng thái loading skeleton cho bảng lớn (dùng Skeleton xanh nhạt tạm thời).
