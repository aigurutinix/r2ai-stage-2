(() => {
  'use strict';

  const STORAGE_KEY = 'vifinqa.sessions.v1';
  const STAGE_ORDER = ['metadata', 'plan', 'retrieve', 'rerank', 'answer'];
  const $ = (selector, root = document) => root.querySelector(selector);
  const elements = {
    body: document.body,
    sidebar: $('#sidebar'),
    sessionList: $('#sessionList'),
    sessionCount: $('#sessionCount'),
    newChat: $('#newChat'),
    menuButton: $('#menuButton'),
    serviceState: $('#serviceState'),
    serviceLabel: $('#serviceLabel'),
    fixtureBanner: $('#fixtureBanner'),
    panes: $('#panes'),
    messages: $('#messages'),
    evidenceBody: $('#evidenceBody'),
    evidenceHeading: $('#evidenceHeading'),
    evidenceCount: $('#evidenceCount'),
    evidenceBack: $('#evidenceBack'),
    answerPane: $('#answerPane'),
    panesResizer: $('#paneResizer'),
    railToggle: $('#railToggle'),
    evidencePane: $('#evidencePane'),
    form: $('#promptForm'),
    input: $('#promptInput'),
    sendButton: $('#sendButton'),
    stopButton: $('#stopButton'),
    traceHost: null,
  };

  const state = {
    sessions: loadSessions(),
    activeId: null,
    controller: null,
    currentRequestId: null,
    trace: [],
    stageOpen: {},
    sources: new Map(),
    previewsInFlight: new Set(),
    savedTables: new Map(),
    confirmDelete: null,
    cards: [],
    openTable: null,
    elapsedTimer: null,
    startedAt: null,
    stickToBottom: false,
  };
  state.activeId = state.sessions[0]?.id || createSession().id;

  function createId() {
    return globalThis.crypto?.randomUUID?.() || `chat-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function createSession() {
    const session = { id: createId(), title: 'Cuộc trò chuyện mới', messages: [] };
    state.sessions.unshift(session);
    return session;
  }

  function activeSession() {
    return state.sessions.find((session) => session.id === state.activeId);
  }

  function loadSessions() {
    try {
      const sessions = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
      return Array.isArray(sessions) ? sessions.filter((session) => session && session.id && Array.isArray(session.messages)) : [];
    } catch (_) {
      return [];
    }
  }

  /* Saved evidence is the largest thing in storage, so a full quota drops it
     from the oldest sessions first rather than silently losing the write. */
  function saveSessions() {
    const sessions = state.sessions.slice(0, 30);
    for (let dropped = 0; dropped <= sessions.length; dropped += 1) {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
        return;
      } catch (_) {
        const victim = sessions[sessions.length - 1 - dropped];
        victim?.messages?.forEach((message) => { delete message.evidence; });
        if (!victim) return;
      }
    }
  }

  function renderSessions() {
    elements.sessionList.replaceChildren();
    elements.sessionCount.textContent = String(state.sessions.length);
    state.sessions.slice(0, 12).forEach((session) => {
      const title = session.title || 'Cuộc trò chuyện mới';
      const row = document.createElement('div');
      row.className = `session-row${session.id === state.activeId ? ' is-active' : ''}`;

      const open = document.createElement('button');
      open.type = 'button';
      open.className = 'session-item';
      open.textContent = title;
      open.title = title;
      open.addEventListener('click', () => {
        if (state.controller) stopRequest();
        state.activeId = session.id;
        restoreEvidence(session);
        render();
        closeMenu();
      });
      row.append(open);

      if (state.confirmDelete === session.id) {
        // Confirmation happens in the row itself. A destructive action deserves
        // a second step, but not a modal that interrupts the whole screen.
        row.classList.add('is-confirming');
        const yes = document.createElement('button');
        yes.type = 'button';
        yes.className = 'session-confirm';
        yes.textContent = 'Xoá';
        yes.setAttribute('aria-label', `Xác nhận xoá "${title}"`);
        yes.addEventListener('click', (event) => { event.stopPropagation(); deleteSession(session.id); });
        const no = document.createElement('button');
        no.type = 'button';
        no.className = 'session-cancel';
        no.textContent = 'Huỷ';
        no.setAttribute('aria-label', 'Huỷ xoá');
        no.addEventListener('click', (event) => { event.stopPropagation(); state.confirmDelete = null; renderSessions(); });
        row.append(yes, no);
      } else {
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'session-delete';
        remove.setAttribute('aria-label', `Xoá "${title}"`);
        remove.append(icon('trash', 'icon'));
        remove.addEventListener('click', (event) => {
          event.stopPropagation();
          state.confirmDelete = session.id;
          renderSessions();
        });
        row.append(remove);
      }
      elements.sessionList.append(row);
    });
  }

  function deleteSession(id) {
    const wasActive = id === state.activeId;
    if (wasActive && state.controller) stopRequest();
    state.sessions = state.sessions.filter((session) => session.id !== id);
    state.confirmDelete = null;
    if (wasActive) {
      // Never leave the workspace pointing at a session that no longer exists.
      const next = state.sessions[0] || createSession();
      state.activeId = next.id;
      state.trace = [];
      state.stageOpen = {};
      state.currentRequestId = null;
      restoreEvidence(next);
    }
    saveSessions();
    render();
  }

  function render() {
    renderSessions();
    renderMessages();
  }

  const STAGE_META = {
    metadata: { index: '01', label: 'Xác định công ty' },
    plan:     { index: '02', label: 'Lập kế hoạch' },
    retrieve: { index: '03', label: 'Truy xuất bảng' },
    rerank:   { index: '04', label: 'Xếp hạng & chọn' },
    answer:   { index: '05', label: 'Tính toán' },
    pipeline: { index: '--', label: 'Pipeline' },
    input:    { index: '--', label: 'Đầu vào' },
  };
  const STATUS_TEXT = { running: 'đang chạy', done: 'xong', complete: 'xong', degraded: 'một phần', error: 'lỗi', failed: 'lỗi' };

  const previewCache = new Map();

  function renderWelcome() {
    const welcome = document.createElement('section');
    welcome.className = 'welcome';
    const heading = document.createElement('h2');
    heading.textContent = 'Tìm câu trả lời trong báo cáo tài chính';
    const copy = document.createElement('p');
    copy.textContent = 'Đặt câu hỏi bằng tiếng Việt. VIFin truy xuất bảng dữ liệu gốc, xếp hạng bằng chứng và hiển thị phép tính cùng ô nguồn để bạn kiểm chứng.';
    const chips = document.createElement('div');
    chips.className = 'example-chips';
    [
      'Doanh thu của FPT năm 2023 là bao nhiêu?',
      'So sánh lợi nhuận sau thuế 2022 và 2023',
      'Tổng tài sản của công ty thay đổi thế nào?',
    ].forEach((question) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'example-chip';
      button.dataset.question = question;
      button.textContent = question;
      chips.append(button);
    });
    welcome.append(heading, copy, chips);
    elements.messages.append(welcome);
  }

  function renderMessages() {
    const el = elements.messages;
    const atBottom = state.stickToBottom || el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    el.replaceChildren();
    const session = activeSession();
    if (!state.controller) {
      const saved = [...(session?.messages || [])].reverse().find((message) => Array.isArray(message.trace));
      state.trace = saved?.trace || [];
    }
    if (!session?.messages.length) {
      renderWelcome();
      elements.traceHost = null;
      renderTrace([]);
      return;
    }
    // A workspace shows the exchange being worked on, not a scrollback: the
    // question as a command line, then the answer, then the trace beneath it.
    session.messages.forEach((message, index) => {
      const node = messageElement(message);
      node.dataset.messageIndex = String(index);
      el.append(node);
    });
    const host = document.createElement('div');
    el.append(host);
    elements.traceHost = host;
    if (atBottom) el.scrollTop = el.scrollHeight;
    state.stickToBottom = false;
    renderTrace(state.trace);
  }

  /* ── source identity ─────────────────────────────────────────────────
     A table_id is `report_id|line`. SOURCE:: numbers are assigned in the
     order the ranked selection returns them, so the number on a citation and
     the number on its card are the same object. LINE:: is the table's start
     line in the source document — deliberately not called PAGE, because the
     corpus does not carry page numbers and labelling it one would be a claim
     the data cannot support. */
  function sourceParts(tableId) {
    const [reportId, line] = String(tableId || '').split('|');
    const match = /^([A-Z0-9]+)_financial_statements_(\d{4})(?:_(\w+?))?(?:_\d+)?$/i.exec(reportId || '');
    return {
      reportId: reportId || '',
      line: line || '',
      ticker: match ? match[1].toUpperCase() : (reportId || '').split('_')[0].toUpperCase(),
      year: match ? match[2] : '',
      scope: match && match[3] ? match[3].toUpperCase() : '',
    };
  }

  function sourceNumber(tableId) {
    if (!tableId) return null;
    if (!state.sources.has(tableId)) state.sources.set(tableId, String(state.sources.size + 1).padStart(2, '0'));
    return state.sources.get(tableId);
  }

  function registerSources(cards) {
    (cards || []).forEach((card) => sourceNumber(card.table_id));
  }

  function messageElement(message) {
    const article = document.createElement('article');
    article.className = `message ${message.role}${message.pending ? ' pending' : ''}${message.error ? ' error' : ''}`;

    if (message.role === 'user') {
      const line = document.createElement('div');
      line.className = 'query-line';
      const caret = document.createElement('span');
      caret.className = 'query-caret';
      caret.setAttribute('aria-hidden', 'true');
      caret.textContent = '>';
      const text = document.createElement('p');
      text.className = 'query-text';
      text.textContent = message.content || '';
      line.append(caret, text);
      article.append(line);
      return article;
    }

    const body = document.createElement('div');
    body.className = 'message-body';

    if (message.pending || message.error || message.stopped || !message.content) {
      const text = document.createElement('p');
      text.className = 'message-text';
      text.textContent = message.content || (message.pending ? 'Đang xử lý…' : '');
      body.append(text);
      article.append(body);
      return article;
    }

    if (message.question) {
      const head = document.createElement('p');
      head.className = 'answer-head';
      head.textContent = message.question;
      body.append(head);
    }

    // Doto is a Latin-only dot-matrix face reserved for figures. A status line
    // like "Đã dừng yêu cầu." is Vietnamese prose and must never land in it —
    // the diacritics are not in the subset and render as broken fallbacks.
    if (isFigure(String(message.content).trim())) {
      const value = document.createElement('div');
      value.className = 'answer-value';
      const number = document.createElement('span');
      number.textContent = message.content;
      value.append(number);
      const unit = questionUnit(message.question || '');
      if (unit) {
        const unitEl = document.createElement('span');
        unitEl.className = 'answer-unit';
        unitEl.textContent = unit;
        value.append(unitEl);
      }
      body.append(value);
    } else {
      const text = document.createElement('p');
      text.className = 'message-text';
      text.textContent = message.content;
      body.append(text);
    }

    const refs = document.createElement('div');
    refs.className = 'answer-refs';
    const seen = new Set();
    (message.citations || []).forEach((citation, index) => {
      const number = sourceNumber(citation.table_id);
      const key = number || citation.id;
      if (seen.has(key)) return;
      seen.add(key);
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'source-ref';
      button.dataset.citationIndex = String(index);
      button.dataset.requestId = message.requestId || '';
      if (citation.table_id) button.dataset.tableId = citation.table_id;
      button.textContent = number ? `Nguồn ${number}` : `Nguồn ${index + 1}`;
      refs.append(button);
    });
    if (message.checks && message.checks.executed) {
      const tag = document.createElement('span');
      tag.className = 'verified-tag';
      tag.textContent = 'Đã kiểm chứng';
      refs.append(tag);
    }
    if (refs.childElementCount) body.append(refs);

    if (message.code || Object.keys(message.checks || {}).length) {
      const calculation = document.createElement('button');
      calculation.type = 'button';
      calculation.className = 'calculation-button';
      calculation.dataset.calculation = 'true';
      calculation.textContent = 'Xem phép tính';
      body.append(calculation);
    }
    article.append(body);
    return article;
  }

  function questionUnit(question) {
    const text = question.toLowerCase();
    if (text.includes('nghìn tỷ')) return 'NGHÌN TỶ VND';
    if (text.includes('tỷ')) return 'TỶ VND';
    if (text.includes('triệu')) return 'TRIỆU VND';
    if (text.includes('nghìn')) return 'NGHÌN VND';
    if (text.includes('%') || text.includes('phần trăm') || text.includes('tỷ lệ')) return '%';
    return '';
  }

  function icon(name, className = 'icon') {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', className);
    svg.setAttribute('aria-hidden', 'true');
    const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
    use.setAttribute('href', `#i-${name}`);
    svg.append(use);
    return svg;
  }

  /* ── evidence gallery ─────────────────────────────────────────────── */
  function evidenceCard(card, options = {}) {
    const parts = sourceParts(card.table_id);
    const number = sourceNumber(card.table_id);
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'evidence-card';
    button.dataset.tableId = card.table_id;
    // The full title is boilerplate on most tables, so it lives here rather
    // than taking the card's most valuable line.
    button.title = card.title || card.table_id;

    const head = document.createElement('div');
    head.className = 'card-head';
    const id = document.createElement('span');
    id.textContent = `Nguồn ${number}`;
    head.append(id);
    if (typeof card.score === 'number') {
      const rank = document.createElement('span');
      rank.className = 'card-rank';
      const fill = document.createElement('span');
      fill.style.width = `${Math.round(Math.max(0, Math.min(1, card.score)) * 100)}%`;
      rank.append(fill);
      const score = document.createElement('span');
      score.className = 'card-score';
      score.textContent = card.score.toFixed(2);
      head.append(rank, score);
    } else {
      head.append(document.createElement('span'));
    }
    button.append(head);

    const rows = document.createElement('div');
    rows.className = 'card-rows is-empty';
    rows.textContent = '…';
    button.append(rows);
    loadPreview(card.table_id, card.requestId || state.currentRequestId, rows);

    const meta = document.createElement('div');
    meta.className = 'card-meta';
    const lead = document.createElement('b');
    lead.textContent = [parts.ticker, parts.year].filter(Boolean).join(' ');
    meta.append(lead);
    button.append(meta);
    if (options.selected) button.classList.add('is-chosen');
    return button;
  }

  /* The server keys tables by a request id it forgets on restart, so a finished
     answer carries its own evidence. Only the selected tables are kept, and
     their rows are trimmed, to stay inside the localStorage budget. */
  const SAVED_ROWS = 40;

  function snapshotEvidence() {
    const cards = state.cards.filter((card) => card.selected);
    const tables = {};
    cards.forEach((card) => {
      const payload = previewCache.get(`${card.requestId || state.currentRequestId}::${card.table_id}`);
      if (!payload) return;
      tables[card.table_id] = {
        ...payload,
        rows: (payload.rows || []).slice(0, SAVED_ROWS),
        truncated: Boolean(payload.truncated) || (payload.rows || []).length > SAVED_ROWS,
      };
    });
    return { cards: cards.map(({ requestId, ...rest }) => rest), tables };
  }

  function restoreEvidence(session) {
    const saved = [...(session?.messages || [])].reverse().find((message) => message.evidence)?.evidence;
    state.sources.clear();
    state.savedTables = new Map(Object.entries(saved?.tables || {}));
    state.cards = (saved?.cards || []).map((card) => ({ ...card }));
    state.cards.forEach((card) => sourceNumber(card.table_id));
    state.openTable = null;
    renderEvidencePane();
  }

  /* Previews resolve into whatever card is on screen for that table id.
     Holding a node reference loses the write when a later stage re-renders the
     grid, which left every card stuck on its loading placeholder. */
  function paintPreview(tableId) {
    const payload = previewCache.get(previewKey(tableId)) || state.savedTables.get(tableId);
    if (!payload) return;
    document.querySelectorAll(`.evidence-card[data-table-id="${CSS.escape(tableId)}"] .card-rows`)
      .forEach((node) => fillRows(node, payload));
  }

  function previewKey(tableId) {
    return `${state.currentRequestId}::${tableId}`;
  }

  /* A lone figure: optional parentheses or minus, grouped digits, optional
     percent. OCR sometimes merges two columns into one cell, and those must not
     be shown as a value — a card that prints "44.009.528(37.214" is worse than
     one that prints nothing. */
  const FIGURE = /^[(\-]?\s*\d[\d.,\s]*\)?\s*%?$/;

  function isFigure(cell) {
    if (!FIGURE.test(cell)) return false;
    // A separator or four digits separates a financial figure from a "Mã số"
    // code like 01 or 310, which sit in their own column and are not values.
    return (cell.match(/\d/g) || []).length >= 4 || /[.,]/.test(cell);
  }

  function rowFigure(row) {
    const cells = [...row].slice(1).map((cell) => String(cell ?? '').trim()).filter(Boolean);
    return [...cells].reverse().find(isFigure) || '';
  }

  function fillRows(node, payload) {
    const labelled = (payload?.rows || []).filter((row) => String(row[0] || '').trim());
    // Rows carrying a real figure describe the table best; section headings and
    // merged-cell rows fall back only when there is nothing better to show.
    const withFigures = labelled.filter((row) => rowFigure(row));
    const rows = (withFigures.length ? withFigures : labelled).slice(0, 3);
    if (!rows.length) { node.textContent = 'không có dòng dữ liệu'; return; }
    node.classList.remove('is-empty');
    node.replaceChildren();
    rows.forEach((row) => {
      const line = document.createElement('div');
      line.className = 'card-row';
      const label = document.createElement('span');
      label.className = 'card-row-label';
      label.textContent = String(row[0]).trim();
      label.title = label.textContent;
      line.append(label);
      const figure = rowFigure(row);
      if (figure) {
        const value = document.createElement('span');
        value.className = 'card-row-value';
        value.textContent = figure;
        value.title = figure;
        line.append(value);
      }
      node.append(line);
    });
  }

  function loadPreview(tableId, requestId, node) {
    if (!tableId) return;
    const saved = state.savedTables.get(tableId);
    if (saved) { fillRows(node, saved); return; }
    if (!requestId) { node.textContent = 'bảng chưa được lưu'; return; }
    const key = `${requestId}::${tableId}`;
    if (previewCache.has(key)) { fillRows(node, previewCache.get(key)); return; }
    if (state.previewsInFlight.has(key)) return;
    state.previewsInFlight.add(key);
    fetch(`/api/table/${encodeURIComponent(requestId)}/${encodeURIComponent(tableId)}`, { headers: { Accept: 'application/json' } })
      .then((response) => (response.ok ? response.json() : null))
      .then((payload) => {
        state.previewsInFlight.delete(key);
        if (!payload) {
          document.querySelectorAll(`.evidence-card[data-table-id="${CSS.escape(tableId)}"] .card-rows`)
            .forEach((n) => { n.textContent = 'không đọc được bảng'; });
          return;
        }
        previewCache.set(key, payload);
        paintPreview(tableId);
      })
      .catch(() => {
        state.previewsInFlight.delete(key);
        document.querySelectorAll(`.evidence-card[data-table-id="${CSS.escape(tableId)}"] .card-rows`)
          .forEach((n) => { n.textContent = 'lỗi tải bảng'; });
      });
  }

  function statRow(entries) {
    const row = document.createElement('div');
    row.className = 'stat-row';
    entries.filter(Boolean).forEach(([label, value]) => {
      const stat = document.createElement('span');
      stat.className = 'stat';
      stat.append(document.createTextNode(`${label} `));
      const strong = document.createElement('b');
      strong.textContent = String(value);
      stat.append(strong);
      row.append(stat);
    });
    return row;
  }

  function detailLine(text) {
    const line = document.createElement('p');
    line.className = 'stage-detail';
    line.textContent = text;
    return line;
  }

  function renderStageBody(stage, data) {
    const body = document.createElement('div');
    body.className = 'stage-body';

    if (stage === 'metadata') {
      const facts = [
        (data.tickers || []).length && `tên công ty: ${(data.tickers || []).join(', ')}`,
        (data.years || []).length && `năm: ${(data.years || []).join(', ')}`,
      ].filter(Boolean);
      body.append(detailLine(facts.join(', ')));
      return body;
    }

    if (stage === 'plan') {
      // The plan is the question decomposed: facts are the values to look up,
      // steps the arithmetic over them. Both carry their id so a step's inputs
      // point at something the reader can already see above it.
      (data.facts || []).forEach((fact) => body.append(detailLine(
        `${fact.id} truy xuất: ${fact.query}${fact.period ? `, năm: ${fact.period}` : ''}`)));
      (data.steps || []).forEach((step) => body.append(detailLine(
        `${step.id} tính toán: ${step.op}(${(step.inputs || []).join(', ')}), ${step.instruction}`)));
      if (data.steps_rejected) body.append(detailLine('Giữ nguyên dữ kiện chính, không dùng các bước phụ.'));
      return body;
    }

    if (stage === 'retrieve') {
      const cards = data.candidate_tables || [];
      setCards(cards);
      body.append(statRow([
        ['Tài liệu', (data.docs || []).length],
        ['Ứng viên', data.candidate_count ?? cards.length],
        ['Ngân sách', data.table_budget],
        data.graph_added ? ['Liên kết', `+${data.graph_added}`] : null,
        data.dense_enabled ? ['Dense', `+${data.dense_added || 0}`] : null,
      ]));
      if (data.line_items?.length) body.append(detailLine(`chỉ tiêu: ${data.line_items.join(', ')}`));
      return body;
    }

    if (stage === 'rerank') {
      const cards = data.top_tables || [];
      registerSources(cards);
      setCards(cards, { selected: true });
      body.append(statRow([
        ['Đã chấm', data.scored],
        ['Ngân sách', data.budget],
        ['Đã chọn', data.selected],
        data.note_links ? ['Thuyết minh', data.note_links] : null,
      ]));
      return body;
    }

    return body;
  }

  function stageTone(status) {
    const value = String(status || '').toLowerCase();
    if (value === 'running') return 'running';
    if (value === 'error' || value === 'failed') return 'error';
    if (value === 'degraded') return 'degraded';
    return 'done';
  }

  function stageNode(item, isOpen) {
    const tone = stageTone(item.status);
    const meta = STAGE_META[item.stage] || { index: '--', label: item.stage || 'pipeline' };
    const wrapper = document.createElement('div');
    wrapper.className = `stage is-${tone}`;
    wrapper.dataset.stage = item.stage;

    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'stage-row';
    row.dataset.stageToggle = item.stage;
    row.setAttribute('aria-expanded', String(isOpen));

    const index = document.createElement('span');
    index.className = 'stage-index';
    index.textContent = meta.index;
    const name = document.createElement('span');
    name.className = 'stage-name';
    name.textContent = meta.label;
    const status = document.createElement('span');
    status.className = 'stage-status';
    status.textContent = tone === 'running'
      ? STATUS_TEXT.running
      : (STATUS_TEXT[String(item.status || '').toLowerCase()] || 'xong');
    row.append(index, name, status);
    wrapper.append(row);

    if (isOpen && tone !== 'running') wrapper.append(renderStageBody(item.stage, item.data || {}));
    return wrapper;
  }

  function renderTrace(trace) {
    const host = elements.traceHost;
    if (!host) return;
    host.replaceChildren();
    if (!trace.length) { host.hidden = true; return; }
    host.hidden = false;

    const panel = document.createElement('section');
    panel.className = 'trace';
    const head = document.createElement('div');
    head.className = 'trace-head';
    const label = document.createElement('span');
    const running = state.controller && !['error', 'failed'].includes(String(trace[trace.length - 1].status).toLowerCase());
    label.textContent = running ? 'Đang xử lý' : 'Quy trình';
    const meter = document.createElement('span');
    meter.className = 'trace-elapsed';
    meter.textContent = running ? elapsedText() : `${trace.length}/${STAGE_ORDER.length}`;
    head.append(label, meter);
    panel.append(head);

    const list = document.createElement('div');
    list.className = 'trace-list';
    const latest = trace[trace.length - 1].stage;
    trace.forEach((item) => {
      const isOpen = item.stage in state.stageOpen ? state.stageOpen[item.stage] : item.stage === latest;
      list.append(stageNode(item, isOpen));
    });
    if (running && trace.length < STAGE_ORDER.length) {
      const known = new Set(trace.map((item) => item.stage));
      const next = STAGE_ORDER.find((stage) => !known.has(stage));
      if (next) list.append(stageNode({ stage: next, status: 'running' }, false));
    }
    panel.append(list);
    host.append(panel);
  }

  function setTraceStage(event) {
    const stage = event.stage || event.data?.stage || 'pipeline';
    const next = {
      stage,
      status: event.status || event.data?.status || 'complete',
      message: event.message || event.data?.message || '',
      data: event.data || {},
    };
    const index = state.trace.findIndex((item) => item.stage === stage);
    if (index === -1) {
      state.trace.push(next);
      Object.keys(state.stageOpen).forEach((key) => { state.stageOpen[key] = false; });
      state.stageOpen[stage] = true;
    } else {
      state.trace[index] = { ...state.trace[index], ...next };
    }
    renderTrace(state.trace);
  }

  /* Liveness without blinking: the running stage counts elapsed seconds, which
     is information a presenter can use, not decoration. */
  function startElapsed() {
    stopElapsed();
    state.startedAt = Date.now();
    state.elapsedTimer = setInterval(paintElapsed, 1000);
    paintElapsed();
  }

  function stopElapsed() {
    if (state.elapsedTimer) clearInterval(state.elapsedTimer);
    state.elapsedTimer = null;
  }

  function elapsedText() {
    const seconds = Math.max(0, Math.round((Date.now() - (state.startedAt || Date.now())) / 1000));
    return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
  }

  function paintElapsed() {
    const node = document.querySelector('.trace-elapsed');
    if (node && state.controller) node.textContent = elapsedText();
  }

  function setLoading(loading) {
    elements.input.disabled = loading;
    elements.sendButton.disabled = loading;
    elements.stopButton.hidden = !loading;
  }

  async function submitQuestion(question) {
    question = question.trim();
    if (!question || state.controller) return;
    let session = activeSession();
    if (!session) { session = createSession(); state.activeId = session.id; }
    if (session.title === 'Cuộc trò chuyện mới') session.title = question.slice(0, 56);
    session.messages.push({ role: 'user', content: question });
    session.messages.push({ role: 'assistant', content: '', pending: true, citations: [], question });
    state.trace = [];
    state.stageOpen = {};
    state.sources.clear();
    state.cards = [];
    state.savedTables = new Map();
    state.openTable = null;
    renderEvidencePane();
    state.currentRequestId = null;
    elements.input.value = '';
    saveSessions();
    render();
    setLoading(true);
    state.stickToBottom = true;
    startElapsed();
    const pending = session.messages.at(-1);
    state.controller = new AbortController();
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/x-ndjson' },
        body: JSON.stringify({ question }),
        signal: state.controller.signal,
      });
      if (response.status === 409) throw new Error('Một câu hỏi khác đang chạy. Chờ nó xong hoặc bấm Dừng.');
      if (response.status === 422) throw new Error('Câu hỏi trống hoặc quá dài.');
      if (response.status === 503) throw new Error('Máy chủ mô hình chưa sẵn sàng.');
      if (!response.ok) throw new Error(`Máy chủ trả về lỗi ${response.status}.`);
      if (!response.body) throw new Error('API không trả về luồng dữ liệu');
      await readEventStream(response.body, (event) => handleEvent(event, pending));
      if (pending.pending) throw new Error('Luồng dữ liệu kết thúc trước khi có câu trả lời');
    } catch (error) {
      if (error.name === 'AbortError') {
        pending.pending = false;
        pending.stopped = true;
        pending.content = 'Đã dừng yêu cầu.';
      } else {
        pending.pending = false;
        pending.error = true;
        pending.content = error.message || 'Không thể kết nối tới máy chủ.';
      }
      saveSessions();
      render();
    } finally {
      state.controller = null;
      stopElapsed();
      setLoading(false);
      renderTrace(state.trace);
    }
  }

  async function readEventStream(body, onEvent) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() || '';
      lines.forEach((line) => parseEventLine(line, onEvent));
      if (done) break;
    }
    if (buffer.trim()) parseEventLine(buffer, onEvent);
  }

  function parseEventLine(line, onEvent) {
    if (!line.trim()) return;
    let event;
    // Only the parse belongs in this try. onEvent throws deliberately to surface
    // a pipeline error message, and wrapping it here reported every such failure
    // as malformed NDJSON.
    try {
      event = JSON.parse(line);
    } catch (_) {
      throw new Error('API trả về NDJSON không hợp lệ');
    }
    onEvent(event);
  }

  function handleEvent(event, pending) {
    if (event.request_id) state.currentRequestId = event.request_id;
    if (event.type === 'stage') {
      setTraceStage(event);
      return;
    }
    if (event.type === 'error') {
      throw new Error(event.message || event.data?.message || 'Pipeline gặp lỗi');
    }
    if (event.type !== 'result') return;
    const data = event.data || {};
    state.currentRequestId = event.request_id || data.request_id || state.currentRequestId;
    pending.pending = false;
    pending.content = typeof data.answer === 'string' ? data.answer : String(data.answer ?? '');
    pending.code = typeof data.code === 'string' ? data.code : '';
    pending.citations = Array.isArray(data.citations) ? data.citations.filter(Boolean) : [];
    pending.checks = data.checks && typeof data.checks === 'object' ? data.checks : {};
    pending.requestId = state.currentRequestId;
    pending.trace = state.trace;
    pending.evidence = snapshotEvidence();
    if (state.trace.length) renderTrace(state.trace);
    saveSessions();
    renderMessages();
  }

  function stopRequest() {
    if (state.controller) state.controller.abort();
  }

  function metaRow(key, value) {
    const row = document.createElement('div');
    row.className = 'meta-key';
    row.append(document.createTextNode(key));
    const strong = document.createElement('b');
    strong.textContent = String(value);
    row.append(strong);
    return row;
  }

  /* ── the evidence pane ────────────────────────────────────────────────
     Evidence is never hidden behind a drawer: the pane is always on screen.
     Opening a card swaps the grid for that table inside the same column, so
     the answer beside it never moves. */
  function renderEvidencePane() {
    const host = elements.evidenceBody;
    host.replaceChildren();
    elements.evidenceBack.hidden = !state.openTable;
    if (state.openTable) {
      elements.evidenceHeading.textContent = 'Bảng nguồn';
      elements.evidenceCount.textContent = '';
      host.append(state.openTable);
      return;
    }
    elements.evidenceHeading.textContent = 'Bằng chứng';
    elements.evidenceCount.textContent = state.cards.length ? String(state.cards.length) : '';
    if (!state.cards.length) {
      const empty = document.createElement('p');
      empty.className = 'evidence-empty';
      empty.textContent = 'Bảng dữ liệu được truy xuất sẽ hiện ở đây, kèm điểm xếp hạng và ô nguồn.';
      host.append(empty);
      return;
    }
    const grid = document.createElement('div');
    grid.className = 'evidence-grid';
    state.cards.forEach((card) => grid.append(evidenceCard(card, { selected: card.selected })));
    host.append(grid);
  }

  function setCards(cards, options = {}) {
    const byId = new Map(state.cards.map((card) => [card.table_id, card]));
    (cards || []).forEach((card) => {
      // The request id is bound per card, never read from a global at fetch
      // time: a later question must not make an earlier card fetch its table.
      byId.set(card.table_id, { ...byId.get(card.table_id), ...card, ...options, requestId: state.currentRequestId });
    });
    // Selected tables lead; the rest keep their arrival order, so a streaming
    // update never reshuffles cards the user is already looking at.
    state.cards = [...byId.values()].sort((a, b) => Number(!!b.selected) - Number(!!a.selected));
    renderEvidencePane();
  }

  function tableView({ title, status, error, meta, table, code, checks }) {
    const view = document.createElement('div');
    view.className = 'table-view';
    if (title) {
      const heading = document.createElement('div');
      heading.className = 'table-title';
      heading.textContent = title;
      view.append(heading);
    }
    if (status) {
      const line = document.createElement('p');
      line.className = `evidence-status${error ? ' is-error' : ''}`;
      line.textContent = status;
      view.append(line);
    }
    if (meta?.length) {
      const grid = document.createElement('div');
      grid.className = 'meta-grid';
      meta.forEach(([key, value]) => grid.append(metaRow(key, value)));
      view.append(grid);
    }
    if (table) view.append(table);
    if (code || Object.keys(checks || {}).length) view.append(codePanel(code, checks));
    return view;
  }

  function tableElement(payload, highlight) {
    const rows = payload?.rows || [];
    const headers = payload?.headers || [];
    if (!rows.length) {
      const raw = document.createElement('p');
      raw.className = 'evidence-raw';
      raw.textContent = 'Bảng trống.';
      return raw;
    }
    const wrap = document.createElement('div');
    wrap.className = 'evidence-table-wrap';
    const table = document.createElement('table');
    table.className = 'evidence-table';
    const head = document.createElement('thead');
    const headRow = document.createElement('tr');
    const columns = headers.length ? headers : rows[0].map((_, index) => `C${index + 1}`);
    columns.forEach((value) => {
      const cell = document.createElement('th');
      cell.scope = 'col';
      cell.textContent = String(value ?? '');
      headRow.append(cell);
    });
    head.append(headRow);
    table.append(head);
    const body = document.createElement('tbody');
    const target = highlight || payload?.highlight || {};
    const targetRow = Number.isFinite(Number(target.row)) ? Number(target.row) : -1;
    const targetColumn = Number.isFinite(Number(target.column)) ? Number(target.column) : -1;
    rows.forEach((values, rowIndex) => {
      const row = document.createElement('tr');
      (Array.isArray(values) ? values : [values]).forEach((value, columnIndex) => {
        const cell = document.createElement('td');
        cell.textContent = String(value ?? '');
        if (rowIndex === targetRow && columnIndex === targetColumn) cell.className = 'is-cited';
        row.append(cell);
      });
      body.append(row);
    });
    table.append(body);
    wrap.append(table);
    return wrap;
  }

  function codePanel(code, checks) {
    const panel = document.createElement('details');
    panel.className = 'code-panel';
    const summary = document.createElement('summary');
    summary.append(document.createTextNode('Phép tính'), icon('chevron', 'icon code-chevron'));
    panel.append(summary);
    if (code) {
      const pre = document.createElement('pre');
      pre.textContent = code;
      panel.append(pre);
    }
    if (Object.keys(checks || {}).length) {
      const list = document.createElement('div');
      list.className = 'checks';
      Object.entries(checks).forEach(([key, value]) => {
        const row = document.createElement('div');
        row.className = 'check-row';
        const name = document.createElement('span');
        name.textContent = key;
        const result = document.createElement('span');
        result.textContent = typeof value === 'object' ? JSON.stringify(value) : String(value);
        row.append(name, result);
        list.append(row);
      });
      panel.append(list);
    }
    return panel;
  }

  function showOpenTable(node) {
    state.openTable = node;
    renderEvidencePane();
  }

  function closeTable() {
    state.openTable = null;
    document.querySelectorAll('.evidence-card.is-selected').forEach((n) => n.classList.remove('is-selected'));
    renderEvidencePane();
  }

  function openCitation(citation, message) {
    const requestId = message?.requestId || state.currentRequestId;
    const meta = metaFor(citation.table_id);
    if (citation.column_label) meta.push(['Cột', citation.column_label]);
    if (citation.raw_value) meta.push(['Giá trị', citation.raw_value]);
    showOpenTable(tableView({
      title: citation.row_label || citation.label || 'Bằng chứng',
      status: 'Đang tải…', meta, code: message?.code, checks: message?.checks,
    }));
    highlightSource(citation.table_id);
    if (!requestId || citation.id == null) {
      showOpenTable(tableView({ title: citation.label, status: 'Thiếu request_id để tra cứu.', error: true, meta }));
      return;
    }
    fetch(`/api/evidence/${encodeURIComponent(requestId)}/${encodeURIComponent(citation.id)}`, { headers: { Accept: 'application/json' } })
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then((payload) => showOpenTable(tableView({
        title: citation.row_label || citation.label || 'Bằng chứng',
        status: 'Ô được trích dẫn đã đánh dấu.', meta,
        table: tableElement(payload), code: message?.code, checks: message?.checks,
      })))
      .catch((error) => showOpenTable(tableView({
        title: citation.label, status: error.message || 'Không tải được bằng chứng.', error: true, meta,
      })));
  }

  function metaFor(tableId) {
    const parts = sourceParts(tableId);
    const number = sourceNumber(tableId);
    return [
      number && ['Nguồn', number],
      parts.ticker && ['Mã', parts.ticker],
      parts.year && ['Năm', parts.year],
      parts.line && ['Dòng', parts.line],
    ].filter(Boolean);
  }

  function openTablePreview(tableId) {
    const requestId = state.currentRequestId;
    const meta = metaFor(tableId);
    highlightSource(tableId);
    const saved = state.savedTables.get(tableId);
    const render = (payload, note) => showOpenTable(tableView({
      title: payload.title || tableId,
      status: note === undefined ? (payload.truncated ? 'Đã cắt bớt phần cuối bảng.' : '') : note,
      meta, table: tableElement(payload),
    }));

    if (saved) { render(saved, saved.truncated ? 'Đã cắt bớt phần cuối bảng.' : ''); return; }
    showOpenTable(tableView({ title: tableId, status: 'Đang tải…', meta }));
    if (!requestId) {
      showOpenTable(tableView({ title: tableId, status: 'Bảng này chưa được lưu cùng câu trả lời.', error: true, meta }));
      return;
    }
    const key = `${requestId}::${tableId}`;
    if (previewCache.has(key)) { render(previewCache.get(key)); return; }
    fetch(`/api/table/${encodeURIComponent(requestId)}/${encodeURIComponent(tableId)}`, { headers: { Accept: 'application/json' } })
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then((payload) => { previewCache.set(key, payload); render(payload); })
      .catch((error) => showOpenTable(tableView({ title: tableId, status: error.message || 'Không tải được bảng.', error: true, meta })));
  }

  /* Hovering a source reference lights its card, and the reverse. */
  function highlightSource(tableId) {
    document.querySelectorAll('.evidence-card.is-linked, .source-ref.is-linked')
      .forEach((node) => node.classList.remove('is-linked'));
    if (!tableId) return;
    document.querySelectorAll(`[data-table-id="${CSS.escape(tableId)}"]`)
      .forEach((node) => node.classList.add('is-linked'));
  }

  function openCalculation(message) {
    showOpenTable(tableView({
      title: 'Phép tính', status: 'Biểu thức đã thực thi.',
      code: message?.code, checks: message?.checks,
    }));
  }

  async function loadHealth() {
    try {
      const response = await fetch('/api/health', { headers: { Accept: 'application/json' } });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.message || `HTTP ${response.status}`);
      const profileValue = data.model_profile || data.profile || data.model || data.data?.model_profile || '';
      const profile = typeof profileValue === 'object'
        ? profileValue.label || profileValue.mode || ''
        : String(profileValue);
      const mode = `${data.mode || ''} ${profile} ${data.profile?.mode || ''}`.toLowerCase();
      const fixture = data.fixture_mode === true || data.fixture === true || data.data?.fixture_mode === true || mode.includes('fixture');
      const endpoints = Object.values(data.endpoints || {});
      if (endpoints.includes('busy')) throw new Error('busy');
      if (data.ready === false || ['error', 'unavailable', 'not_ready'].includes(String(data.status || '').toLowerCase())) {
        throw new Error(data.message || 'Dịch vụ chưa sẵn sàng');
      }
      elements.fixtureBanner.hidden = !fixture;
      elements.serviceState.hidden = true;
    } catch (error) {
      elements.serviceState.className = 'service-state is-error';
      elements.serviceState.hidden = false;
      elements.serviceLabel.textContent = error.message === 'busy'
        ? 'Máy chủ mô hình đang bận'
        : 'Máy chủ mô hình không khả dụng';
    }
  }

  /* The evidence column is user-sizable; the width survives a reload. */
  const EVIDENCE_WIDTH_KEY = 'vifinqa.evidenceWidth';
  const EVIDENCE_MIN = 300;
  const EVIDENCE_MAX = 760;

  function setEvidenceWidth(px, persist) {
    const clamped = Math.min(EVIDENCE_MAX, Math.max(EVIDENCE_MIN, Math.round(px)));
    document.documentElement.style.setProperty('--evidence-w', `${clamped}px`);
    if (persist) { try { localStorage.setItem(EVIDENCE_WIDTH_KEY, String(clamped)); } catch (_) {} }
  }

  function restoreEvidenceWidth() {
    let saved = null;
    try { saved = localStorage.getItem(EVIDENCE_WIDTH_KEY); } catch (_) {}
    if (saved) setEvidenceWidth(Number(saved), false);
  }

  function startResize(event) {
    if (matchMedia('(max-width: 900px)').matches) return;
    event.preventDefault();
    const resizer = elements.panesResizer;
    resizer.classList.add('is-dragging');
    elements.body.classList.add('is-resizing');
    const move = (move_event) => {
      setEvidenceWidth(elements.panes.getBoundingClientRect().right - move_event.clientX, false);
    };
    const end = (end_event) => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', end);
      resizer.classList.remove('is-dragging');
      elements.body.classList.remove('is-resizing');
      setEvidenceWidth(elements.panes.getBoundingClientRect().right - end_event.clientX, true);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', end);
  }

  /* Rail collapse. The width lives on the grid track so the workspace reflows
     with it; the choice persists per browser. */
  const RAIL_KEY = 'vifinqa.railCollapsed';

  function setRailCollapsed(collapsed, persist) {
    elements.body.classList.toggle('rail-collapsed', collapsed);
    elements.railToggle.setAttribute('aria-expanded', String(!collapsed));
    elements.railToggle.setAttribute('aria-label', collapsed ? 'Mở thanh bên' : 'Thu gọn thanh bên');
    elements.sidebar.inert = collapsed;
    if (persist) { try { localStorage.setItem(RAIL_KEY, collapsed ? '1' : '0'); } catch (_) {} }
  }

  function restoreRail() {
    let saved = null;
    try { saved = localStorage.getItem(RAIL_KEY); } catch (_) {}
    setRailCollapsed(saved === '1', false);
  }

  function closeMenu() {
    elements.body.classList.remove('menu-open');
    elements.menuButton.setAttribute('aria-expanded', 'false');
  }

  elements.form.addEventListener('submit', (event) => { event.preventDefault(); submitQuestion(elements.input.value); });
  elements.input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); elements.form.requestSubmit(); }
    requestAnimationFrame(() => { elements.input.style.height = 'auto'; elements.input.style.height = `${Math.min(elements.input.scrollHeight, 150)}px`; });
  });
  elements.messages.addEventListener('click', (event) => {
    const example = event.target.closest('[data-question]');
    if (example) { elements.input.value = example.dataset.question; elements.form.requestSubmit(); return; }

    const stageRow = event.target.closest('[data-stage-toggle]');
    if (stageRow) {
      const stage = stageRow.dataset.stageToggle;
      state.stageOpen[stage] = !(stage in state.stageOpen ? state.stageOpen[stage] : false);
      renderTrace(state.trace);
      return;
    }

    const ref = event.target.closest('.source-ref');
    if (ref) {
      const message = activeSession()?.messages.find((item) => item.requestId === ref.dataset.requestId);
      const citation = message?.citations?.[Number(ref.dataset.citationIndex)];
      if (citation) openCitation(citation, message);
      return;
    }

    const calculation = event.target.closest('.calculation-button');
    if (calculation) {
      const index = Number(calculation.closest('.message')?.dataset.messageIndex);
      const assistant = activeSession()?.messages[index];
      if (assistant) openCalculation(assistant);
    }
  });

  elements.evidenceBody.addEventListener('click', (event) => {
    const card = event.target.closest('.evidence-card');
    if (!card) return;
    document.querySelectorAll('.evidence-card.is-selected').forEach((node) => node.classList.remove('is-selected'));
    card.classList.add('is-selected');
    openTablePreview(card.dataset.tableId);
  });
  elements.evidenceBack.addEventListener('click', closeTable);
  elements.panesResizer.addEventListener('pointerdown', startResize);
  elements.panesResizer.addEventListener('keydown', (event) => {
    const step = event.shiftKey ? 48 : 16;
    const current = elements.evidencePane?.getBoundingClientRect().width
      || parseInt(getComputedStyle(document.documentElement).getPropertyValue('--evidence-w'), 10) || 420;
    if (event.key === 'ArrowLeft') { event.preventDefault(); setEvidenceWidth(current + step, true); }
    if (event.key === 'ArrowRight') { event.preventDefault(); setEvidenceWidth(current - step, true); }
  });

  // Hovering a source reference lights its evidence card immediately.
  document.addEventListener('pointerover', (event) => {
    const node = event.target.closest?.('[data-table-id]');
    if (node) highlightSource(node.dataset.tableId);
  });

  elements.newChat.addEventListener('click', () => {
    if (state.controller) stopRequest();
    const session = createSession();
    state.activeId = session.id;
    state.trace = [];
    state.stageOpen = {};
    state.sources.clear();
    state.cards = [];
    state.savedTables = new Map();
    state.openTable = null;
    renderEvidencePane();
    saveSessions();
    render();
    elements.input.focus();
    closeMenu();
  });
  elements.railToggle.addEventListener('click', () => {
    setRailCollapsed(!elements.body.classList.contains('rail-collapsed'), true);
  });
  elements.stopButton.addEventListener('click', stopRequest);
  elements.menuButton.addEventListener('click', () => {
    const open = elements.body.classList.toggle('menu-open');
    elements.menuButton.setAttribute('aria-expanded', String(open));
  });
  document.addEventListener('keydown', (event) => { if (event.key === 'Escape') { closeTable(); closeMenu(); } });

  restoreEvidenceWidth();
  restoreRail();
  restoreEvidence(activeSession());
  render();
  loadHealth();
})();
