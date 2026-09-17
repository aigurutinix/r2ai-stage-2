export type Citation = {
  table_ref: string;
  document: string;
  company: string;
  ticker: string;
  year: string;
  scope: string;
  page?: number;
  table_line?: number;
  section: string;
  verified: boolean;
};

export type TablePreviewResponse = {
  status: "ok";
  table_ref: string;
  columns: string[];
  rows: string[][];
  row_count: number;
  column_count: number;
  truncated: boolean;
};

export type RetrievalChecks = {
  requested_pairs: [string, string][];
  found_pairs: [string, string][];
  pair_coverage: number;
  anchor_coverage: number;
  confidence: number;
  threshold: number;
};

export type AgentTraceStage = {
  stage: string;
  title: string;
  status: "started" | "running" | "completed" | "warning" | "error";
  elapsed_ms: number;
  detail: string;
  metadata: Record<string, unknown>;
};

export type VerifiedSourceCell = {
  question_id?: number;
  source_index?: number;
  table_ref: string;
  source_path?: string;
  row_idx: number;
  col_idx: number;
  source_label?: string;
  header_path?: string[];
  raw_manifest?: string;
  raw_physical?: string;
  ticker?: string;
  year?: string;
  metric_key?: string;
  verified?: boolean;
  verification?: string;
};

export type AnsweredResponse = {
  trace_id: string;
  status: "answered";
  grounded: true;
  answer: number;
  unit: string;
  pandas_query: string;
  citations: Citation[];
  retrieval: {
    facets: { tickers: string[]; years: string[]; scope: string; analytic: boolean };
    checks: RetrievalChecks;
    documents: string[];
    tables: string[];
  };
  verification: {
    citation_bound: boolean;
    replay_match: boolean;
    vote_ratio: number;
    registry_match?: boolean;
    registry_exact?: boolean;
    mode?: "verified_registry" | "verified_registry_paraphrase" | "deterministic_compiler" | "generated";
    compiler_metric?: string;
    source_cells?: VerifiedSourceCell[];
    cell_lineage?: {
      status: "verified" | "unavailable";
      verified_cells: number;
      causal_cells?: number;
      policy: string;
    };
    paraphrase_match?: {
      exact: false;
      score: number;
      query_precision: number;
      candidate_recall: number;
      shared_terms: number;
      margin: number;
      source_question_id?: number;
    } | null;
  };
  trace?: AgentTraceStage[];
  elapsed_ms: number;
};

export type RefusedResponse = {
  trace_id: string;
  status: "refused";
  grounded: false;
  answer: null;
  pandas_query: "";
  citations: [];
  refusal: {
    code: string;
    message: string;
    details: Record<string, unknown>;
  };
};

export type ProductResponse = AnsweredResponse | RefusedResponse;

export type HealthResponse = {
  status: string;
  catalog: boolean;
  bm25_index: boolean;
  llm_configured: boolean;
  model: string;
  model_allowed: boolean;
  allowed_models: string[];
  endpoint_allowed?: boolean;
  endpoint_host?: string;
  endpoint_provider?: string;
  endpoint_secure_transport?: boolean;
  model_attested?: boolean;
  dynamic_generation_available?: boolean;
  demo_readiness?: {
    report_present: boolean;
    report_name: string;
    stage_safe_now: boolean;
    full_demo_ready_now: boolean;
    technical_passed: number;
    technical_total: number;
    smoke_passed: number;
    smoke_total: number;
    smoke_elapsed_ms: number;
    runtime_operational_reported: boolean;
    manual_confirmed: number;
    manual_total: number;
    manual_complete: boolean;
    manual_confirmed_keys: string[];
    manual_pending_keys: string[];
    claim_limit: string;
  };
  deterministic_compiler?: boolean;
  compiler_metrics?: string[];
  replay_registry: boolean;
  replay_entries: number;
  replay_artifact?: string;
  cell_lineage?: {
    manifest_questions: number;
    source_audit_questions: number;
    audited_union_questions: number;
    counterfactual_questions: number;
    runtime_lineage_questions: number;
    submission_questions: number;
    policy: string;
  };
  public_champion?: {
    version: string;
    submission_id: number;
    artifact: string;
    artifact_sha256: string;
    on_leaderboard: boolean;
    scores: {
      execution_accuracy: number;
      answer_accuracy: number;
      tables_f2_macro: number;
      tables_precision: number;
      tables_recall: number;
      tables_mrr5: number;
      docs_f2_macro: number;
      docs_precision: number;
      docs_recall: number;
      docs_mrr5: number;
    };
    claim_scope: string;
  };
  local_successor?: {
    version: string;
    artifact: string;
    artifact_sha256: string;
    submission_id?: number;
    measured: boolean;
    complete_score_vector?: boolean;
    scores?: { execution_accuracy?: number };
    claim_scope: string;
  };
  audited_candidate?: {
    version: string;
    artifact: string;
    artifact_sha256: string;
    submission_json_sha256: string;
    measured: boolean;
    submission_id: number | null;
    complete_score_vector?: boolean;
    scores?: { execution_accuracy?: number; answer_accuracy?: number };
    release_gate: "PASS" | "FAIL";
    individual_audit_questions: number;
    claim_scope: string;
  };
  source_clean_fallback?: {
    version: string;
    artifact: string;
    artifact_sha256: string;
    submission_json_sha256: string;
    measured: boolean;
    submission_id: number;
    complete_score_vector: boolean;
    scores: {
      execution_accuracy: number;
      answer_accuracy: number;
      tables_f2_macro: number;
      tables_precision: number;
      tables_recall: number;
      tables_mrr5: number;
      docs_f2_macro: number;
      docs_precision: number;
      docs_recall: number;
      docs_mrr5: number;
    };
    release_gate: "PASS" | "FAIL";
    source_lineage_repairs: number[];
    on_leaderboard: boolean;
    claim_scope: string;
  };
  paraphrase_replay?: boolean;
  paraphrase_entries?: number;
  paraphrase_exact_only_entries?: number;
  paraphrase_policy?: {
    min_score: number;
    min_candidate_recall: number;
    min_query_precision: number;
    min_margin: number;
    min_shared_terms: number;
    exact_facets: string[];
  };
};

export type CatalogCompany = {
  ticker: string;
  company: string;
  year_from: string;
  year_to: string;
  year_count: number;
  report_count: number;
  table_count: number;
  scopes: string[];
};

export type CatalogResponse = {
  status: string;
  company_count: number;
  report_count: number;
  table_count: number;
  years: string[];
  scope_counts: Record<string, number>;
  companies: CatalogCompany[];
};
