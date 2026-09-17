/** Forward end-user bearer auth, or use a server-only service token for local UI. */
export function backendHeaders(request?: Request, extra: Record<string, string> = {}) {
  const headers: Record<string, string> = { ...extra };
  const incoming = request?.headers.get("authorization")?.trim();
  const service = process.env.KINGPRO_SERVICE_TOKEN?.trim();
  if (incoming) headers.Authorization = incoming;
  else if (service) headers.Authorization = `Bearer ${service}`;
  return headers;
}

