"""Authentication and RBAC boundary for HTTP product traffic.

Modes:
- ``off``: loopback/demo operator, all roles (default; Compose only publishes localhost).
- ``token``: static bearer tokens from ``KINGPRO_API_TOKENS_JSON``.
- ``oidc``: RS256 JWT verified against issuer JWKS using PyJWT.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from typing import Any, Mapping


ALL_ROLES = frozenset({"viewer", "analyst", "approver", "admin"})


class AuthError(RuntimeError):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class Principal:
    subject: str
    tenant_id: str
    roles: frozenset[str]
    auth_mode: str

    def can(self, role: str) -> bool:
        return "admin" in self.roles or role in self.roles

    def public(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "tenant_id": self.tenant_id,
            "roles": sorted(self.roles),
            "auth_mode": self.auth_mode,
        }


def _bearer(headers: Mapping[str, str]) -> str:
    raw = str(headers.get("Authorization") or "")
    if not raw.lower().startswith("bearer "):
        raise AuthError(401, "missing_bearer_token", "Bearer authentication is required.")
    token = raw[7:].strip()
    if not token:
        raise AuthError(401, "missing_bearer_token", "Bearer authentication is required.")
    return token


class AuthPolicy:
    def __init__(self, mode: str | None = None) -> None:
        self.mode = (mode or os.getenv("KINGPRO_AUTH_MODE", "off")).strip().lower()
        if self.mode not in {"off", "token", "oidc"}:
            raise ValueError("KINGPRO_AUTH_MODE must be off, token or oidc")

    def _token_principal(self, token: str) -> Principal:
        raw = os.getenv("KINGPRO_API_TOKENS_JSON", "{}")
        try:
            records = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("KINGPRO_API_TOKENS_JSON is invalid") from exc
        if not isinstance(records, dict):
            raise RuntimeError("KINGPRO_API_TOKENS_JSON must be an object")
        supplied = hashlib.sha256(token.encode("utf-8")).digest()
        selected = None
        for configured, record in records.items():
            expected = hashlib.sha256(str(configured).encode("utf-8")).digest()
            if hmac.compare_digest(supplied, expected):
                selected = record
        if not isinstance(selected, dict):
            raise AuthError(401, "invalid_token", "Bearer token is invalid.")
        roles = frozenset(str(role) for role in selected.get("roles", ["viewer"]))
        if not roles <= ALL_ROLES:
            raise AuthError(403, "invalid_role", "Token contains an unsupported role.")
        return Principal(
            subject=str(selected.get("subject") or "token-user"),
            tenant_id=str(selected.get("tenant_id") or "default"),
            roles=roles,
            auth_mode="token",
        )

    def _oidc_principal(self, token: str) -> Principal:
        try:
            import jwt
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise RuntimeError("PyJWT[crypto] is required for OIDC mode") from exc
        issuer = os.getenv("KINGPRO_OIDC_ISSUER", "").rstrip("/")
        audience = os.getenv("KINGPRO_OIDC_AUDIENCE", "")
        jwks_url = os.getenv("KINGPRO_OIDC_JWKS_URL", f"{issuer}/.well-known/jwks.json")
        if not issuer or not audience:
            raise RuntimeError("OIDC issuer and audience must be configured")
        signing_key = jwt.PyJWKClient(jwks_url).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=audience,
            issuer=issuer,
            options={"require": ["exp", "iat", "sub"]},
        )
        role_claim = os.getenv("KINGPRO_OIDC_ROLES_CLAIM", "roles")
        tenant_claim = os.getenv("KINGPRO_OIDC_TENANT_CLAIM", "tenant_id")
        raw_roles = claims.get(role_claim) or []
        if isinstance(raw_roles, str):
            raw_roles = raw_roles.split()
        roles = frozenset(str(role) for role in raw_roles) or frozenset({"viewer"})
        if not roles <= ALL_ROLES:
            raise AuthError(403, "invalid_role", "OIDC token contains an unsupported role.")
        tenant = str(claims.get(tenant_claim) or "").strip()
        if not tenant:
            raise AuthError(403, "missing_tenant", "OIDC token has no tenant claim.")
        return Principal(str(claims["sub"]), tenant, roles, "oidc")

    def authenticate(self, headers: Mapping[str, str], required_role: str = "viewer") -> Principal:
        if required_role not in ALL_ROLES:
            raise ValueError(f"unsupported role: {required_role}")
        if self.mode == "off":
            principal = Principal("local-operator", "default", ALL_ROLES, "off")
        else:
            token = _bearer(headers)
            principal = (
                self._token_principal(token)
                if self.mode == "token"
                else self._oidc_principal(token)
            )
        if not principal.can(required_role):
            raise AuthError(403, "insufficient_role", f"Role {required_role} is required.")
        return principal

    def health(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "enabled": self.mode != "off",
            "oidc_configured": bool(
                os.getenv("KINGPRO_OIDC_ISSUER") and os.getenv("KINGPRO_OIDC_AUDIENCE")
            ),
            "roles": sorted(ALL_ROLES),
        }

