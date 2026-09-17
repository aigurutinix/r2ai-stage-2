"""AES-256-GCM envelope for tenant-uploaded confidential payloads."""

from __future__ import annotations

import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .secure_config import resolve_secret


MAGIC = b"KINGPRO-AESGCM-v1\x00"


def encryption_key() -> bytes | None:
    encoded = resolve_secret("KINGPRO_DATA_ENCRYPTION_KEY", "")
    if not encoded:
        return None
    try:
        key = base64.urlsafe_b64decode(encoded.encode("ascii"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("KINGPRO_DATA_ENCRYPTION_KEY must be urlsafe base64") from exc
    if len(key) != 32:
        raise RuntimeError("KINGPRO_DATA_ENCRYPTION_KEY must decode to 32 bytes")
    return key


def encrypt_bytes(payload: bytes, *, aad: bytes) -> bytes:
    key = encryption_key()
    if key is None:
        raise RuntimeError("data encryption key is not configured")
    nonce = os.urandom(12)
    return MAGIC + nonce + AESGCM(key).encrypt(nonce, payload, aad)


def decrypt_bytes(envelope: bytes, *, aad: bytes) -> bytes:
    key = encryption_key()
    if key is None:
        raise RuntimeError("data encryption key is not configured")
    if not envelope.startswith(MAGIC) or len(envelope) <= len(MAGIC) + 12:
        raise ValueError("invalid encrypted envelope")
    nonce = envelope[len(MAGIC) : len(MAGIC) + 12]
    ciphertext = envelope[len(MAGIC) + 12 :]
    return AESGCM(key).decrypt(nonce, ciphertext, aad)


def encryption_health() -> dict[str, object]:
    configured = encryption_key() is not None
    required = os.getenv("KINGPRO_REQUIRE_DATA_ENCRYPTION", "false").lower() in {
        "1", "true", "yes", "on"
    }
    return {
        "configured": configured,
        "required": required,
        "ready": configured or not required,
        "algorithm": "AES-256-GCM" if configured else None,
    }
