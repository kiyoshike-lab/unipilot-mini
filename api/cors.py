"""Bounded, exact-origin CORS configuration. Invalid additions fail closed."""
from __future__ import annotations

import re
from urllib.parse import urlsplit

DEFAULT_ORIGINS = (
    'http://localhost:3000',
    'http://127.0.0.1:3000',
    'https://unipilot-mini-pjgy.vercel.app',
)


def allowed_origins(value: str | None = None) -> list[str]:
    origins = list(DEFAULT_ORIGINS)
    if not value or not value.strip():
        return origins
    if len(value) > 8192 or len(value.split(',')) > 20:
        raise ValueError('CORS additions exceed bounded allowlist (20 origins / 8192 characters)')
    for item in value.split(','):
        origin = item.strip()
        parsed = urlsplit(origin)
        if (not origin or '*' in origin or re.search(r'[\s\\]', origin)
                or parsed.scheme not in ('http', 'https') or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.path or parsed.query or parsed.fragment
                or '?' in origin or '#' in origin
                or len(origin) > 512):
            raise ValueError('CORS entries must be exact http(s) origins, without paths or wildcards')
        if parsed.scheme == 'http' and parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
            raise ValueError('Non-local CORS origins require HTTPS')
        # Accessing port validates malformed / out-of-range values as well.
        port = parsed.port
        host = f'[{parsed.hostname}]' if ':' in parsed.hostname else parsed.hostname
        if ':' not in parsed.hostname and not re.fullmatch(r'[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?', parsed.hostname):
            raise ValueError('Invalid CORS hostname')
        canonical = f'{parsed.scheme}://{host}' + (f':{port}' if port is not None else '')
        if canonical != origin:
            raise ValueError('Use a canonical, lowercase exact origin')
        if origin not in origins:
            origins.append(origin)
    return origins
