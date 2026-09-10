import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from api.cors import DEFAULT_ORIGINS, allowed_origins


def test_defaults_and_bounded_exact_additions():
    assert allowed_origins() == list(DEFAULT_ORIGINS)
    assert allowed_origins('  ') == list(DEFAULT_ORIGINS)
    origin = 'https://approved-preview.vercel.app'
    assert allowed_origins(f' {origin}, {origin} ') == [*DEFAULT_ORIGINS, origin]
    with pytest.raises(ValueError):
        allowed_origins(','.join([origin]*21))


@pytest.mark.parametrize('value', ['*', 'https://*.vercel.app', 'https://ok.app/',
    'https://ok.app/path', 'https://ok.app?', 'https://ok.app#', 'https://a:b@ok.app',
    'http://preview.vercel.app', 'https://ok.app:99999', 'https://ok.app,', 'null',
    'https://UPPER.app', 'https://bad\\.app', 'https://ok.app:'])
def test_invalid_origin_fails_closed(value):
    with pytest.raises(ValueError):
        allowed_origins(value)


def test_exact_origin_preflight_not_suffix_or_wildcard():
    app = FastAPI()
    origin = 'https://approved-preview.vercel.app'
    app.add_middleware(CORSMiddleware, allow_origins=allowed_origins(origin),
        allow_methods=['GET', 'POST'], allow_headers=['*'])
    @app.get('/health')
    def health(): return {'status':'ok'}
    with TestClient(app) as client:
        for candidate, accepted in [(origin, True), (DEFAULT_ORIGINS[-1], True),
            (origin+'.evil.test', False), ('https://other.vercel.app', False)]:
            response = client.options('/chat/stream', headers={'Origin':candidate,
                'Access-Control-Request-Method':'POST', 'Access-Control-Request-Headers':'content-type'})
            assert response.status_code == (200 if accepted else 400)
            assert response.headers.get('access-control-allow-origin') == (candidate if accepted else None)
            assert 'access-control-allow-credentials' not in response.headers
        assert client.get('/health', headers={'Origin':origin}).headers['access-control-allow-origin']==origin
