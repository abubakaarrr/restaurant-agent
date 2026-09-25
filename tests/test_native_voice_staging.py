"""Staging boundaries: explicit opt-in, real database identity, HTTPS and auth."""
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware
from starlette.websockets import WebSocketDisconnect
from app.config import Settings
from app.native_voice import database_guard as guard
import app.config as config
from app.native_voice.dashboard_voice import attach_local_voice

ORIGIN = "https://agent.servicesground.com"

def valid_settings(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fictional-key")
    monkeypatch.setenv("NATIVE_VOICE_ALLOW_SHARED_DATABASE", "true")
    monkeypatch.setenv("NATIVE_VOICE_DATABASE_WRITE_ENABLED", "true")
    monkeypatch.setenv("NATIVE_VOICE_DATABASE_MARKER", "fictional-qa-marker-1234567890")
    return Settings(_env_file=None, app_env="staging", native_voice_staging_enabled=True,
        native_voice_realtime_enabled=True, voice_live_writes_enabled=True,
        openai_api_key="fictional-openai-key", session_secret="s"*32,
        dashboard_api_key="d"*32, login_username="qa", login_password="fictional-password",
        allowed_origins=ORIGIN, enable_legacy_retell_custom_llm=False,
        database_url="postgresql://qa:password@127.0.0.1:5432/restaurant_agent")

def test_explicit_staging_configuration(monkeypatch):
    valid_settings(monkeypatch).validate_native_voice_staging()

@pytest.mark.parametrize("field,value", [
    ("app_env","production"), ("app_env","development"),
    ("native_voice_staging_enabled",False),
    ("allowed_origins","http://agent.servicesground.com"),
    ("allowed_origins","https://*.servicesground.com"),
    ("allowed_origins",ORIGIN+"/"), ("session_secret",""),
    ("login_password","admin"), ("enable_legacy_retell_custom_llm",True),
    ("native_voice_realtime_enabled",False), ("voice_live_writes_enabled",False),
])
def test_unsafe_staging_configuration_fails(monkeypatch,field,value):
    settings=valid_settings(monkeypatch)
    setattr(settings,field,value)
    with pytest.raises(RuntimeError): settings.validate_native_voice_staging()

@pytest.mark.parametrize("env", [
    "NATIVE_VOICE_ALLOW_SHARED_DATABASE", "NATIVE_VOICE_DATABASE_WRITE_ENABLED",
    "NATIVE_VOICE_DATABASE_MARKER", "GEMINI_API_KEY",
])
def test_missing_required_environment_fails(monkeypatch,env):
    settings=valid_settings(monkeypatch)
    monkeypatch.delenv(env)
    monkeypatch.delenv("GOOGLE_API_KEY",raising=False)
    with pytest.raises(RuntimeError): settings.validate_native_voice_staging()

@pytest.mark.asyncio
async def test_shared_database_requires_staging_and_matching_database_marker(monkeypatch):
    settings=valid_settings(monkeypatch)
    monkeypatch.setattr(guard,"settings",settings)
    monkeypatch.setenv("DATABASE_URL",settings.database_url)
    assert guard.validate_native_voice_database()==settings.database_url
    pool=SimpleNamespace(fetchrow=AsyncMock(return_value={
        "server_host":"127.0.0.1","server_port":5432,
        "database_name":"restaurant_agent","marker":"fictional-qa-marker-1234567890"}))
    await guard.verify_native_voice_database_connection(pool)
    assert "app.native_voice_staging_marker" in pool.fetchrow.call_args.args[0]
    pool.fetchrow.return_value["marker"]="wrong"
    with pytest.raises(guard.NativeVoiceDatabaseGuardError):
        await guard.verify_native_voice_database_connection(pool)
    pool.fetchrow.return_value["marker"]="fictional-qa-marker-1234567890"
    pool.fetchrow.return_value["database_name"]="wrong"
    with pytest.raises(guard.NativeVoiceDatabaseGuardError):
        await guard.verify_native_voice_database_connection(pool)
    settings.app_env="development"
    monkeypatch.setenv("NATIVE_VOICE_DATABASE_URL",settings.database_url)
    with pytest.raises(guard.NativeVoiceDatabaseGuardError,match="must_be_separate"):
        guard.validate_native_voice_database()

@pytest.mark.parametrize("authenticated,origin,accepted", [
    (True,ORIGIN,True),(False,ORIGIN,False),
    (True,"https://evil.example",False),(True,"http://agent.servicesground.com",False),
    (True,ORIGIN+".evil.example",False),(True,"null",False),
])
def test_https_proxy_voice_requires_login_and_exact_origin(monkeypatch,authenticated,origin,accepted):
    settings=valid_settings(monkeypatch)
    monkeypatch.setattr(config,"settings",settings)
    app=FastAPI()
    app.add_middleware(SessionMiddleware,secret_key="test",https_only=True)
    session=AsyncMock()
    async def connector(*,send,pace):
        async def start():await send({"type":"ready"})
        session.start.side_effect=start
        return session
    attach_local_voice(app,connector=connector,allowed_origin=ORIGIN)
    @app.get("/login-test")
    async def login(request:Request):
        request.session["authenticated"]=True
        return {}
    client=TestClient(app,base_url=ORIGIN)
    if authenticated:
        response=client.get("/login-test")
        assert "secure" in response.headers["set-cookie"].lower()
    def connect():
        return client.websocket_connect(ORIGIN.replace("https:","wss:")+"/voice-gemini",
            headers={"origin":origin,"host":"127.0.0.1:8000"})
    if accepted:
        with connect() as ws:
            assert ws.receive_json()=={"type":"ready"}
            ws.send_json({"type":"end"})
    else:
        with pytest.raises(WebSocketDisconnect):
            with connect():pass
        session.start.assert_not_awaited()


def test_actual_staging_entrypoint_startup_and_shutdown():
    # Isolate app.main's process-wide settings/routes from the rest of the suite.
    import os, subprocess, sys
    environment = os.environ.copy()
    environment.update(APP_ENV="staging", NATIVE_VOICE_STAGING_ENABLED="true",
        NATIVE_VOICE_REALTIME_ENABLED="true", VOICE_LIVE_WRITES_ENABLED="true",
        NATIVE_VOICE_ALLOW_SHARED_DATABASE="true", NATIVE_VOICE_DATABASE_WRITE_ENABLED="true",
        NATIVE_VOICE_DATABASE_MARKER="fictional-qa-marker-1234567890",
        GEMINI_API_KEY="fictional-key", OPENAI_API_KEY="fictional-key",
        SESSION_SECRET="s"*32,DASHBOARD_API_KEY="d"*32,
        LOGIN_USERNAME="qa",LOGIN_PASSWORD="fictional-password", ALLOWED_ORIGINS=ORIGIN,
        ENABLE_LEGACY_RETELL_CUSTOM_LLM="false",ENABLE_LEGACY_VAPI="false",
        ENABLE_PUBLIC_WEB_CALLS="false",WIDGET_ENABLED="false",
        DATABASE_URL="postgresql://qa:password@127.0.0.1:5432/restaurant_agent")
    source = """
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
import app.native_voice.staging as staging
pool = AsyncMock()
pool.fetchval.return_value = True
staging.get_native_voice_pool = AsyncMock(return_value=pool)
staging.close_native_voice_pool = AsyncMock()
with TestClient(staging.app, base_url='https://agent.servicesground.com') as client:
    assert staging.dashboard_pool._pool is pool
    response=client.post('/login',data={'username':'qa','password':'fictional-password'},follow_redirects=False)
    assert response.status_code==302
    assert 'secure' in response.headers['set-cookie'].lower()
    page=client.get('/')
    assert page.status_code==200 and '/assets/gemini.js' in page.text
    assert client.post('/api/settings',json={}).status_code==403
    assert client.get('/info').json()['database']=='QA staging database'
    assert client.get('/',headers={'host':'evil.example'}).status_code==400
assert staging.dashboard_pool._pool is None
staging.close_native_voice_pool.assert_awaited_once()
pool.fetchval.return_value=False
try:
    with TestClient(staging.app):pass
except RuntimeError as exc:
    assert 'migrations' in str(exc)
else:
    raise AssertionError('Missing migration accepted')
assert staging.close_native_voice_pool.await_count==2
"""
    result=subprocess.run([sys.executable,"-B","-c",source],env=environment,
        capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
