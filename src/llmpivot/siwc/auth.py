from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
import time
import uuid
import webbrowser
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
try:
    import jwt
except ImportError:
    raise ImportError("SIWC requires: pip install 'aq-llm-pivot[siwc]'") from None

ISSUER = "https://auth.openai.com"
AUTHORIZE = ISSUER + "/api/accounts/authorize"
TOKEN = ISSUER + "/api/accounts/oauth/token"
RESOURCE = "https://api.openai.com/v1"
SCOPE = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
TERMINAL_REFRESH = {"invalid_grant", "invalid_refresh_token", "token_expired",
                    "refresh_token_expired", "refresh_token_invalidated", "refresh_token_reused"}


class SIWCError(RuntimeError):
    pass


def safe_code(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", value) else "unknown_error"


def check_http(response, operation):
    if response.is_success:
        return
    try:
        body = response.json()
        error = body.get("error", {}) if isinstance(body, dict) else {}
        code = error.get("code") if isinstance(error, dict) else error
    except ValueError:
        code = None
    raise SIWCError(f"{operation}: HTTP {response.status_code}, code={safe_code(code)}. "
                    "Check account permissions/usage at https://chatgpt.com/settings/usage")


class CredentialStore:
    def __init__(self, directory=None, app_name=None):
        from ..config import SIWCConfig
        app_name = app_name or SIWCConfig().app_name
        self.app_name = app_name
        app_id = hashlib.sha256(app_name.encode()).hexdigest()[:16]
        default = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "llm-pivot" / "siwc" / app_id
        self.directory = Path(directory or os.environ.get("LLM_PIVOT_SIWC_DIR", default)).expanduser().resolve()
        self.path = self.directory / "credentials.json"

    @contextmanager
    def locked(self):
        try:
            import fcntl
        except ImportError:
            raise SIWCError("SIWC credential locking currently supports macOS/Linux.") from None
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.directory, 0o700)
        fd = os.open(self.directory / "credentials.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "a") as lock:
            os.fchmod(lock.fileno(), 0o600)
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def read(self):
        if not self.path.exists():
            return {}
        fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd) as handle:
            os.fchmod(handle.fileno(), 0o600)
            return json.load(handle)

    def write(self, record):
        fd, name = tempfile.mkstemp(prefix=".credentials-", dir=self.directory)
        try:
            with os.fdopen(fd, "w") as handle:
                os.fchmod(handle.fileno(), 0o600)
                json.dump(record, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.path)
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(name):
                os.unlink(name)


def token_request(data):
    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(TOKEN, data={**data, "resource": RESOURCE})
        check_http(response, "OAuth token exchange")
        return response.json()
    except httpx.RequestError:
        raise SIWCError("OAuth network failure; credentials preserved. Retry later.") from None


def merge_tokens(record, data, *, initial=False):
    required = ("access_token", "refresh_token", "token_type", "expires_in")
    if any(not data.get(k) for k in required) or data["token_type"].lower() != "bearer":
        raise SIWCError("Token endpoint returned an incomplete token set.")
    scopes = data.get("scope")
    if initial and not isinstance(scopes, str):
        raise SIWCError("Token endpoint omitted granted scopes.")
    result = {**record, **{k: data[k] for k in required}, "saved_at": time.time()}
    result["expires_at"] = result["saved_at"] + float(data["expires_in"])
    result["scopes"] = scopes.split() if isinstance(scopes, str) else record.get("scopes", [])
    if data.get("id_token"):
        result["id_token"] = data["id_token"]
    if "earliest_refresh_at" in data:
        result["earliest_refresh_at"] = data["earliest_refresh_at"]
    return result


def validate_identity(id_token, client_id, nonce):
    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(ISSUER + "/.well-known/jwks.json")
            check_http(response, "OIDC signing keys")
            keys = jwt.PyJWKSet.from_dict(response.json())
        header = jwt.get_unverified_header(id_token)
        key = next(k for k in keys.keys if k.key_id == header.get("kid"))
        identity = jwt.decode(id_token, key.key, algorithms=["RS256"],
                              issuer=ISSUER, audience=client_id, leeway=5,
                              options={"require": ["sub", "exp", "iat", "nonce"]})
        if not identity["sub"] or not secrets.compare_digest(identity["nonce"], nonce):
            raise ValueError("invalid identity")
        return identity
    except (jwt.PyJWTError, StopIteration, ValueError, TypeError, httpx.RequestError):
        raise SIWCError("ID token signature/issuer/audience/expiry/nonce validation failed.") from None


def login(store, config, consent=False):
    with store.locked():
        saved = store.read()
        if saved.get("app_name", config.app_name) != config.app_name:
            raise SIWCError("Configured app name differs from this credential registration.")
        if not saved.get("ext_agent_host_id"):
            saved["app_name"] = config.app_name
            saved["ext_agent_host_id"] = "urn:uuid:" + str(uuid.uuid4())
            store.write(saved)
        client_id = saved.get("client_id", "dynamic_agent_client")
        state, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(3))
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        result = {}

        class Callback(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                parsed = urlsplit(self.path)
                params = parse_qs(parsed.query)
                valid = (parsed.path == "/auth/callback" and
                         len(params.get("state", [])) == 1 and
                         secrets.compare_digest(params["state"][0], state))
                self.send_response(200 if valid else 400)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(f"Return to {config.app_name}.".encode("utf-8") if valid else b"Invalid callback.")
                if valid:
                    result.update(params)

        with HTTPServer(("127.0.0.1", 0), Callback) as server:
            server.timeout = 0.5
            redirect = f"http://127.0.0.1:{server.server_port}/auth/callback"
            query = dict(client_id=client_id, ext_agent_host_id=saved["ext_agent_host_id"],
                         response_type="code", redirect_uri=redirect, scope=SCOPE, resource=RESOURCE,
                         state=state, nonce=nonce, code_challenge_method="S256", code_challenge=challenge)
            if client_id == "dynamic_agent_client":
                query["agent_name_hint"] = config.app_name
            elif saved.get("id_token"):
                query["id_token_hint"] = saved["id_token"]
            if consent:
                query["prompt"] = "consent"

            if not webbrowser.open(AUTHORIZE + "?" + urlencode(query)):
                raise SIWCError("Could not open system browser. Run llm-pivot-siwc login in a local desktop terminal.")
            print("Continue with ChatGPT: 请在浏览器完成登录，并选择是否授权使用 ChatGPT plan。", flush=True)
            deadline = time.monotonic() + config.login_timeout
            while not result and time.monotonic() < deadline:
                server.handle_request()
        if not result:
            raise SIWCError("Browser authorization timed out; run llm-pivot-siwc login again.")
        if result.get("error"):
            raise SIWCError("Browser authorization declined or failed.")
        if len(result.get("code", [])) != 1 or len(result.get("client_id", [client_id])) != 1:
            raise SIWCError("Incomplete OAuth callback.")
        issued = result.get("client_id", [client_id])[0]
        if not issued or issued == "dynamic_agent_client" or (client_id != "dynamic_agent_client" and issued != client_id):
            raise SIWCError("OAuth callback client ID mismatch or missing issued ID.")

        if client_id == "dynamic_agent_client":
            saved["client_id"] = issued
            store.write(saved)
        data = token_request(dict(grant_type="authorization_code", client_id=issued,
                                  code=result["code"][0], code_verifier=verifier, redirect_uri=redirect))
        identity = validate_identity(data.get("id_token", ""), issued, nonce)
        if saved.get("subject") and saved["subject"] != identity["sub"]:
            raise SIWCError("Returning account identity mismatch; existing credentials preserved.")
        record = merge_tokens({**saved, "client_id": issued, "subject": identity["sub"], "issuer": ISSUER},
                              data, initial=True)
        store.write(record)
        if PLAN_SCOPE not in record["scopes"]:
            raise SIWCError("Signed in; ChatGPT plan use is disabled. Run login --consent to enable it.")


def get_access_token(store=None, refresh_margin=60):
    store = store or CredentialStore()
    with store.locked():
        record = store.read()
        if record.get("app_name", store.app_name) != store.app_name:
            raise SIWCError("Configured app name differs from this credential registration.")
        if not record.get("access_token") or not record.get("subject"):
            raise SIWCError("Sign in first: llm-pivot-siwc login (use the same credentials directory).")
        if PLAN_SCOPE not in record.get("scopes", []):
            raise SIWCError("ChatGPT plan use is disabled. Run llm-pivot-siwc login --consent")
        if time.time() < record.get("expires_at", 0) - refresh_margin:
            return record["access_token"]
        try:
            data = token_request(dict(grant_type="refresh_token", client_id=record["client_id"],
                                      refresh_token=record["refresh_token"]))
        except SIWCError as exc:
            if any(f"code={code}." in str(exc) for code in TERMINAL_REFRESH):
                for field in ("access_token", "refresh_token", "id_token", "expires_at", "scopes"):
                    record.pop(field, None)
                store.write(record)
                raise SIWCError("Renewable session expired/revoked. Run llm-pivot-siwc login again.") from None
            raise
        record = merge_tokens(record, data)
        store.write(record)
        if PLAN_SCOPE not in record["scopes"]:
            raise SIWCError("Refreshed grant lacks ChatGPT plan permission; sign in with consent.")
        return record["access_token"]


class SIWCAuth:
    def __init__(self, config):
        self.config = config
        self.store = CredentialStore(config.credentials_dir, app_name=config.app_name)

    def login(self, consent=False):
        login(self.store, self.config, consent=consent)

    def access_token(self):
        return get_access_token(self.store, refresh_margin=self.config.refresh_margin)

    def status(self):
        with self.store.locked():
            record = self.store.read()
        return {"signed_in": bool(record.get("access_token") and record.get("subject")),
                "plan_permission": PLAN_SCOPE in record.get("scopes", [])}
