import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from llmpivot.siwc.auth import (CredentialStore, SIWCError, PLAN_SCOPE, get_access_token,
                                merge_tokens, validate_identity)


def record():
    return dict(client_id="oaiapp_test", subject="test-subject", ext_agent_host_id="urn:uuid:test",
                access_token="old-fake-access", refresh_token="old-fake-refresh",
                expires_at=0, scopes=[PLAN_SCOPE])


def refresh_worker(directory, queue):
    def fake_request(data):
        assert data["client_id"] == "oaiapp_test"
        assert data["refresh_token"] == "old-fake-refresh"
        with open(Path(directory) / "calls", "a") as handle:
            handle.write("refresh\n")
        time.sleep(0.1)
        return dict(access_token="new-fake-access", refresh_token="new-fake-refresh",
                    token_type="Bearer", expires_in=3600)
    with patch("llmpivot.siwc.auth.token_request", fake_request):
        queue.put(get_access_token(CredentialStore(directory)))


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CredentialStore(self.tmp.name)
        with self.store.locked():
            self.store.write(record())

    def tearDown(self):
        self.tmp.cleanup()

    def test_atomic_owner_only(self):
        self.assertEqual(os.stat(self.store.directory).st_mode & 0o777, 0o700)
        self.assertEqual(os.stat(self.store.path).st_mode & 0o777, 0o600)
        self.assertEqual(os.stat(self.store.directory / "credentials.lock").st_mode & 0o777, 0o600)
        self.assertFalse(list(self.store.directory.glob(".credentials-*")))

    def test_multiprocess_refresh_rotates_once(self):
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        processes = [context.Process(target=refresh_worker, args=(self.tmp.name, queue)) for _ in range(6)]
        for process in processes:
            process.start()
        try:
            values = [queue.get(timeout=10) for _ in processes]
            for process in processes:
                process.join(timeout=5)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(values, ["new-fake-access"] * 6)
            self.assertEqual((self.store.directory / "calls").read_text().splitlines(), ["refresh"])
            self.assertEqual(self.store.read()["refresh_token"], "new-fake-refresh")
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()
            queue.close()

    def test_transient_failure_preserves_credentials(self):
        before = self.store.path.read_bytes()
        with patch("llmpivot.siwc.auth.token_request", side_effect=SIWCError("HTTP 503, code=unavailable.")):
            with self.assertRaises(SIWCError):
                get_access_token(self.store)
        self.assertEqual(before, self.store.path.read_bytes())

    def test_terminal_refresh_clears_tokens_retains_registration(self):
        with patch("llmpivot.siwc.auth.token_request", side_effect=SIWCError("code=invalid_grant.")):
            with self.assertRaisesRegex(SIWCError, "login"):
                get_access_token(self.store)
        saved = self.store.read()
        self.assertNotIn("refresh_token", saved)
        self.assertNotIn("access_token", saved)
        self.assertEqual(saved["client_id"], "oaiapp_test")
        self.assertEqual(saved["ext_agent_host_id"], "urn:uuid:test")

    def test_missing_scope_blocks_inference_without_refresh(self):
        with self.store.locked():
            self.store.write({**record(), "scopes": []})
        with patch("llmpivot.siwc.auth.token_request") as request:
            with self.assertRaisesRegex(SIWCError, "disabled"):
                get_access_token(self.store)
            request.assert_not_called()

    def test_incomplete_rotation_not_committed(self):
        before = self.store.path.read_bytes()
        with patch("llmpivot.siwc.auth.token_request", return_value={"access_token": "fake"}):
            with self.assertRaises(SIWCError):
                get_access_token(self.store)
        self.assertEqual(before, self.store.path.read_bytes())

    def test_initial_scopes_required(self):
        with self.assertRaisesRegex(SIWCError, "scopes"):
            merge_tokens({}, dict(access_token="fake", refresh_token="fake", token_type="Bearer", expires_in=10), initial=True)

    def test_signed_identity_checks(self):
        private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
        jwk.update(kid="test", alg="RS256", use="sig")
        now = int(time.time())
        claims = dict(sub="test", iss="https://auth.openai.com", aud="oaiapp_test",
                      exp=now+60, iat=now, nonce="nonce")
        def validate(payload, expected_nonce="nonce", signing_key=private):
            encoded = jwt.encode(payload, signing_key, algorithm="RS256", headers={"kid": "test"})
            response = httpx.Response(200, json={"keys": [jwk]})
            with patch("httpx.Client.get", return_value=response):
                return validate_identity(encoded, "oaiapp_test", expected_nonce)
        self.assertEqual(validate(claims)["sub"], "test")
        for field, value in [("iss", "https://example.com"), ("aud", "other"), ("exp", now-60), ("nonce", "other")]:
            with self.subTest(field=field), self.assertRaises(SIWCError):
                validate({**claims, field: value})
        with self.assertRaises(SIWCError):
            validate(claims, signing_key=rsa.generate_private_key(public_exponent=65537, key_size=2048))


def test_login_uses_config_pkce_and_rejects_invalid_state(tmp_path):
    import base64
    import hashlib
    import io
    from urllib.parse import parse_qs, urlsplit
    from unittest.mock import Mock
    from llmpivot import SIWCConfig, SIWCAuth
    config = SIWCConfig(app_name="Configured Research", credentials_dir=str(tmp_path), login_timeout=10)
    auth = SIWCAuth(config)
    query = {}
    statuses = []
    class Listener:
        server_port = 12345
        def __init__(self, address, handler):
            assert address == ("127.0.0.1", 0)
            self.handler = handler
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def handle_request(self):
            callback = object.__new__(self.handler)
            state = "invalid" if not statuses else query["state"][0]
            callback.path = f"/auth/callback?state={state}&code=fake-code&client_id=oaiapp_new"
            callback.send_response = statuses.append
            callback.send_header = Mock()
            callback.end_headers = Mock()
            callback.wfile = io.BytesIO()
            callback.do_GET()
    def browser(url):
        query.update(parse_qs(urlsplit(url).query))
        return True
    def exchange(data):
        assert data["redirect_uri"] == "http://127.0.0.1:12345/auth/callback"
        assert data["client_id"] == "oaiapp_new"
        challenge = base64.urlsafe_b64encode(hashlib.sha256(data["code_verifier"].encode()).digest()).rstrip(b"=").decode()
        assert challenge == query["code_challenge"][0]
        return dict(access_token="fake-access", refresh_token="fake-refresh", id_token="fake-id",
                    token_type="Bearer", expires_in=3600, scope=PLAN_SCOPE)
    with patch("llmpivot.siwc.auth.HTTPServer", Listener), \
         patch("llmpivot.siwc.auth.webbrowser.open", side_effect=browser), \
         patch("llmpivot.siwc.auth.token_request", side_effect=exchange), \
         patch("llmpivot.siwc.auth.validate_identity", return_value={"sub": "subject"}) as identity:
        auth.login()
    assert statuses == [400, 200]
    assert query["agent_name_hint"] == ["Configured Research"]
    assert query["client_id"] == ["dynamic_agent_client"]
    assert query["code_challenge_method"] == ["S256"]
    identity.assert_called_once_with("fake-id", "oaiapp_new", query["nonce"][0])
    assert auth.store.read()["client_id"] == "oaiapp_new"
    assert auth.status() == {"signed_in": True, "plan_permission": True}
