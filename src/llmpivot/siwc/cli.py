import argparse
import logging
import sys


def model_catalog(auth):
    import httpx
    from .auth import RESOURCE, SIWCError, check_http
    token = auth.access_token()
    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(RESOURCE + "/models", headers={"Authorization": f"Bearer {token}"})
        check_http(response, "Model catalog")
        data = response.json().get("models")
        if not isinstance(data, list):
            raise SIWCError("Unexpected SIWC model catalog format.")
        return [m for m in data if m.get("visibility") == "list" and m.get("slug")]
    except httpx.RequestError:
        raise SIWCError("Model catalog network failure; retry later.") from None


def smoke(config, auth):
    from llmpivot import LLMPivot
    from .auth import SIWCError
    models = model_catalog(auth)
    if config.model_id not in [m["slug"] for m in models]:
        raise SIWCError("Configured model is absent from this account's visible catalog.")
    print(f"Using ChatGPT plan; model={config.model_id}. One inference request; no retries.", flush=True)
    with LLMPivot(config) as llm:
        result = llm.generate([{"role": "user", "content": "Reply with exactly: SIWC_OK"}])
    if result["finish_reason"] != "completed" or (result["content"] or "").strip() != "SIWC_OK":
        raise SIWCError("Smoke did not return a completed SIWC_OK result.")
    print("PASS: response.completed and SIWC_OK.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sign in with ChatGPT for your configured research app")
    parser.add_argument("--config", required=True, help="PivotConfig JSON file")
    commands = parser.add_subparsers(dest="command", required=True)
    signin = commands.add_parser("login", help="Continue with ChatGPT in system browser")
    signin.add_argument("--consent", action="store_true", help="Explicitly request ChatGPT plan-use consent again")
    signin.add_argument("--smoke", action="store_true", help="Run one inference after authorization")
    commands.add_parser("smoke", help="One inference using saved authorization")
    commands.add_parser("models", help="List account-visible model slugs")
    commands.add_parser("status", help="Show credential-free connection state")
    args = parser.parse_args(argv)
    for name in ("openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        from .auth import SIWCAuth, SIWCError
    except ImportError:
        print("Install SIWC support: pip install 'aq-llm-pivot[siwc]'", file=sys.stderr)
        return 1
    try:
        from ..config import PivotConfig
        config = PivotConfig.from_file(args.config)
        if config.api_type != "siwc" or config.model_type != "online":
            raise SIWCError("SIWC requires model_type=online and api_type=siwc in config.")
        auth = SIWCAuth(config.siwc)
        if args.command == "login":
            print("OAuth credentials and verified account identity are stored locally. "
                  "Experiment inputs are sent to OpenAI when you request inference. "
                  "Manage usage/access: https://chatgpt.com/settings/usage", flush=True)
            auth.login(consent=args.consent)
            print("Signed in. ChatGPT plan use enabled.")
            if args.smoke:
                smoke(config, auth)
        elif args.command == "smoke":
            smoke(config, auth)
        elif args.command == "models":
            for model in model_catalog(auth):
                print(model["slug"])
        elif args.command == "status":
            status = auth.status()
            print(f"Signed in: {status['signed_in']}; plan permission: {status['plan_permission']}")
    except SIWCError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except Exception:
        print("Local operation failed. Check config, permissions, dependencies and network.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
