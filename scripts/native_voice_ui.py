"""Start the native voice browser harness against an explicit test database."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-env", type=Path, required=True, help="Read only OPENAI_API_KEY from this file")
    parser.add_argument("--database-env", type=Path, required=True, help="Read native test database URL and marker")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--test-clock", help="Optional ISO local restaurant time, shown prominently in the UI")
    args = parser.parse_args()
    from dotenv import dotenv_values
    key = dotenv_values(args.key_env).get("OPENAI_API_KEY")
    database = dotenv_values(args.database_env)
    if not key or not all(database.get(k) for k in ("NATIVE_VOICE_DATABASE_URL", "NATIVE_VOICE_DATABASE_MARKER")):
        parser.error("Key or marked test database configuration missing")
    os.environ.update(
        OPENAI_API_KEY=key,
        APP_ENV="development",
        DATABASE_URL="postgresql://disabled:disabled@127.0.0.1:1/disabled",
        NATIVE_VOICE_REALTIME_ENABLED="true",
        NATIVE_VOICE_REALTIME_MODEL="gpt-realtime",
        NATIVE_VOICE_DATABASE_WRITE_ENABLED="true",
        VOICE_LIVE_WRITES_ENABLED="true",
        NATIVE_VOICE_DATABASE_URL=database["NATIVE_VOICE_DATABASE_URL"],
        NATIVE_VOICE_DATABASE_MARKER=database["NATIVE_VOICE_DATABASE_MARKER"],
    )
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if args.test_clock:
        from zoneinfo import ZoneInfo
        from app.restaurant_knowledge import get_restaurant_knowledge
        import app.services.restaurant as service
        zone = ZoneInfo(get_restaurant_knowledge().identity["timezone"])
        moment = datetime.fromisoformat(args.test_clock)
        if moment.tzinfo is not None:
            parser.error("test-clock must be local restaurant time without offset")
        moment = moment.replace(tzinfo=zone)
        service._restaurant_now = lambda: moment
        os.environ["NATIVE_VOICE_UI_TEST_CLOCK"] = moment.strftime("%a %d %b %Y, %I:%M %p %Z (fixed test clock)")
    else:
        os.environ.pop("NATIVE_VOICE_UI_TEST_CLOCK", None)
    from app.native_voice.ui_server import create_app
    import uvicorn
    print(f"Open http://localhost:{args.port} — native voice, separate test database", flush=True)
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, proxy_headers=False, ws_max_size=3_000_000, access_log=False)


if __name__ == "__main__":
    main()
