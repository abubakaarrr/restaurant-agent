"""Wipe demo bookings/orders for a session or phone so rehearsal can restart clean."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import asyncpg

from app.agent.runner import clear_session
from app.call_memory import clear_call_memory


def _rowcount(status: str) -> int:
    try:
        return int(str(status).split()[-1])
    except (TypeError, ValueError, IndexError):
        return 0


async def reset(*, session_id: str = "", phone: str = "") -> dict[str, int]:
    if not session_id and not phone:
        raise ValueError("Pass --session-id and/or --phone")
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    connection = await asyncpg.connect(database_url)
    try:
        booking_ids: list[int] = []
        if phone:
            rows = await connection.fetch(
                "SELECT id FROM bookings WHERE customer_phone = $1",
                phone,
            )
            booking_ids.extend(int(row["id"]) for row in rows)
        if session_id:
            state_rows = await connection.fetch(
                """
                SELECT (state->>'booking_id') AS booking_id
                FROM call_sessions
                WHERE session_id = $1
                """,
                session_id,
            )
            for row in state_rows:
                raw = row["booking_id"]
                if raw and str(raw).isdigit():
                    booking_ids.append(int(raw))
            order_rows = await connection.fetch(
                "SELECT booking_id FROM orders WHERE session_id = $1",
                session_id,
            )
            booking_ids.extend(
                int(row["booking_id"])
                for row in order_rows
                if row["booking_id"]
            )
        booking_ids = sorted(set(booking_ids))

        deleted_items = await connection.execute(
            """
            DELETE FROM order_items
            WHERE order_id IN (
                SELECT id FROM orders
                WHERE ($1 <> '' AND session_id = $1)
                   OR (CARDINALITY($2::int[]) > 0 AND booking_id = ANY($2::int[]))
            )
            """,
            session_id or "",
            booking_ids,
        )
        deleted_orders = await connection.execute(
            """
            DELETE FROM orders
            WHERE ($1 <> '' AND session_id = $1)
               OR (CARDINALITY($2::int[]) > 0 AND booking_id = ANY($2::int[]))
            """,
            session_id or "",
            booking_ids,
        )
        deleted_bookings = await connection.execute(
            """
            DELETE FROM bookings
            WHERE id = ANY($1::int[])
               OR ($2 <> '' AND customer_phone = $2)
            """,
            booking_ids,
            phone or "",
        )
        deleted_sessions = await connection.execute(
            """
            DELETE FROM call_sessions
            WHERE ($1 <> '' AND session_id = $1)
               OR ($2 <> '' AND caller_phone = $2)
            """,
            session_id or "",
            phone or "",
        )
        deleted_idempotency = await connection.execute(
            """
            DELETE FROM voice_action_idempotency
            WHERE $1 <> '' AND call_id = $1
            """,
            session_id or "",
        )
        counts = {
            "order_items": _rowcount(deleted_items),
            "orders": _rowcount(deleted_orders),
            "bookings": _rowcount(deleted_bookings),
            "call_sessions": _rowcount(deleted_sessions),
            "idempotency": _rowcount(deleted_idempotency),
        }
    finally:
        await connection.close()

    if session_id:
        clear_session(session_id)
        clear_call_memory(session_id)
    return counts


async def hygiene() -> dict[str, int]:
    """Remove eval leftovers so the demo menu and floor plan stay clean."""
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    connection = await asyncpg.connect(database_url)
    try:
        await connection.execute(
            "DELETE FROM bookings WHERE customer_name LIKE 'PatioBlocker%'"
        )
        extra_tables = await connection.fetch(
            "SELECT id FROM tables WHERE table_number >= 900"
        )
        extra_ids = [int(row["id"]) for row in extra_tables]
        if extra_ids:
            await connection.execute(
                "DELETE FROM bookings WHERE table_id = ANY($1::int[])",
                extra_ids,
            )
            await connection.execute(
                "DELETE FROM tables WHERE id = ANY($1::int[])",
                extra_ids,
            )
        hidden = await connection.execute(
            """
            UPDATE menu_items
            SET available = FALSE
            WHERE name ILIKE 'Eval %'
               OR LOWER(name) IN ('margherita pizza', 'garden salad', 'tomato basil soup')
            """
        )
        from db.seed import seed

        await seed(connection, with_embeddings=False)
        try:
            from app.services.restaurant import restaurant_service

            restaurant_service._seating_limits = None
        except Exception:
            pass
        return {
            "extra_tables": len(extra_ids),
            "hidden_menu": _rowcount(hidden),
        }
    finally:
        await connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", default="", help="Retell call id or /chat session id")
    parser.add_argument("--phone", default="", help="Customer phone as stored on bookings")
    parser.add_argument(
        "--hygiene",
        action="store_true",
        help="Remove eval tables/menu leftovers and reseed the live menu",
    )
    args = parser.parse_args()
    if args.hygiene:
        print("hygiene", asyncio.run(hygiene()))
        if not args.session_id and not args.phone:
            return 0
    if not args.session_id and not args.phone:
        parser.error("Pass --session-id and/or --phone, or --hygiene")
    counts = asyncio.run(reset(session_id=args.session_id, phone=args.phone))
    print("reset", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
