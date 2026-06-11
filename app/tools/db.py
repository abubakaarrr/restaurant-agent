"""Database tools — booking operations backed by PostgreSQL."""

import asyncpg
from datetime import datetime, timedelta
from langchain_core.tools import tool

from app.config import settings


async def _connect() -> asyncpg.Connection:
    return await asyncpg.connect(settings.database_url)


# ── Core DB operations (non-tool, called internally) ─────────

async def get_available_tables(date: str, time: str, party_size: int) -> list[dict]:
    """Find tables not already booked in a ±90-minute window around the requested slot."""
    dt = datetime.fromisoformat(f"{date}T{time}")
    window_start = dt - timedelta(minutes=30)
    window_end = dt + timedelta(minutes=90)

    conn = await _connect()
    try:
        rows = await conn.fetch(
            """
            SELECT t.id, t.table_number, t.capacity, t.location
            FROM tables t
            WHERE t.capacity >= $1
              AND t.id NOT IN (
                SELECT b.table_id FROM bookings b
                WHERE b.status = 'confirmed'
                  AND b.booked_at < $3
                  AND b.booked_at + (b.duration_mins * interval '1 minute') > $2
              )
            ORDER BY t.capacity ASC
            LIMIT 5
            """,
            party_size, window_start, window_end,
        )
        return [dict(r) for r in rows]
    finally:
        await conn.close()


async def get_alternative_slots(date: str, time: str, party_size: int) -> list[dict]:
    """Return up to 2 alternative slots near the requested time."""
    dt = datetime.fromisoformat(f"{date}T{time}")
    candidates = [-60, 60, -120, 120, -30, 30]
    results = []
    for delta in candidates:
        candidate = dt + timedelta(minutes=delta)
        available = await get_available_tables(
            candidate.date().isoformat(),
            candidate.strftime("%H:%M"),
            party_size,
        )
        if available:
            results.append({
                "time": candidate.strftime("%I:%M %p"),
                "date": candidate.date().isoformat(),
                "tables_available": len(available),
            })
        if len(results) == 2:
            break
    return results


async def create_booking_record(
    name: str,
    phone: str,
    date: str,
    time: str,
    party_size: int,
    notes: str,
    table_id: int,
) -> dict:
    """Atomically create a booking with row-level locking to prevent double-booking."""
    dt = datetime.fromisoformat(f"{date}T{time}")
    conn = await _connect()
    try:
        async with conn.transaction():
            await conn.fetchrow(
                "SELECT id FROM tables WHERE id = $1 FOR UPDATE",
                table_id,
            )
            conflict = await conn.fetchrow(
                """
                SELECT id FROM bookings
                WHERE table_id = $1
                  AND status = 'confirmed'
                  AND booked_at < $2::timestamp + interval '90 minutes'
                  AND booked_at + interval '90 minutes' > $2::timestamp
                """,
                table_id, dt,
            )
            if conflict:
                raise ValueError("That table just became unavailable. Please choose another slot.")

            row = await conn.fetchrow(
                """
                INSERT INTO bookings
                    (customer_name, customer_phone, table_id, booked_at, party_size, notes)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
                """,
                name, phone, table_id, dt, party_size, notes,
            )
            return {"booking_id": row["id"], "confirmed": True}
    finally:
        await conn.close()


# ── LangChain tool wrappers ───────────────────────────────────

@tool
async def check_table_availability(date: str, time: str, party_size: int) -> str:
    """Check if tables are available for a given date, time, and party size.
    date format: YYYY-MM-DD (e.g. 2026-06-07)
    time format: HH:MM in 24-hour (e.g. 19:00)
    party_size: integer number of guests
    Returns a summary of available tables, or suggests alternatives if none are free.
    """
    try:
        tables = await get_available_tables(date, time, party_size)
    except Exception as e:
        return f"Could not check availability: {e}"

    if tables:
        table_list = ", ".join(
            f"Table {t['table_number']} ({t['capacity']} seats, {t['location']})"
            for t in tables
        )
        return f"Available tables for {party_size} guests on {date} at {time}: {table_list}"

    # No tables — find alternatives
    alts = await get_alternative_slots(date, time, party_size)
    if alts:
        alt_text = " or ".join(f"{a['time']} ({a['tables_available']} table(s) free)" for a in alts)
        return (
            f"No tables available for {party_size} guests at {time} on {date}. "
            f"Nearest available slots: {alt_text}."
        )
    return f"No tables available for {party_size} guests on {date}. Please try a different day."


@tool
async def create_booking(
    name: str,
    phone: str,
    date: str,
    time: str,
    party_size: int,
    notes: str = "",
) -> str:
    """Confirm and save a table reservation in the database.
    Only call this after you have confirmed all details with the caller:
    name, date (YYYY-MM-DD), time (HH:MM), and party_size.
    phone is optional but recommended.
    Returns a booking confirmation message.
    """
    try:
        tables = await get_available_tables(date, time, party_size)
        if not tables:
            alts = await get_alternative_slots(date, time, party_size)
            if alts:
                alt_text = " or ".join(f"{a['time']}" for a in alts)
                return (
                    f"Sorry, no tables available at {time} on {date} for {party_size} guests. "
                    f"Available nearby: {alt_text}. Please confirm a new time with the caller."
                )
            return f"No tables available on {date} for {party_size} guests."

        # Pick the smallest table that fits
        table = tables[0]
        result = await create_booking_record(
            name=name,
            phone=phone,
            date=date,
            time=time,
            party_size=party_size,
            notes=notes,
            table_id=table["id"],
        )
        return (
            f"Booking confirmed! Booking ID: {result['booking_id']}. "
            f"Table {table['table_number']} ({table['location']}) reserved for {party_size} guests "
            f"on {date} at {time} under the name {name}. "
            f"Please note booking ID {result['booking_id']} for arrival, verification, or cancellation."
        )
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Booking failed due to a system error: {e}. Please try again."


@tool
async def lookup_booking(
    booking_id: int = 0,
    customer_name: str = "",
    customer_phone: str = "",
) -> str:
    """Look up an existing booking by ID, or by customer name (and optionally phone).
    Pass booking_id if the caller has it; otherwise pass customer_name.
    Returns booking details or a not-found message.
    """
    conn = await _connect()
    try:
        if booking_id:
            row = await conn.fetchrow(
                """SELECT id, customer_name, customer_phone, booked_at, party_size, status
                   FROM bookings WHERE id = $1""",
                booking_id,
            )
        elif customer_name:
            row = await conn.fetchrow(
                """SELECT id, customer_name, customer_phone, booked_at, party_size, status
                   FROM bookings
                   WHERE LOWER(customer_name) LIKE LOWER($1)
                     AND status = 'confirmed'
                   ORDER BY booked_at DESC LIMIT 1""",
                f"%{customer_name}%",
            )
        else:
            return "Please provide a booking ID or your name to look up your reservation."

        if not row:
            return "No confirmed booking found with those details. Please double-check your booking ID or name."

        booked_at = row["booked_at"]
        return (
            f"Booking #{row['id']} for {row['customer_name']}: "
            f"party of {row['party_size']}, "
            f"{booked_at.strftime('%A %d %B %Y at %I:%M %p')}. "
            f"Status: {row['status']}."
        )
    finally:
        await conn.close()


@tool
async def cancel_booking(booking_id: int, reason: str = "") -> str:
    """Cancel an existing confirmed booking by its ID.
    reason is optional — if the caller declines to provide one, pass an empty string.
    Returns a cancellation confirmation or an error message.
    """
    conn = await _connect()
    try:
        row = await conn.fetchrow(
            "SELECT id, customer_name, status FROM bookings WHERE id = $1",
            booking_id,
        )
        if not row:
            return f"Booking #{booking_id} was not found."
        if row["status"] == "cancelled":
            return f"Booking #{booking_id} is already cancelled."
        if row["status"] == "completed":
            return f"Booking #{booking_id} has already been completed and cannot be cancelled."

        await conn.execute(
            "UPDATE bookings SET status = 'cancelled', cancellation_reason = $1 WHERE id = $2",
            reason or "No reason provided",
            booking_id,
        )
        return (
            f"Booking #{booking_id} for {row['customer_name']} has been successfully cancelled. "
            f"We hope to welcome you another time."
        )
    finally:
        await conn.close()


@tool
async def add_order_item(
    session_id: str,
    item_name: str,
    quantity: int = 1,
    notes: str = "",
    booking_id: int = 0,
    customer_name: str = "",
) -> str:
    """Add one menu item to the order for this call session.
    Call this every time the caller names a dish or drink they want to order.
    item_name must closely match a real menu item name.
    quantity defaults to 1.
    notes is for special instructions like 'medium rare' or 'no onions'.
    Returns confirmation of what was added and the running total.
    """
    conn = await _connect()
    try:
        # First try full phrase match
        menu_row = await conn.fetchrow(
            "SELECT id, name, price, available FROM menu_items WHERE LOWER(name) LIKE LOWER($1) LIMIT 1",
            f"%{item_name}%",
        )
        # If not found, try matching any single word from the item name
        if not menu_row:
            words = [w for w in item_name.split() if len(w) > 3]
            for word in words:
                menu_row = await conn.fetchrow(
                    "SELECT id, name, price, available FROM menu_items WHERE LOWER(name) LIKE LOWER($1) LIMIT 1",
                    f"%{word}%",
                )
                if menu_row:
                    break
        if not menu_row:
            # Return available drinks/items for agent to suggest alternatives
            similar = await conn.fetch(
                "SELECT name FROM menu_items WHERE available = TRUE ORDER BY category, name LIMIT 8"
            )
            suggestions = ", ".join(r["name"] for r in similar)
            return (
                f"I couldn't find '{item_name}' on our menu. "
                f"Please ask the customer to clarify — some available options are: {suggestions}. "
                f"Do not say there is a glitch; just ask the customer what they'd like."
            )
        if not menu_row["available"]:
            return f"Sorry, {menu_row['name']} is currently unavailable. Can I suggest something else from the menu?"

        # Find or create pending order for this session
        order_row = await conn.fetchrow(
            "SELECT id FROM orders WHERE session_id = $1 AND status = 'pending' LIMIT 1",
            session_id,
        )
        if order_row:
            order_id = order_row["id"]
        else:
            new_order = await conn.fetchrow(
                """INSERT INTO orders (session_id, booking_id, customer_name, status)
                   VALUES ($1, $2, $3, 'pending') RETURNING id""",
                session_id,
                booking_id if booking_id else None,
                customer_name or "",
            )
            order_id = new_order["id"]

        await conn.execute(
            """INSERT INTO order_items (order_id, menu_item_id, item_name, quantity, unit_price, notes)
               VALUES ($1, $2, $3, $4, $5, $6)""",
            order_id, menu_row["id"], menu_row["name"], quantity, menu_row["price"], notes,
        )

        total_row = await conn.fetchrow(
            "SELECT COALESCE(SUM(subtotal), 0) AS total FROM order_items WHERE order_id = $1",
            order_id,
        )
        running_total = float(total_row["total"])

        return (
            f"Added {quantity}x {menu_row['name']} (${float(menu_row['price']):.2f} each). "
            f"Running total: ${running_total:.2f}. Would you like anything else?"
        )
    finally:
        await conn.close()


@tool
async def confirm_order(session_id: str) -> str:
    """Finalise and confirm the complete order for this call session.
    Call this ONLY when the caller explicitly says they are done ordering.
    Returns a full itemised order summary with total.
    """
    conn = await _connect()
    try:
        order_row = await conn.fetchrow(
            "SELECT id, booking_id FROM orders WHERE session_id = $1 AND status = 'pending' LIMIT 1",
            session_id,
        )
        if not order_row:
            return "No pending order found for this session."

        order_id = order_row["id"]
        is_dine_in = order_row["booking_id"] is not None
        items = await conn.fetch(
            "SELECT item_name, quantity, unit_price, subtotal, notes FROM order_items WHERE order_id = $1 ORDER BY id",
            order_id,
        )
        if not items:
            return "The order is empty — nothing to confirm."

        total_row = await conn.fetchrow(
            "SELECT COALESCE(SUM(subtotal), 0) AS total FROM order_items WHERE order_id = $1",
            order_id,
        )
        total = float(total_row["total"])

        await conn.execute(
            "UPDATE orders SET status = 'confirmed', total_amount = $1 WHERE id = $2",
            total, order_id,
        )

        lines = []
        for item in items:
            line = f"  - {item['quantity']}x {item['item_name']} — ${float(item['subtotal']):.2f}"
            if item["notes"]:
                line += f" ({item['notes']})"
            lines.append(line)

        # Dine-in pre-orders are served at the table; standalone orders are for pickup.
        if is_dine_in:
            timing = (
                f"Your order has been sent to the kitchen and will be freshly prepared "
                f"and served at your table when you arrive."
            )
        else:
            timing = (
                f"Your order has been sent to the kitchen and will be ready for pickup "
                f"in about 30 minutes."
            )

        return (
            f"Order confirmed! Order ID: {order_id}. Here is your full order:\n" + "\n".join(lines) +
            f"\nTotal: ${total:.2f}. {timing} "
            f"Please note order ID {order_id} for verification or changes. "
            f"Is there anything else I can help you with?"
        )
    finally:
        await conn.close()


@tool
async def lookup_order(order_id: int = 0, customer_name: str = "") -> str:
    """Look up an existing confirmed order by its order ID (and optionally verify by name).
    Use this when a caller asks about an existing order — e.g. "when will my order be ready?"
    or "what's the status of my order?". Always ask for the order ID and name first.
    Returns the order details, status, and estimated pickup time.
    """
    if not order_id:
        return "Please ask the customer for their order ID so you can look it up."

    conn = await _connect()
    try:
        order = await conn.fetchrow(
            """SELECT id, customer_name, status, total_amount, created_at
               FROM orders WHERE id = $1""",
            order_id,
        )
        if not order:
            return f"No order found with ID {order_id}. Please double-check the order number."

        # Optional name verification
        if customer_name and order["customer_name"]:
            if customer_name.strip().lower() not in order["customer_name"].strip().lower() \
               and order["customer_name"].strip().lower() not in customer_name.strip().lower():
                return (
                    f"The name doesn't match our records for order #{order_id}. "
                    f"Please confirm the name on the order."
                )

        items = await conn.fetch(
            "SELECT item_name, quantity FROM order_items WHERE order_id = $1 ORDER BY id",
            order_id,
        )
        item_summary = ", ".join(f"{i['quantity']}x {i['item_name']}" for i in items) or "no items"

        # Estimate pickup time: 30 minutes after the order was created
        created = order["created_at"]
        ready_at = created + timedelta(minutes=30)
        now = datetime.now()
        mins_left = int((ready_at - now).total_seconds() // 60)

        if order["status"] == "cancelled":
            timing = "This order has been cancelled."
        elif mins_left > 0:
            timing = f"It will be ready for pickup in about {mins_left} minute(s), at {ready_at.strftime('%I:%M %p')}."
        else:
            timing = "It should be ready for pickup now."

        return (
            f"Order #{order['id']} for {order['customer_name'] or 'guest'}: {item_summary}. "
            f"Total ${float(order['total_amount']):.2f}. Status: {order['status']}. {timing}"
        )
    finally:
        await conn.close()


@tool
async def get_full_menu() -> str:
    """Return the complete current menu with real item names and prices, grouped by category.
    Use this whenever a caller asks for the menu, asks what's available, or asks what you serve.
    Always use this instead of guessing — never invent menu items or prices.
    """
    conn = await _connect()
    try:
        rows = await conn.fetch(
            """SELECT name, category, price, dietary, available
               FROM menu_items
               WHERE available = TRUE
               ORDER BY
                 CASE category
                   WHEN 'starter' THEN 1
                   WHEN 'main' THEN 2
                   WHEN 'dessert' THEN 3
                   WHEN 'drink' THEN 4
                   ELSE 5
                 END,
                 name"""
        )
        if not rows:
            return "The menu is currently empty."

        category_labels = {
            "starter": "Starters",
            "main": "Mains",
            "dessert": "Desserts",
            "drink": "Drinks",
        }
        grouped: dict[str, list[str]] = {}
        for r in rows:
            label = category_labels.get(r["category"], r["category"].title())
            line = f"{r['name']} - ${float(r['price']):.2f}"
            grouped.setdefault(label, []).append(line)

        parts = []
        for label in ["Starters", "Mains", "Desserts", "Drinks"]:
            if label in grouped:
                parts.append(f"{label}: " + "; ".join(grouped[label]))
        return "Here is our current menu. " + " | ".join(parts)
    finally:
        await conn.close()


@tool
async def check_menu_item_availability(item_name: str) -> str:
    """Check whether a specific menu item is currently available (not sold out).
    Use this when a caller asks 'do you still have X?' or 'is X available tonight?'
    """
    conn = await _connect()
    try:
        row = await conn.fetchrow(
            "SELECT name, available FROM menu_items WHERE LOWER(name) LIKE LOWER($1) LIMIT 1",
            f"%{item_name}%",
        )
        if not row:
            return f"'{item_name}' was not found in our menu system."
        status = "currently available" if row["available"] else "currently unavailable (sold out)"
        return f"{row['name']} is {status}."
    finally:
        await conn.close()
