-- Weekend-demo gates: patio is inventory, paid-item approval is a flag,
-- proposed order lines are not kitchen-committed.

BEGIN;

ALTER TABLE bookings
    ADD COLUMN IF NOT EXISTS require_approval_for_paid_items BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE order_items
    ADD COLUMN IF NOT EXISTS proposed BOOLEAN NOT NULL DEFAULT FALSE;

COMMIT;
