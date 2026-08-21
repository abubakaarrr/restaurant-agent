-- Explicit order fulfillment; nullable until set on first pending create.

BEGIN;

ALTER TABLE orders
    ADD COLUMN IF NOT EXISTS fulfillment_type TEXT;

ALTER TABLE orders
    DROP CONSTRAINT IF EXISTS orders_fulfillment_type_check;

ALTER TABLE orders
    ADD CONSTRAINT orders_fulfillment_type_check
    CHECK (
        fulfillment_type IS NULL
        OR fulfillment_type IN ('dine_in', 'pickup')
    );

COMMIT;
