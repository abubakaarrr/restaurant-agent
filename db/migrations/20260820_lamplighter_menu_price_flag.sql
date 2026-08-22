-- Estimated menu prices must stay distinguishable from venue-confirmed prices.

BEGIN;

ALTER TABLE menu_items
    ADD COLUMN IF NOT EXISTS price_estimated BOOLEAN NOT NULL DEFAULT FALSE;

COMMIT;
