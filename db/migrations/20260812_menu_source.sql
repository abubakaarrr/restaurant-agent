-- Enforce one live menu record per case-insensitive item name.

BEGIN;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM menu_items
        GROUP BY LOWER(name)
        HAVING COUNT(*) > 1
    ) THEN
        RAISE EXCEPTION
            'Duplicate case-insensitive menu names exist; merge them before applying this migration';
    END IF;
END
$$;

CREATE UNIQUE INDEX IF NOT EXISTS uq_menu_items_name_ci
    ON menu_items (LOWER(name));

COMMIT;
