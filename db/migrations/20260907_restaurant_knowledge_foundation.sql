-- Phase 1 synthetic Harbor & Hearth knowledge and canonical order readbacks.

ALTER TABLE menu_items ADD COLUMN IF NOT EXISTS canonical_id TEXT;
ALTER TABLE menu_items ADD COLUMN IF NOT EXISTS aliases TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE menu_items ADD COLUMN IF NOT EXISTS ingredients TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE menu_items ADD COLUMN IF NOT EXISTS allergens TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE menu_items ADD COLUMN IF NOT EXISTS service_periods TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE menu_items ADD COLUMN IF NOT EXISTS availability_status TEXT NOT NULL DEFAULT 'available';
ALTER TABLE menu_items ADD COLUMN IF NOT EXISTS knowledge_metadata JSONB NOT NULL DEFAULT '{}';
ALTER TABLE menu_items ADD COLUMN IF NOT EXISTS source_id TEXT NOT NULL DEFAULT '';
ALTER TABLE menu_items ADD COLUMN IF NOT EXISTS data_version TEXT NOT NULL DEFAULT '';
ALTER TABLE menu_items ADD COLUMN IF NOT EXISTS effective_from DATE;
ALTER TABLE menu_items ADD COLUMN IF NOT EXISTS effective_to DATE;
CREATE UNIQUE INDEX IF NOT EXISTS uq_menu_items_canonical_id
    ON menu_items (canonical_id) WHERE canonical_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS restaurant_knowledge_records (
    canonical_id   TEXT PRIMARY KEY,
    record_type    TEXT NOT NULL,
    category_id    TEXT NOT NULL,
    source_id      TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    data_version   TEXT NOT NULL,
    effective_from DATE NOT NULL,
    effective_to   DATE,
    status          TEXT NOT NULL DEFAULT 'current',
    display_text    TEXT NOT NULL DEFAULT '',
    payload         JSONB NOT NULL,
    synthetic       BOOLEAN NOT NULL DEFAULT TRUE,
    CHECK (synthetic IS TRUE)
);
CREATE INDEX IF NOT EXISTS idx_restaurant_knowledge_category
    ON restaurant_knowledge_records(record_type, category_id);
CREATE INDEX IF NOT EXISTS idx_restaurant_knowledge_effective
    ON restaurant_knowledge_records(effective_from, effective_to);

ALTER TABLE orders ADD COLUMN IF NOT EXISTS fulfillment_details JSONB NOT NULL DEFAULT '{}';
ALTER TABLE orders ADD COLUMN IF NOT EXISTS allergy_notes TEXT DEFAULT '';
ALTER TABLE orders DROP CONSTRAINT IF EXISTS orders_fulfillment_type_check;
ALTER TABLE orders ADD CONSTRAINT orders_fulfillment_type_check
    CHECK (fulfillment_type IS NULL OR fulfillment_type IN ('dine_in', 'pickup', 'delivery'));

ALTER TABLE order_items ADD COLUMN IF NOT EXISTS modifiers JSONB NOT NULL DEFAULT '[]';
ALTER TABLE order_items ADD COLUMN IF NOT EXISTS removals TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE order_items ADD COLUMN IF NOT EXISTS substitutions JSONB NOT NULL DEFAULT '[]';

-- Operator-authored answers from a prior example identity must never be
-- returned under Harbor & Hearth. Existing unscoped rows remain preserved for
-- audit/history but are invisible to the canonical restaurant query path.
ALTER TABLE operator_knowledge ADD COLUMN IF NOT EXISTS restaurant_id TEXT NOT NULL DEFAULT '';
DROP INDEX IF EXISTS uq_operator_knowledge_question_ci;
CREATE UNIQUE INDEX IF NOT EXISTS uq_operator_knowledge_restaurant_question_ci
    ON operator_knowledge (restaurant_id, LOWER(question));

ALTER TABLE knowledge_gaps ADD COLUMN IF NOT EXISTS restaurant_id TEXT;
DROP INDEX IF EXISTS uq_knowledge_gaps_session_question;
CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_gaps_restaurant_session_question
    ON knowledge_gaps (restaurant_id, session_id, question_normalized);
