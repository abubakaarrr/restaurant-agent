-- Production voice pilot: durable call telemetry and idempotent mutations.
-- Apply once to an existing database:
--   psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f db/migrations/20260811_voice_pilot.sql

BEGIN;

ALTER TABLE call_sessions
    ADD COLUMN IF NOT EXISTS provider TEXT DEFAULT '',
    ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS behavior_state JSONB DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW();

ALTER TABLE orders
    ADD COLUMN IF NOT EXISTS draft_version INT NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS confirmed_at TIMESTAMP;

CREATE TABLE IF NOT EXISTS voice_action_idempotency (
    id                  BIGSERIAL PRIMARY KEY,
    action              TEXT NOT NULL,
    idempotency_key     TEXT NOT NULL,
    call_id             TEXT NOT NULL,
    request_hash        TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'processing',
    response            JSONB,
    created_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMP,
    UNIQUE (action, idempotency_key),
    CHECK (status IN ('processing', 'completed'))
);

CREATE INDEX IF NOT EXISTS idx_voice_action_call
    ON voice_action_idempotency(call_id, created_at DESC);

CREATE TABLE IF NOT EXISTS call_events (
    id                  BIGSERIAL PRIMARY KEY,
    call_id             TEXT NOT NULL,
    provider            TEXT NOT NULL DEFAULT 'retell',
    event_type          TEXT NOT NULL,
    response_id         BIGINT,
    duration_ms         INT,
    payload             JSONB DEFAULT '{}',
    created_at          TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_call_events_call
    ON call_events(call_id, created_at);
CREATE INDEX IF NOT EXISTS idx_call_events_type
    ON call_events(event_type, created_at DESC);

CREATE TABLE IF NOT EXISTS provider_webhook_events (
    provider            TEXT NOT NULL,
    event_id            TEXT NOT NULL,
    event_type          TEXT NOT NULL,
    call_id             TEXT NOT NULL DEFAULT '',
    payload             JSONB NOT NULL DEFAULT '{}',
    received_at         TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (provider, event_id)
);

-- One mutable order draft per call. Abort instead of silently deleting data if
-- an old deployment already created duplicate pending orders.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM orders
        WHERE status = 'pending'
        GROUP BY session_id
        HAVING COUNT(*) > 1
    ) THEN
        RAISE EXCEPTION
            'Duplicate pending orders exist; resolve them before applying the voice pilot migration';
    END IF;
END
$$;

CREATE UNIQUE INDEX IF NOT EXISTS uq_orders_one_pending_per_session
    ON orders(session_id)
    WHERE status = 'pending';

COMMIT;
