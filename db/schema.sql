-- ══════════════════════════════════════════════════════════════
-- Harbor & Hearth Kitchen — Database Schema
-- Run: psql restaurant_agent -f db/schema.sql
-- ══════════════════════════════════════════════════════════════

-- Enable pgvector for RAG embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- ─────────────────────────── RAG / Knowledge ─────────────────
-- Stores embedded chunks from menu.md, slots.md, restaurant_info.md
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id          SERIAL PRIMARY KEY,
    source      TEXT NOT NULL,          -- 'menu' | 'slots' | 'info'
    content     TEXT NOT NULL,
    embedding   vector(1536),           -- OpenAI text-embedding-3-small
    metadata    JSONB DEFAULT '{}'
);

-- Cosine similarity index for fast nearest-neighbour search
CREATE INDEX IF NOT EXISTS idx_knowledge_embedding
    ON knowledge_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

-- ─────────────────────────── Tables ──────────────────────────
CREATE TABLE IF NOT EXISTS tables (
    id            SERIAL PRIMARY KEY,
    table_number  INT UNIQUE NOT NULL,
    capacity      INT NOT NULL,
    location      TEXT DEFAULT 'main'   -- 'main' | 'patio' | 'private'
);

-- ─────────────────────────── Bookings ────────────────────────
CREATE TABLE IF NOT EXISTS bookings (
    id              SERIAL PRIMARY KEY,
    customer_name   TEXT NOT NULL,
    customer_phone  TEXT NOT NULL DEFAULT '',
    table_id        INT REFERENCES tables(id),
    booked_at       TIMESTAMP NOT NULL,
    party_size      INT NOT NULL,
    duration_mins   INT DEFAULT 90,
    status          TEXT DEFAULT 'confirmed',  -- confirmed | cancelled | completed
    notes           TEXT DEFAULT '',
    require_approval_for_paid_items BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_bookings_time ON bookings(booked_at, table_id);

-- ─────────────────────────── Menu Items ──────────────────────
-- Live availability state — separate from the embedded menu.md
CREATE TABLE IF NOT EXISTS menu_items (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    category    TEXT NOT NULL,           -- starter | main | dessert | drink | special
    price       NUMERIC(10,2) NOT NULL,
    description TEXT,
    dietary     TEXT[] DEFAULT '{}',     -- ['vegetarian','vegan','gluten-free','halal']
    available   BOOLEAN DEFAULT TRUE,
    price_estimated BOOLEAN NOT NULL DEFAULT FALSE,
    canonical_id TEXT,
    aliases TEXT[] NOT NULL DEFAULT '{}',
    ingredients TEXT[] NOT NULL DEFAULT '{}',
    allergens TEXT[] NOT NULL DEFAULT '{}',
    service_periods TEXT[] NOT NULL DEFAULT '{}',
    availability_status TEXT NOT NULL DEFAULT 'available',
    knowledge_metadata JSONB NOT NULL DEFAULT '{}',
    source_id TEXT NOT NULL DEFAULT '',
    data_version TEXT NOT NULL DEFAULT '',
    effective_from DATE,
    effective_to DATE
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_menu_items_name_ci
    ON menu_items (LOWER(name));
CREATE UNIQUE INDEX IF NOT EXISTS uq_menu_items_canonical_id
    ON menu_items (canonical_id) WHERE canonical_id IS NOT NULL;

-- Versioned structured restaurant facts. Payload remains JSONB because record
-- shapes differ across hours, policies, areas, escalation routes, and style.
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

-- ─────────────────────────── Call Sessions ───────────────────
CREATE TABLE IF NOT EXISTS call_sessions (
    id              SERIAL PRIMARY KEY,
    session_id      TEXT UNIQUE NOT NULL,
    caller_phone    TEXT DEFAULT '',
    state           JSONB DEFAULT '{}',
    provider        TEXT DEFAULT '',
    metadata        JSONB DEFAULT '{}',
    behavior_state  JSONB DEFAULT '{}',
    started_at      TIMESTAMP DEFAULT NOW(),
    ended_at        TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT NOW()
);

-- ─────────────────────────── Orders ──────────────────────────
-- One order per call session (pre-order or phone order)
CREATE TABLE IF NOT EXISTS orders (
    id              SERIAL PRIMARY KEY,
    session_id      TEXT NOT NULL,
    booking_id      INT REFERENCES bookings(id) ON DELETE SET NULL,
    customer_name   TEXT DEFAULT '',
    customer_phone  TEXT DEFAULT '',
    status          TEXT DEFAULT 'pending',   -- pending | confirmed | cancelled
    fulfillment_type TEXT,                   -- dine_in | pickup | delivery | null until set
    fulfillment_details JSONB NOT NULL DEFAULT '{}',
    total_amount    NUMERIC(10,2) DEFAULT 0,
    notes           TEXT DEFAULT '',
    allergy_notes   TEXT DEFAULT '',
    draft_version   INT NOT NULL DEFAULT 1,
    created_at      TIMESTAMP DEFAULT NOW(),
    confirmed_at    TIMESTAMP,
    CHECK (
        fulfillment_type IS NULL
        OR fulfillment_type IN ('dine_in', 'pickup', 'delivery')
    )
);

CREATE INDEX IF NOT EXISTS idx_orders_session ON orders(session_id, status);
CREATE UNIQUE INDEX IF NOT EXISTS uq_orders_one_pending_per_session
    ON orders(session_id) WHERE status = 'pending';

-- ─────────────────────────── Order Items ─────────────────────
CREATE TABLE IF NOT EXISTS order_items (
    id              SERIAL PRIMARY KEY,
    order_id        INT REFERENCES orders(id) ON DELETE CASCADE,
    menu_item_id    INT REFERENCES menu_items(id),
    item_name       TEXT NOT NULL,
    quantity        INT DEFAULT 1,
    unit_price      NUMERIC(10,2) NOT NULL,
    subtotal        NUMERIC(10,2) GENERATED ALWAYS AS (quantity * unit_price) STORED,
    notes           TEXT DEFAULT '',
    modifiers       JSONB NOT NULL DEFAULT '[]',
    removals        TEXT[] NOT NULL DEFAULT '{}',
    substitutions   JSONB NOT NULL DEFAULT '[]',
    proposed        BOOLEAN NOT NULL DEFAULT FALSE
);

-- ─────────────────────────── Schema migrations ────────────────
-- Add cancellation_reason to bookings if it doesn't exist yet
ALTER TABLE bookings ADD COLUMN IF NOT EXISTS cancellation_reason TEXT DEFAULT '';

-- ───────────────────── Voice action idempotency ──────────────
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

-- ───────────────────── Voice call observability ──────────────
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

-- ───────────────────── Operator knowledge loop ───────────────
CREATE TABLE IF NOT EXISTS knowledge_gaps (
    id                  SERIAL PRIMARY KEY,
    restaurant_id       TEXT,
    session_id          TEXT NOT NULL DEFAULT '',
    question            TEXT NOT NULL,
    question_normalized TEXT NOT NULL,
    context_excerpt     TEXT NOT NULL DEFAULT '',
    agent_response      TEXT NOT NULL DEFAULT '',
    status              TEXT NOT NULL DEFAULT 'unresolved',
    resolved_answer     TEXT NOT NULL DEFAULT '',
    resolved_by         TEXT NOT NULL DEFAULT '',
    created_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    resolved_at         TIMESTAMP,
    CHECK (status IN ('unresolved', 'resolved'))
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_gaps_restaurant_session_question
    ON knowledge_gaps (restaurant_id, session_id, question_normalized);
CREATE INDEX IF NOT EXISTS idx_knowledge_gaps_status
    ON knowledge_gaps (status, created_at DESC);

CREATE TABLE IF NOT EXISTS operator_knowledge (
    id              SERIAL PRIMARY KEY,
    restaurant_id   TEXT NOT NULL DEFAULT '',
    question        TEXT NOT NULL,
    answer          TEXT NOT NULL,
    source_gap_id   INT REFERENCES knowledge_gaps(id) ON DELETE SET NULL,
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_operator_knowledge_restaurant_question_ci
    ON operator_knowledge (restaurant_id, LOWER(question));
