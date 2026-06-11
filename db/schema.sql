-- ══════════════════════════════════════════════════════════════
-- La Casa Restaurant — Database Schema
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
    available   BOOLEAN DEFAULT TRUE
);

-- ─────────────────────────── Call Sessions ───────────────────
CREATE TABLE IF NOT EXISTS call_sessions (
    id              SERIAL PRIMARY KEY,
    session_id      TEXT UNIQUE NOT NULL,
    caller_phone    TEXT DEFAULT '',
    state           JSONB DEFAULT '{}',
    started_at      TIMESTAMP DEFAULT NOW(),
    ended_at        TIMESTAMP
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
    total_amount    NUMERIC(10,2) DEFAULT 0,
    notes           TEXT DEFAULT '',
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_orders_session ON orders(session_id, status);

-- ─────────────────────────── Order Items ─────────────────────
CREATE TABLE IF NOT EXISTS order_items (
    id              SERIAL PRIMARY KEY,
    order_id        INT REFERENCES orders(id) ON DELETE CASCADE,
    menu_item_id    INT REFERENCES menu_items(id),
    item_name       TEXT NOT NULL,
    quantity        INT DEFAULT 1,
    unit_price      NUMERIC(10,2) NOT NULL,
    subtotal        NUMERIC(10,2) GENERATED ALWAYS AS (quantity * unit_price) STORED,
    notes           TEXT DEFAULT ''
);

-- ─────────────────────────── Schema migrations ────────────────
-- Add cancellation_reason to bookings if it doesn't exist yet
ALTER TABLE bookings ADD COLUMN IF NOT EXISTS cancellation_reason TEXT DEFAULT '';
