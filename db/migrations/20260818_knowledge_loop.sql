-- Unknown-question loop: log gaps from calls, resolve them into live FAQ.

BEGIN;

CREATE TABLE IF NOT EXISTS knowledge_gaps (
    id                  SERIAL PRIMARY KEY,
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

CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_gaps_session_question
    ON knowledge_gaps (session_id, question_normalized);

CREATE INDEX IF NOT EXISTS idx_knowledge_gaps_status
    ON knowledge_gaps (status, created_at DESC);

CREATE TABLE IF NOT EXISTS operator_knowledge (
    id              SERIAL PRIMARY KEY,
    question        TEXT NOT NULL,
    answer          TEXT NOT NULL,
    source_gap_id   INT REFERENCES knowledge_gaps(id) ON DELETE SET NULL,
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_operator_knowledge_question_ci
    ON operator_knowledge (LOWER(question));

COMMIT;
