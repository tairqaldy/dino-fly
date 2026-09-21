-- Action logs (the exact input sequence of every session) live in Postgres, not on the container disk:
-- Railway containers have ephemeral storage, so a redeploy used to lose every recorded session.
CREATE TABLE IF NOT EXISTS blobs (
  key        text PRIMARY KEY,
  body       text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
