# Sync panel skeleton

Feed events into the panel with `<SyncPanel events={...} />`. It understands `sync.index`, `sync.warning`, `sync.graph`, and `baseline.alarm` channels.

For a local full event stream, `backend/sync/mock_emitters.py` writes `backend/sync/mock_events.jsonl` when run with its documented CLI options. This sample is hand-written and independent of that output.
