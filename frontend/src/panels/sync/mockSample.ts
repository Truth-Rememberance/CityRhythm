import type { SyncEvent } from "./types";

export const mockSample: SyncEvent[] = [
  { channel: "sync.index", zone: "silk_board", t: 120, payload: { zone: "silk_board", sync_index: 84.6, status: "healthy", phase_coherence: null, csd_score: 0.18, coupling_break: null, phases: null } },
  { channel: "sync.index", zone: "whitefield", t: 120, payload: { zone: "whitefield", sync_index: 78.2, status: "healthy", phase_coherence: null, csd_score: 0.26, coupling_break: null, phases: null } },
  { channel: "sync.index", zone: "hebbal", t: 120, payload: { zone: "hebbal", sync_index: 65.4, status: "drifting", phase_coherence: null, csd_score: 0.43, coupling_break: null, phases: null } },
  { channel: "sync.index", zone: "koramangala", t: 120, payload: { zone: "koramangala", sync_index: 91.1, status: "healthy", phase_coherence: null, csd_score: 0.08, coupling_break: null, phases: null } },
  { channel: "sync.index", zone: "silk_board", t: 125, payload: { zone: "silk_board", sync_index: 68.7, status: "drifting", phase_coherence: null, csd_score: 0.51, coupling_break: null, phases: null } },
  { channel: "sync.index", zone: "whitefield", t: 125, payload: { zone: "whitefield", sync_index: 77.5, status: "healthy", phase_coherence: null, csd_score: 0.28, coupling_break: null, phases: null } },
  { channel: "sync.index", zone: "hebbal", t: 125, payload: { zone: "hebbal", sync_index: 62.0, status: "drifting", phase_coherence: null, csd_score: 0.48, coupling_break: null, phases: null } },
  { channel: "sync.index", zone: "koramangala", t: 125, payload: { zone: "koramangala", sync_index: 89.3, status: "healthy", phase_coherence: null, csd_score: 0.11, coupling_break: null, phases: null } },
  { channel: "sync.warning", zone: "silk_board", t: 125, payload: { zone: "silk_board", level: "drifting", sync_index: 68.7, trigger: "csd", baseline_fired: false, lead_min: null } },
  { channel: "baseline.alarm", zone: "silk_board", t: 130, payload: { zone: "silk_board", system: "traffic", metric: "flow", value: 0.96, threshold: 0.9, sustained_min: 5 } },
  { channel: "sync.graph", zone: null, t: 120, payload: { nodes: ["silk_board", "whitefield", "hebbal", "koramangala"], edges: [{ source: "silk_board", target: "hebbal", lag_min: 5, strength: 0.62 }, { source: "silk_board", target: "whitefield", lag_min: 8, strength: 0.31 }] } },
];
