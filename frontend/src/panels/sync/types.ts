export type SyncStatus = "healthy" | "drifting" | "critical";

export interface SyncIndexPayload {
  zone: string;
  sync_index: number;
  status: SyncStatus;
  phase_coherence: number | null;
  csd_score: number;
  coupling_break: number | null;
  phases: Record<string, number> | null;
}

export interface SyncWarningPayload {
  zone: string;
  level: SyncStatus;
  sync_index: number;
  trigger: string;
  baseline_fired: boolean;
  lead_min: number | null;
}

// PROVISIONAL shape, confirmed by R1 for now, R6 to confirm.
export interface SyncGraphPayload {
  nodes: string[];
  edges: {
    source: string;
    target: string;
    lag_min: number;
    strength: number;
  }[];
}

export interface SyncEvent {
  channel: string;
  zone: string | null;
  t?: number;
  payload: unknown;
}
