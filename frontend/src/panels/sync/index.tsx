import React from "react";
import type {
  SyncEvent,
  SyncGraphPayload,
  SyncIndexPayload,
  SyncStatus,
  SyncWarningPayload,
} from "./types";

const ZONES = [
  "silk_board",
  "whitefield",
  "hebbal",
  "electronic_city",
  "koramangala",
  "indiranagar",
  "jayanagar",
  "yelahanka",
] as const;

const STATUS_COLORS: Record<SyncStatus, string> = {
  healthy: "var(--sync-healthy, #218739)",
  drifting: "var(--sync-drifting, #c17b00)",
  critical: "var(--sync-critical, #c53030)",
};

const styles: Record<string, React.CSSProperties> = {
  root: {
    color: "var(--sync-text, #1f2937)",
    fontFamily: "var(--sync-font, sans-serif)",
    padding: "1rem",
  },
  cards: {
    display: "grid",
    gap: "0.75rem",
    gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
    marginBottom: "1.5rem",
  },
  card: {
    border: "1px solid var(--sync-border, #d1d5db)",
    borderRadius: "0.5rem",
    padding: "0.75rem",
  },
  section: { marginTop: "1.25rem" },
  graph: {
    border: "1px solid var(--sync-border, #d1d5db)",
    borderRadius: "0.5rem",
    display: "block",
    maxWidth: "100%",
  },
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isSyncStatus(value: unknown): value is SyncStatus {
  return value === "healthy" || value === "drifting" || value === "critical";
}

function readIndex(payload: unknown): SyncIndexPayload | null {
  if (!isRecord(payload)) return null;
  const { zone, sync_index, status, phase_coherence, csd_score, coupling_break, phases } = payload;
  if (
    typeof zone !== "string" ||
    typeof sync_index !== "number" ||
    !isSyncStatus(status) ||
    !(typeof phase_coherence === "number" || phase_coherence === null) ||
    typeof csd_score !== "number" ||
    !(typeof coupling_break === "number" || coupling_break === null) ||
    !(phases === null || isRecord(phases))
  ) return null;
  return payload as unknown as SyncIndexPayload;
}

function readWarning(payload: unknown): SyncWarningPayload | null {
  if (!isRecord(payload)) return null;
  const { zone, level, sync_index, trigger, baseline_fired, lead_min } = payload;
  if (
    typeof zone !== "string" ||
    !isSyncStatus(level) ||
    typeof sync_index !== "number" ||
    typeof trigger !== "string" ||
    typeof baseline_fired !== "boolean" ||
    !(typeof lead_min === "number" || lead_min === null)
  ) return null;
  return payload as unknown as SyncWarningPayload;
}

function readGraph(payload: unknown): SyncGraphPayload | null {
  if (!isRecord(payload) || !Array.isArray(payload.nodes) || !Array.isArray(payload.edges)) return null;
  const nodesAreStrings = payload.nodes.every((node) => typeof node === "string");
  const edgesAreValid = payload.edges.every((edge) =>
    isRecord(edge) &&
    typeof edge.source === "string" &&
    typeof edge.target === "string" &&
    typeof edge.lag_min === "number" &&
    typeof edge.strength === "number",
  );
  return nodesAreStrings && edgesAreValid ? payload as unknown as SyncGraphPayload : null;
}

function zonePoint(index: number, count: number) {
  const angle = (2 * Math.PI * index) / Math.max(count, 1) - Math.PI / 2;
  return { x: 180 + 135 * Math.cos(angle), y: 160 + 115 * Math.sin(angle) };
}

export default function SyncPanel({ events }: { events: SyncEvent[] }) {
  const latestByZone = new Map<string, SyncIndexPayload>();
  const warnings: SyncWarningPayload[] = [];
  const alarms: SyncEvent[] = [];
  let latestGraph: SyncGraphPayload | null = null;

  for (const event of events) {
    if (event.channel === "sync.index") {
      const payload = readIndex(event.payload);
      if (payload) latestByZone.set(payload.zone, payload);
    } else if (event.channel === "sync.warning") {
      const payload = readWarning(event.payload);
      if (payload) warnings.push(payload);
    } else if (event.channel === "sync.graph") {
      latestGraph = readGraph(event.payload);
    } else if (event.channel === "baseline.alarm") {
      alarms.push(event);
    }
  }

  const graphNodes = latestGraph?.nodes ?? [];
  const graphPositions = new Map(graphNodes.map((node, index) => [node, zonePoint(index, graphNodes.length)]));

  return (
    <section style={styles.root} aria-label="Sync panel">
      <h2>Sync Index</h2>
      <div style={styles.cards}>
        {ZONES.map((zone) => {
          const item = latestByZone.get(zone);
          const status = item?.status;
          const indexText = typeof item?.sync_index === "number" ? String(Math.round(item.sync_index)) : "n/a";
          return (
            <article key={zone} style={{ ...styles.card, borderTop: `4px solid ${status ? STATUS_COLORS[status] : "var(--sync-border, #d1d5db)"}` }}>
              <strong>{zone}</strong>
              <div>{indexText}</div>
              <div style={{ color: status ? STATUS_COLORS[status] : "var(--sync-muted, #6b7280)" }}>{status ?? "n/a"}</div>
              <small>Phase: {item?.phase_coherence ?? "n/a"}</small>
              <br />
              <small>CSD: {item?.csd_score ?? "n/a"}</small>
              <br />
              <small>Coupling: {item?.coupling_break ?? "n/a"}</small>
            </article>
          );
        })}
      </div>

      <section style={styles.section} aria-label="Warnings">
        <h3>Warnings</h3>
        {warnings.length === 0 ? <p>No warnings</p> : (
          <ul>
            {warnings.map((warning, index) => (
              <li key={`${warning.zone}-${index}`}>
                {warning.zone}: {warning.level}
                {warning.baseline_fired && warning.lead_min !== null
                  ? ` — baseline fired, lead ${warning.lead_min} min`
                  : ""}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section style={styles.section} aria-label="Baseline alarms">
        <h3>Baseline alarms</h3>
        {alarms.length === 0 ? <p>No baseline alarms</p> : (
          <ul>
            {alarms.map((event, index) => {
              const payload = isRecord(event.payload) ? event.payload : {};
              const zone = typeof payload.zone === "string" ? payload.zone : event.zone ?? "n/a";
              const system = typeof payload.system === "string" ? ` (${payload.system})` : "";
              return <li key={`${zone}-${index}`}>{zone}{system}</li>;
            })}
          </ul>
        )}
      </section>

      <section style={styles.section} aria-label="Sync graph">
        <h3>Sync graph</h3>
        {!latestGraph ? <p>no graph yet</p> : (
          <svg style={styles.graph} viewBox="0 0 360 320" role="img" aria-label="Zone coupling graph">
            {latestGraph.edges.map((edge, index) => {
              const source = graphPositions.get(edge.source);
              const target = graphPositions.get(edge.target);
              if (!source || !target) return null;
              const middleX = (source.x + target.x) / 2;
              const middleY = (source.y + target.y) / 2;
              return (
                <g key={`${edge.source}-${edge.target}-${index}`}>
                  <line x1={source.x} y1={source.y} x2={target.x} y2={target.y} stroke="var(--sync-edge, #718096)" strokeWidth={Math.max(1, edge.strength * 6)} />
                  <text x={middleX} y={middleY} fontSize="9" textAnchor="middle" fill="var(--sync-text, #1f2937)">{edge.lag_min}m</text>
                </g>
              );
            })}
            {graphNodes.map((node, index) => {
              const point = graphPositions.get(node);
              if (!point) return null;
              return (
                <g key={node}>
                  <circle cx={point.x} cy={point.y} r="8" fill="var(--sync-node, #3182ce)" />
                  <text x={point.x} y={point.y + 21} fontSize="9" textAnchor="middle" fill="var(--sync-text, #1f2937)">{node}</text>
                </g>
              );
            })}
          </svg>
        )}
      </section>
    </section>
  );
}
