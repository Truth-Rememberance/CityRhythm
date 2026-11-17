# Sync Intelligence: seeing a city fall out of rhythm

**Idea:** Each zone's traffic, power and water normally move together. Before an outage, that coordination weakens. The Sync Index (0-100) measures it per zone.

**How it works:**
- Remove the daily pattern from each signal (rush hours are normal)
- Critical slowing down: variance and autocorrelation rise before a failure
- Warning when the index stays low for 10 minutes; coupling and phase scores come next

**Compared against:** a plain threshold alarm (value above a limit for N minutes).

**Result:** lead time and false-alarm rate: PENDING (will come from docs/RESULTS.md only).

**Honest limits:** tested so far on simulated calm data; coupled scenarios not in yet.
