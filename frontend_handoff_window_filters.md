# Frontend Handoff — Window Filters (City Risk Map + Analytics)

## What changed

Both dashboards' date dropdowns ("Today / Last 7 Days / Last 30 Days") now
have real backend support. This note explains exactly what to send, what
comes back, and — most importantly — which numbers will NOT change when you
switch the dropdown. That last part isn't a bug to fix on the frontend; it's
intentional, explained below.

---

## City Risk Map

**Dropdown options**: `Today`, `Last 7 Days` only.

`Last 30 Days` is intentionally not available here. We checked the real
data: 250 of 364 clusters (69%) have fewer than 30 days of recorded activity
in the full dataset. A 30-day total would show as empty or near-empty for
most zones — not because enforcement improved, but because the data simply
isn't there. Rather than show something misleading, we're not offering it
at the per-zone level. **If the dropdown UI currently has a 30-day option
for City Risk Map, please remove it** — calling the API with `window=30d`
on these endpoints returns a `400` error on purpose.

### Endpoints

```
GET /api/v1/zones?window=today|7d
GET /api/v1/zones/{zoneId}?window=today|7d
GET /api/v1/zones/hotspots?window=today|7d
GET /api/v1/zones/risk-map?window=today|7d
```

Default is `today` if `window` is omitted.

### Fields that DO change with the window

- `violations` / `violationCount` / `activeViolations` — real historical
  count, summed over the selected window
- `daysCoveredInWindow` — how many distinct days of data actually exist in
  that window for that specific cluster. For `today` this is always `1`.
  For `7d` it can be less than 7 if the cluster didn't have violations on
  every day. **Recommend showing this somewhere** (e.g. a small "(3 days
  of data)" caption) for clusters where it's noticeably less than the
  requested window — it tells the user the number is real but thin,
  rather than implying it's directly comparable to a well-populated zone.

### Fields that do NOT change with the window

- `riskScore` / `impact_score`
- `riskLevel` (critical/elevated/routine)
- `estimatedViolations` / `estimatedViolations24h`
- `recommendedOfficers` / `recommendedTowTrucks` (via `availableUnitsNearby`)
- `priority_rank`

**Why**: there is one trained model, and it produces one next-day forecast.
There's no 7-day-ahead model. So the risk score and forecast are always
"what we predict for tomorrow," regardless of which historical window
you're looking at. Switching the dropdown changes what history you're
viewing, not what's being predicted.

Every response includes a `meta.note` saying this explicitly — useful if
you want to show a tooltip ("Risk score reflects next-day forecast,
independent of selected period") instead of leaving it unexplained.

### Error handling

`window=30d` (or anything else not in `today`/`7d`) returns:
```json
{
  "error": {
    "code": "INVALID_WINDOW",
    "message": "window must be one of ['today', '7d'] (30d is not supported for per-zone data — insufficient real history per cluster; use violations/summary for 30d city-wide totals)"
  }
}
```
Treat this as a client-side validation case (shouldn't happen if the
dropdown only offers `today`/`7d`), not something to silently swallow.

---

## Analytics

**Dropdown options**: `Today`, `Last 7 Days`, `Last 30 Days` — all three
work here. This page aggregates city-wide, not per-zone, so the data
sparsity issue above doesn't apply.

### Endpoint

```
GET /api/v1/analytics/summary?window=today|7d|30d
```

Default is `today` if `window` is omitted.

### Response shape

```json
{
  "data": {
    "window": "7d",
    "daysCoveredInWindow": 7,
    "totalViolationsInWindow": 14823,
    "overallCityRiskLevel": "routine",
    "overallCityRiskScore": 17,
    "criticalZonesToday": 26,
    "recommendedDeployments": 320,
    "hourlyDistribution": [{"hour": 0, "violations": 142}, ...],
    "dailyTrend": [{"date": "2024-04-02", "violations": 2083}, ...],
    "topZones": [{"police_station": "Upparpet", "violations": 795}, ...]
  },
  "meta": { "note": "..." }
}
```

### Fields that DO change with the window

- `daysCoveredInWindow`, `totalViolationsInWindow`
- `hourlyDistribution` — real, recomputed per window
- `dailyTrend` — real, one entry per actual day in the window
- `topZones` — real, top 10 zones by violation count in that window

### Fields that do NOT change with the window

- `overallCityRiskLevel`, `overallCityRiskScore`
- `criticalZonesToday`
- `recommendedDeployments`

Same reasoning as City Risk Map — these reflect the next-day forecast.

### What's NOT in this endpoint at all (stays on the existing static data)

- **Violation type breakdown** (Wrong Parking 48%, No Parking 40%, etc.) —
  keep using `GET /api/v1/violations/breakdown` as you already do. This
  doesn't change with the window selector; it's full-history only. The
  underlying file we use for windowed queries doesn't retain violation
  type per row (it was dropped to save memory on Render's free tier), so
  there's no way to window this without a second, costlier data load we're
  avoiding given current hosting constraints.
- **Approval rate / SCITA-sent rate** — not available in the windowed data
  source at all (not a memory tradeoff, the fields genuinely don't exist
  there). These stay static if you're showing them anywhere.

**Practical effect**: if your Analytics page currently has a single
"Violation Breakdown" pie/list, it should keep hitting
`/violations/breakdown` unconditionally — don't wire its data fetch to the
window dropdown. Only the hourly chart, daily trend chart, and zone list
should refetch when the dropdown changes.

---

## Quick checklist for wiring the dropdowns

- [ ] City Risk Map dropdown: remove "Last 30 Days" option if present
- [ ] City Risk Map: dropdown `onChange` → refetch `zones/hotspots`,
      `zones/risk-map` (and `zones/{id}` if a zone detail panel is open)
      with `?window=today` or `?window=7d`
- [ ] Analytics dropdown: keep all three options, wire to
      `/analytics/summary?window=...`
- [ ] Analytics: do NOT wire `violations/breakdown` to the same dropdown —
      it stays a separate, unconditional fetch
- [ ] Anywhere risk score / forecast / officer counts are displayed next to
      the dropdown, consider a small label or tooltip clarifying it's a
      forecast that doesn't change with the selected period (optional, but
      prevents "why didn't this number change" questions/bug reports)
- [ ] Handle `400 INVALID_WINDOW` gracefully (shouldn't be reachable from
      normal UI use, but don't let it crash the page if it happens)
