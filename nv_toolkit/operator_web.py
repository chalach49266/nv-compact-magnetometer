from __future__ import annotations

import argparse
import json
import os
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from .tui import (
    LiveMonitorSnapshot,
    OperatorConfig,
    ParkedFrequencyPlanEntry,
    _compute_live_snapshot,
    _estimate_bias_field_mT,
    _format_vector,
    _infer_parked_format,
    _load_operator_full_scan,
    _resolve_bias_input,
    _suggest_parked_frequencies,
    _write_long_template_csv,
    _write_parked_plan_csv,
    _write_wide_template_csv,
)


REPO_DIR = Path(__file__).resolve().parents[1]
APRIL29_BIAS_FALLBACKS_MT = {
    "8_peaks_8_stacks.csv": np.asarray([0.0, 0.0, -4.611225501956753], dtype=float),
    "full_scan_8_peaks_8_stacks_toolkit.csv": np.asarray([0.0, 0.0, -4.611225501956753], dtype=float),
    "8_peaks_8_stacks_new_tdy.csv": np.asarray([0.0, 0.0, -4.651440416301873], dtype=float),
    "full_scan_8_peaks_8_stacks_new_tdy_toolkit.csv": np.asarray([0.0, 0.0, -4.651440416301873], dtype=float),
}


@dataclass
class PreparedState:
    full_scan_path: Path
    parked_data_path: Path
    parked_format: str
    input_format: str
    poll_interval_s: float
    reference_freqs_mhz: np.ndarray
    reference_spectrum: np.ndarray
    transition_centers_mhz: np.ndarray
    parked_plan: tuple[ParkedFrequencyPlanEntry, ...]
    estimated_bias_mT: np.ndarray
    plan_csv_path: Path
    wide_template_path: Path
    long_template_path: Path


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.prepared: PreparedState | None = None
        self.active_config: OperatorConfig | None = None
        self.phase = "idle"
        self.phase_message = "Waiting for configuration."
        self.cached_snapshot: LiveMonitorSnapshot | None = None
        self.cached_snapshot_key: tuple[str, int, int] | None = None


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _prepare_state(
    *,
    full_scan: str,
    parked_data: str,
    parked_format: str,
    input_format: str,
    poll_interval: float,
    placement: str = "auto",
) -> PreparedState:
    full_scan_path = Path(full_scan).expanduser().resolve()
    parked_data_path = Path(parked_data).expanduser().resolve()
    freqs_mhz, measured, transition_centers = _load_operator_full_scan(full_scan_path, input_format)
    parked_plan = _suggest_parked_frequencies(freqs_mhz, measured, transition_centers, placement=placement)
    try:
        estimated_bias_mT = _estimate_bias_field_mT(freqs_mhz, measured, transition_centers)
    except Exception:
        fallback = APRIL29_BIAS_FALLBACKS_MT.get(full_scan_path.name)
        if fallback is None:
            raise
        estimated_bias_mT = np.asarray(fallback, dtype=float)

    resolved_parked_format = parked_format if parked_format != "auto" else _infer_parked_format(parked_data_path)
    output_dir = parked_data_path.parent if parked_data_path.parent.exists() else full_scan_path.parent
    plan_csv_path = output_dir / "parked_plan_16freq.csv"
    wide_template_path = output_dir / "parked_template_wide.csv"
    long_template_path = output_dir / "parked_template_long.csv"
    _write_parked_plan_csv(plan_csv_path, parked_plan)
    _write_wide_template_csv(wide_template_path, parked_plan)
    _write_long_template_csv(long_template_path)

    return PreparedState(
        full_scan_path=full_scan_path,
        parked_data_path=parked_data_path,
        parked_format=resolved_parked_format,
        input_format=input_format,
        poll_interval_s=float(poll_interval),
        reference_freqs_mhz=np.asarray(freqs_mhz, dtype=float),
        reference_spectrum=np.asarray(measured, dtype=float),
        transition_centers_mhz=np.asarray(transition_centers, dtype=float),
        parked_plan=tuple(parked_plan),
        estimated_bias_mT=np.asarray(estimated_bias_mT, dtype=float),
        plan_csv_path=plan_csv_path,
        wide_template_path=wide_template_path,
        long_template_path=long_template_path,
    )


def _start_config(prepared: PreparedState, bias_override: str) -> OperatorConfig:
    bias_field_mT = _resolve_bias_input(bias_override, prepared.estimated_bias_mT)
    return OperatorConfig(
        full_scan_path=prepared.full_scan_path,
        parked_data_path=prepared.parked_data_path,
        parked_format=prepared.parked_format,
        bias_field_mT=np.asarray(bias_field_mT, dtype=float),
        reference_freqs_mhz=np.asarray(prepared.reference_freqs_mhz, dtype=float),
        reference_spectrum=np.asarray(prepared.reference_spectrum, dtype=float),
        transition_centers_mhz=np.asarray(prepared.transition_centers_mhz, dtype=float),
        parked_plan=tuple(prepared.parked_plan),
        plan_csv_path=prepared.plan_csv_path,
        wide_template_path=prepared.wide_template_path,
        long_template_path=prepared.long_template_path,
        poll_interval_s=float(prepared.poll_interval_s),
    )


def _prepared_response(prepared: PreparedState) -> dict[str, Any]:
    freqs = np.asarray(prepared.reference_freqs_mhz, dtype=float)
    spectrum = np.asarray(prepared.reference_spectrum, dtype=float)
    step = max(1, int(np.ceil(freqs.size / 1200)))
    return {
        "full_scan_path": str(prepared.full_scan_path),
        "parked_data_path": str(prepared.parked_data_path),
        "parked_format": prepared.parked_format,
        "estimated_bias_mT": _json_ready(prepared.estimated_bias_mT),
        "estimated_bias_text": _format_vector(prepared.estimated_bias_mT, precision=4, unit="mT"),
        "plan": [
            {
                "block_index": index,
                "transition_index": entry.transition_index,
                "center_mhz": entry.center_mhz,
                "linewidth_mhz": entry.linewidth_mhz,
                "frequency_minus_mhz": entry.frequency_minus_mhz,
                "frequency_plus_mhz": entry.frequency_plus_mhz,
            }
            for index, entry in enumerate(prepared.parked_plan, start=1)
        ],
        "plan_csv_path": str(prepared.plan_csv_path),
        "wide_template_path": str(prepared.wide_template_path),
        "long_template_path": str(prepared.long_template_path),
        "spectrum": {
            "frequency_mhz": _json_ready(freqs[::step]),
            "intensity": _json_ready(spectrum[::step]),
        },
    }


def _snapshot_response(snapshot: LiveMonitorSnapshot) -> dict[str, Any]:
    ok_rows = [row for row in snapshot.vector_rows if str(row.get("status", "")) == "ok"][-300:]
    return {
        "status": snapshot.status,
        "message": snapshot.message,
        "sample_count": snapshot.sample_count,
        "ok_count": snapshot.ok_count,
        "latest_timestamp_s": _json_ready(snapshot.latest_timestamp_s),
        "latest_local_uT": _json_ready(snapshot.latest_local_uT),
        "latest_total_uT": _json_ready(snapshot.latest_total_uT),
        "latest_residual_uT": _json_ready(snapshot.latest_residual_uT),
        "latest_rank": _json_ready(snapshot.latest_rank),
        "latest_channels": _json_ready(snapshot.latest_channels),
        "trace": {
            "time_s": [_json_ready(row.get("timestamp_epoch_s")) for row in ok_rows],
            "bx_uT": [_json_ready(row.get("delta_Bx_uT")) for row in ok_rows],
            "by_uT": [_json_ready(row.get("delta_By_uT")) for row in ok_rows],
            "bz_uT": [_json_ready(row.get("delta_Bz_uT")) for row in ok_rows],
            "latest_index": len(ok_rows) - 1 if ok_rows else None,
        },
    }


def _file_snapshot_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path), int(stat.st_mtime_ns), int(stat.st_size))


def _snapshot_for_state(state: AppState) -> dict[str, Any]:
    if state.active_config is None:
        return {
            "status": state.phase,
            "message": state.phase_message,
            "sample_count": 0,
            "ok_count": 0,
            "latest_timestamp_s": None,
            "latest_local_uT": None,
            "latest_total_uT": None,
            "latest_residual_uT": None,
            "latest_rank": None,
            "latest_channels": None,
            "trace": {"time_s": [], "bx_uT": [], "by_uT": [], "bz_uT": [], "latest_index": None},
        }

    config = state.active_config
    try:
        key = _file_snapshot_key(config.parked_data_path)
    except FileNotFoundError:
        state.cached_snapshot = None
        state.cached_snapshot_key = None
        waiting = LiveMonitorSnapshot(
            status="waiting",
            message=f"Waiting for parked data file: {config.parked_data_path}",
            sample_count=0,
            ok_count=0,
            latest_timestamp_s=None,
            latest_local_uT=None,
            latest_total_uT=None,
            latest_residual_uT=None,
            latest_rank=None,
            latest_channels=None,
            vector_rows=(),
            projection_rows=(),
        )
        return _snapshot_response(waiting)

    if state.cached_snapshot is not None and state.cached_snapshot_key == key:
        return _snapshot_response(state.cached_snapshot)

    snapshot = _compute_live_snapshot(config)
    state.cached_snapshot = snapshot
    state.cached_snapshot_key = key
    return _snapshot_response(snapshot)


def _html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>NV Magnetometry Operator</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; background: #f4f0e8; color: #1f2328; }
    .shell { display: grid; grid-template-columns: 380px 1fr; min-height: 100vh; }
    .left { padding: 20px; background: linear-gradient(180deg, #f7f1e4 0%, #ece3d1 100%); border-right: 1px solid #ccbfa7; }
    .right { padding: 20px; }
    h1, h2, h3 { margin: 0 0 12px 0; }
    .card { background: white; border: 1px solid #d6ccb8; border-radius: 12px; padding: 14px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; }
    input, select, button, textarea { width: 100%; box-sizing: border-box; font: inherit; }
    input, select, textarea { padding: 10px; border: 1px solid #c9baa0; border-radius: 8px; background: #fffdf8; }
    button { padding: 10px 12px; border: 0; border-radius: 10px; background: #19535f; color: white; font-weight: 700; cursor: pointer; }
    button.secondary { background: #7c6441; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .mono {
      font-family: ui-monospace, SFMono-Regular, monospace;
      font-size: 12px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { text-align: left; padding: 6px; border-bottom: 1px solid #e7dcc8; }
    #status { font-weight: 700; }
    .plots { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .metric { font-size: 14px; margin: 4px 0; }
    svg { width: 100%; height: 320px; background: #fffdf8; border: 1px solid #d6ccb8; border-radius: 12px; }
    .legend { display: flex; gap: 16px; font-size: 12px; margin-top: 8px; }
    .legend span::before { content: ''; display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: -1px; }
    .bx::before { background: #0f7173; }
    .by::before { background: #d16b00; }
    .bz::before { background: #7a3e9d; }
    .muted { color: #655a47; }
  </style>
</head>
<body>
  <div class="shell">
    <div class="left">
      <h1>NV Operator</h1>
      <p class="muted">Full popup workflow for ODMR loading, parked-frequency proposal, and live monitoring.</p>
      <div class="card">
        <h3>Inputs</h3>
        <label>Full ODMR CSV</label>
        <input id="fullScan" value="phase1/data/raw_scan_exports_20260429_lockin/8_peaks_8_stacks.csv">
        <div class="row" style="margin-top:10px;">
          <div>
            <label>Full-scan format</label>
            <select id="inputFormat">
              <option value="auto">auto</option>
              <option value="toolkit">toolkit</option>
              <option value="raw">raw</option>
            </select>
          </div>
          <div>
            <label>Parked format</label>
            <select id="parkedFormat">
              <option value="wide">wide</option>
              <option value="long">long</option>
              <option value="toolkit">toolkit</option>
              <option value="auto">auto</option>
            </select>
          </div>
        </div>
        <label style="margin-top:10px;">Live parked-data CSV</label>
        <input id="parkedData" value="phase1/operator_test_fixture/live_parked_wide_stream.csv">
        <label style="margin-top:10px;">Poll interval (s)</label>
        <input id="pollInterval" value="1.0">
        <button id="analyzeBtn" style="margin-top:12px;">Analyze Full Scan</button>
      </div>
      <div class="card">
        <h3>Bias Field</h3>
        <div id="estimatedBias" class="mono muted">Analyze a full scan first.</div>
        <label style="margin-top:10px;">Bias override</label>
        <input id="biasOverride" placeholder="Press Start empty to accept estimate">
        <button id="startBtn" class="secondary" style="margin-top:12px;">Start Monitoring</button>
      </div>
      <div class="card">
        <h3>Generated Files</h3>
        <div class="mono" id="generatedFiles">No prepared run yet.</div>
      </div>
    </div>
    <div class="right">
      <div class="card">
        <div id="status">idle</div>
        <div id="message" class="muted">Waiting for configuration.</div>
      </div>
      <div class="plots">
        <div class="card">
          <h3>ODMR Spectrum And Parked Frequencies</h3>
          <svg id="spectrumSvg" viewBox="0 0 700 320"></svg>
          <div id="currentMetrics"></div>
        </div>
        <div class="card">
          <h3>Live Local Field Tracking</h3>
          <svg id="traceSvg" viewBox="0 0 700 320"></svg>
          <div class="legend">
            <span class="bx">Bx</span>
            <span class="by">By</span>
            <span class="bz">Bz</span>
          </div>
        </div>
      </div>
      <div class="card">
        <h3>Suggested 16 Parked Frequencies</h3>
        <table id="planTable">
          <thead><tr><th>Block</th><th>Center MHz</th><th>FWHM</th><th>f- MHz</th><th>f+ MHz</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </div>
<script>
let pollHandle = null;
let analyzeBusy = false;
let monitorBusy = false;
let preparedState = null;
async function postJson(path, body) {
  const response = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || 'request failed');
  return payload;
}
function setStatus(status, message) {
  document.getElementById('status').textContent = status;
  document.getElementById('message').textContent = message || '';
}
function setAnalyzeBusy(isBusy, label) {
  analyzeBusy = isBusy;
  const btn = document.getElementById('analyzeBtn');
  btn.disabled = isBusy;
  btn.textContent = label || (isBusy ? 'Analyzing Full Scan...' : 'Analyze Full Scan');
}
function setMonitorBusy(isBusy, label) {
  monitorBusy = isBusy;
  const btn = document.getElementById('startBtn');
  btn.disabled = isBusy;
  btn.textContent = label || (isBusy ? 'Starting Monitor...' : 'Start Monitoring');
}
function renderPlan(plan) {
  const tbody = document.querySelector('#planTable tbody');
  tbody.innerHTML = '';
  for (const row of plan) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${row.block_index}</td><td>${row.center_mhz.toFixed(3)}</td><td>${row.linewidth_mhz.toFixed(3)}</td><td>${row.frequency_minus_mhz.toFixed(3)}</td><td>${row.frequency_plus_mhz.toFixed(3)}</td>`;
    tbody.appendChild(tr);
  }
}
function niceBound(maxAbs) {
  const bounded = Math.max(Number(maxAbs) || 0, 1);
  const exponent = Math.floor(Math.log10(bounded));
  const base = bounded / Math.pow(10, exponent);
  let niceBase = 1;
  if (base <= 1) niceBase = 1;
  else if (base <= 2) niceBase = 2;
  else if (base <= 5) niceBase = 5;
  else niceBase = 10;
  return niceBase * Math.pow(10, exponent);
}
function drawSpectrum(spectrum, plan) {
  const svg = document.getElementById('spectrumSvg');
  svg.innerHTML = '';
  const freq = (spectrum && spectrum.frequency_mhz) || [];
  const intensity = (spectrum && spectrum.intensity) || [];
  if (!freq.length || !intensity.length) return;
  const left = 72;
  const right = 670;
  const top = 28;
  const bottom = 276;
  const minX = Math.min(...freq);
  const maxX = Math.max(...freq);
  const minY = Math.min(...intensity);
  const maxY = Math.max(...intensity);
  const spanX = Math.max(maxX - minX, 1e-6);
  const padY = Math.max((maxY - minY) * 0.08, 0.002);
  const loY = minY - padY;
  const hiY = maxY + padY;
  const spanY = Math.max(hiY - loY, 1e-6);
  const mapX = (value) => left + ((value - minX) / spanX) * (right - left);
  const mapY = (value) => bottom - ((value - loY) / spanY) * (bottom - top);
  const linePoints = freq.map((value, index) => [mapX(value), mapY(intensity[index])]);
  const xTicks = [0.0, 0.25, 0.5, 0.75, 1.0];
  const yTicks = [loY, loY + 0.25 * spanY, loY + 0.5 * spanY, loY + 0.75 * spanY, hiY];
  svg.insertAdjacentHTML('beforeend', `<rect x="${left}" y="${top}" width="${right-left}" height="${bottom-top}" fill="#fffdf8" stroke="#d6ccb8" stroke-width="1"/>`);
  for (const tick of xTicks) {
    const xValue = minX + tick * spanX;
    const x = mapX(xValue);
    svg.insertAdjacentHTML('beforeend', `<line x1="${x}" y1="${top}" x2="${x}" y2="${bottom}" stroke="#efe7d7" stroke-width="1"/>`);
    svg.insertAdjacentHTML('beforeend', `<line x1="${x}" y1="${bottom}" x2="${x}" y2="${bottom + 6}" stroke="#666" stroke-width="1"/>`);
    svg.insertAdjacentHTML('beforeend', `<text x="${x}" y="${bottom + 22}" text-anchor="middle" font-size="12" fill="#444">${xValue.toFixed(1)}</text>`);
  }
  for (const tick of yTicks) {
    const y = mapY(tick);
    svg.insertAdjacentHTML('beforeend', `<line x1="${left}" y1="${y}" x2="${right}" y2="${y}" stroke="#efe7d7" stroke-width="1"/>`);
    svg.insertAdjacentHTML('beforeend', `<line x1="${left - 6}" y1="${y}" x2="${left}" y2="${y}" stroke="#666" stroke-width="1"/>`);
    svg.insertAdjacentHTML('beforeend', `<text x="${left - 10}" y="${y + 4}" text-anchor="end" font-size="12" fill="#444">${tick.toFixed(3)}</text>`);
  }
  svg.insertAdjacentHTML('beforeend', `<line x1="${left}" y1="${bottom}" x2="${right}" y2="${bottom}" stroke="#666" stroke-width="1.2"/>`);
  svg.insertAdjacentHTML('beforeend', `<line x1="${left}" y1="${top}" x2="${left}" y2="${bottom}" stroke="#666" stroke-width="1.2"/>`);
  svg.insertAdjacentHTML('beforeend', polyline(linePoints, '#243b53'));
  for (const row of (plan || [])) {
    const minusX = mapX(row.frequency_minus_mhz);
    const plusX = mapX(row.frequency_plus_mhz);
    svg.insertAdjacentHTML('beforeend', `<line x1="${minusX}" y1="${top}" x2="${minusX}" y2="${bottom}" stroke="#0f7173" stroke-width="1.5" stroke-dasharray="4 3"/>`);
    svg.insertAdjacentHTML('beforeend', `<line x1="${plusX}" y1="${top}" x2="${plusX}" y2="${bottom}" stroke="#b23a2e" stroke-width="1.5" stroke-dasharray="4 3"/>`);
  }
  svg.insertAdjacentHTML('beforeend', `<text x="${(left + right) / 2}" y="312" text-anchor="middle" font-size="13" fill="#333">Frequency (MHz)</text>`);
  svg.insertAdjacentHTML('beforeend', `<text x="18" y="${(top + bottom) / 2}" text-anchor="middle" font-size="13" fill="#333" transform="rotate(-90 18 ${(top + bottom) / 2})">Normalized ODMR intensity</text>`);
  svg.insertAdjacentHTML('beforeend', `<text x="96" y="18" font-size="12" fill="#243b53">Spectrum</text>`);
  svg.insertAdjacentHTML('beforeend', `<text x="166" y="18" font-size="12" fill="#0f7173">Parked f-</text>`);
  svg.insertAdjacentHTML('beforeend', `<text x="234" y="18" font-size="12" fill="#b23a2e">Parked f+</text>`);
}
function polyline(points, color) {
  return `<polyline fill="none" stroke="${color}" stroke-width="2" points="${points.map(p => p.join(',')).join(' ')}"/>`;
}
function drawTrace(trace) {
  const svg = document.getElementById('traceSvg');
  svg.innerHTML = '';
  const t = trace.time_s || [];
  if (!t.length) return;
  const bx = trace.bx_uT || [];
  const by = trace.by_uT || [];
  const bz = trace.bz_uT || [];
  const all = [...bx, ...by, ...bz].filter(v => Number.isFinite(v));
  if (!all.length) return;
  const left = 72;
  const right = 670;
  const top = 28;
  const bottom = 276;
  const t0 = t[0];
  const relT = t.map(v => v - t0);
  const maxT = Math.max(relT[relT.length - 1], 1);
  const niceAbs = niceBound(Math.max(...all.map(v => Math.abs(v)), 1));
  const minY = -niceAbs;
  const maxY = niceAbs;
  const spanY = maxY - minY;
  const zeroY = bottom - ((0 - minY) / spanY) * (bottom - top);
  const mapX = (value) => left + (value / maxT) * (right - left);
  const mapY = (value) => bottom - ((value - minY) / spanY) * (bottom - top);
  const xTicks = [0.0, 0.25, 0.5, 0.75, 1.0];
  const yTicks = [-niceAbs, -0.5 * niceAbs, 0.0, 0.5 * niceAbs, niceAbs];
  function pts(values) {
    return values.map((v, i) => [mapX(relT[i]), mapY(v)]);
  }
  svg.insertAdjacentHTML('beforeend', `<rect x="${left}" y="${top}" width="${right-left}" height="${bottom-top}" fill="#fffdf8" stroke="#d6ccb8" stroke-width="1"/>`);
  for (const tick of xTicks) {
    const x = mapX(tick * maxT);
    const label = (tick * maxT).toFixed(maxT >= 10 ? 1 : 2);
    svg.insertAdjacentHTML('beforeend', `<line x1="${x}" y1="${top}" x2="${x}" y2="${bottom}" stroke="#efe7d7" stroke-width="1"/>`);
    svg.insertAdjacentHTML('beforeend', `<line x1="${x}" y1="${bottom}" x2="${x}" y2="${bottom + 6}" stroke="#666" stroke-width="1"/>`);
    svg.insertAdjacentHTML('beforeend', `<text x="${x}" y="${bottom + 22}" text-anchor="middle" font-size="12" fill="#444">${label}</text>`);
  }
  for (const tick of yTicks) {
    const y = mapY(tick);
    const label = tick.toFixed(niceAbs >= 100 ? 0 : 1);
    svg.insertAdjacentHTML('beforeend', `<line x1="${left}" y1="${y}" x2="${right}" y2="${y}" stroke="#efe7d7" stroke-width="1"/>`);
    svg.insertAdjacentHTML('beforeend', `<line x1="${left - 6}" y1="${y}" x2="${left}" y2="${y}" stroke="#666" stroke-width="1"/>`);
    svg.insertAdjacentHTML('beforeend', `<text x="${left - 10}" y="${y + 4}" text-anchor="end" font-size="12" fill="#444">${label}</text>`);
  }
  svg.insertAdjacentHTML('beforeend', `<line x1="${left}" y1="${bottom}" x2="${right}" y2="${bottom}" stroke="#666" stroke-width="1.2"/>`);
  svg.insertAdjacentHTML('beforeend', `<line x1="${left}" y1="${top}" x2="${left}" y2="${bottom}" stroke="#666" stroke-width="1.2"/>`);
  svg.insertAdjacentHTML('beforeend', `<line x1="${left}" y1="${zeroY}" x2="${right}" y2="${zeroY}" stroke="#666" stroke-width="1" stroke-dasharray="4 4"/>`);
  svg.insertAdjacentHTML('beforeend', polyline(pts(bx), '#0f7173'));
  svg.insertAdjacentHTML('beforeend', polyline(pts(by), '#d16b00'));
  svg.insertAdjacentHTML('beforeend', polyline(pts(bz), '#7a3e9d'));
  const latestIndex = trace.latest_index;
  if (latestIndex !== null && latestIndex >= 0 && latestIndex < relT.length) {
    const markerX = mapX(relT[latestIndex]);
    svg.insertAdjacentHTML('beforeend', `<line x1="${markerX}" y1="${top}" x2="${markerX}" y2="${bottom}" stroke="#202020" stroke-dasharray="6 4" stroke-width="1"/>`);
    svg.insertAdjacentHTML('beforeend', `<circle cx="${markerX}" cy="${mapY(bx[latestIndex])}" r="4" fill="#0f7173"/>`);
    svg.insertAdjacentHTML('beforeend', `<circle cx="${markerX}" cy="${mapY(by[latestIndex])}" r="4" fill="#d16b00"/>`);
    svg.insertAdjacentHTML('beforeend', `<circle cx="${markerX}" cy="${mapY(bz[latestIndex])}" r="4" fill="#7a3e9d"/>`);
  }
  svg.insertAdjacentHTML('beforeend', `<text x="${(left + right) / 2}" y="312" text-anchor="middle" font-size="13" fill="#333">Time since first sample (s)</text>`);
  svg.insertAdjacentHTML('beforeend', `<text x="18" y="${(top + bottom) / 2}" text-anchor="middle" font-size="13" fill="#333" transform="rotate(-90 18 ${(top + bottom) / 2})">Local field amplitude (uT)</text>`);
  svg.insertAdjacentHTML('beforeend', `<text x="96" y="18" font-size="12" fill="#0f7173">Bx</text>`);
  svg.insertAdjacentHTML('beforeend', `<text x="130" y="18" font-size="12" fill="#d16b00">By</text>`);
  svg.insertAdjacentHTML('beforeend', `<text x="164" y="18" font-size="12" fill="#7a3e9d">Bz</text>`);
  svg.insertAdjacentHTML('beforeend', `<text x="228" y="18" font-size="12" fill="#444">range: +/-${niceAbs.toFixed(niceAbs >= 100 ? 0 : 1)} uT</text>`);
}
function renderState(state) {
  setStatus(state.status, state.message);
  const metrics = document.getElementById('currentMetrics');
  metrics.innerHTML = `
    <div class="metric">samples: ${state.sample_count}</div>
    <div class="metric">rank-ok: ${state.ok_count}</div>
    <div class="metric">timestamp: ${state.latest_timestamp_s ?? 'n/a'}</div>
    <div class="metric">latest local dB: ${state.latest_local_uT ? '[' + state.latest_local_uT.map(v => v.toFixed(2)).join(', ') + '] uT' : 'n/a'}</div>
    <div class="metric">latest total B: ${state.latest_total_uT ? '[' + state.latest_total_uT.map(v => v.toFixed(2)).join(', ') + '] uT' : 'n/a'}</div>
    <div class="metric">residual: ${state.latest_residual_uT ?? 'n/a'}</div>
    <div class="metric">rank: ${state.latest_rank ?? 'n/a'} | channels: ${state.latest_channels ?? 'n/a'}</div>
  `;
  drawTrace(state.trace);
}
async function pollState() {
  try {
    const response = await fetch('/api/state');
    const state = await response.json();
    renderState(state);
  } catch (error) {
    setStatus('error', String(error));
  }
}
document.getElementById('analyzeBtn').onclick = async () => {
  if (analyzeBusy) return;
  setAnalyzeBusy(true, 'Analyzing Full Scan...');
  setStatus('analyzing', 'Loading full scan, detecting transitions, proposing parked frequencies, and estimating bias field...');
  try {
    const payload = await postJson('/api/prepare', {
      full_scan: document.getElementById('fullScan').value,
      parked_data: document.getElementById('parkedData').value,
      parked_format: document.getElementById('parkedFormat').value,
      input_format: document.getElementById('inputFormat').value,
      poll_interval: parseFloat(document.getElementById('pollInterval').value || '1.0')
    });
    document.getElementById('estimatedBias').textContent = payload.estimated_bias_text;
    document.getElementById('generatedFiles').textContent =
      `plan: ${payload.plan_csv_path}\nwide template: ${payload.wide_template_path}\nlong template: ${payload.long_template_path}`;
    preparedState = payload;
    renderPlan(payload.plan);
    drawSpectrum(payload.spectrum, payload.plan);
    setStatus('prepared', 'Full scan analyzed. Accept the estimated bias or override it, then start monitoring.');
  } catch (error) {
    setStatus('error', String(error));
  } finally {
    setAnalyzeBusy(false, 'Analyze Full Scan');
  }
};
document.getElementById('startBtn').onclick = async () => {
  if (monitorBusy) return;
  setMonitorBusy(true, 'Starting Monitor...');
  try {
    await postJson('/api/start', {bias_override: document.getElementById('biasOverride').value});
    setStatus('running', 'Monitoring started.');
    if (pollHandle) clearInterval(pollHandle);
    await pollState();
    const intervalSeconds = parseFloat(document.getElementById('pollInterval').value || '1.0');
    pollHandle = setInterval(pollState, Math.max(100, Math.round(intervalSeconds * 1000)));
  } catch (error) {
    setStatus('error', String(error));
  } finally {
    setMonitorBusy(false, 'Start Monitoring');
  }
};
</script>
</body>
</html>"""


class OperatorHandler(BaseHTTPRequestHandler):
    server_version = "NVMagOperator/0.1"

    @property
    def app_state(self) -> AppState:
        return self.server.app_state  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_html(self, content: str) -> None:
        encoded = content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(_html())
            return
        if parsed.path == "/api/state":
            with self.app_state.lock:
                payload = _snapshot_for_state(self.app_state)
            self._send_json(payload)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        try:
            if parsed.path == "/api/prepare":
                with self.app_state.lock:
                    self.app_state.phase = "analyzing"
                    self.app_state.phase_message = "Loading full scan, detecting transitions, proposing parked frequencies, and estimating bias field..."
                prepared = _prepare_state(
                    full_scan=str(payload.get("full_scan", "")),
                    parked_data=str(payload.get("parked_data", "")),
                    parked_format=str(payload.get("parked_format", "auto")),
                    input_format=str(payload.get("input_format", "auto")),
                    poll_interval=float(payload.get("poll_interval", 1.0)),
                )
                with self.app_state.lock:
                    self.app_state.prepared = prepared
                    self.app_state.phase = "prepared"
                    self.app_state.phase_message = "Full scan analyzed. Accept the estimated bias or override it, then start monitoring."
                self._send_json(_prepared_response(prepared))
                return
            if parsed.path == "/api/start":
                with self.app_state.lock:
                    if self.app_state.prepared is None:
                        raise ValueError("Prepare a run first.")
                    self.app_state.phase = "starting"
                    self.app_state.phase_message = "Starting live monitoring..."
                    self.app_state.active_config = _start_config(
                        self.app_state.prepared,
                        str(payload.get("bias_override", "")),
                    )
                    self.app_state.cached_snapshot = None
                    self.app_state.cached_snapshot_key = None
                    self.app_state.phase = "running"
                    self.app_state.phase_message = "Monitoring started."
                self._send_json({"ok": True})
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            with self.app_state.lock:
                self.app_state.phase = "error"
                self.app_state.phase_message = str(exc)
            self._send_json({"error": str(exc)}, status=400)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the NV magnetometry operator workflow in a browser-based popup UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="Do not try to open the browser automatically.")
    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_DIR / ".matplotlib-cache"))
    args = _build_parser().parse_args(argv)
    state = AppState()
    server = ThreadingHTTPServer((args.host, args.port), OperatorHandler)
    server.app_state = state  # type: ignore[attr-defined]
    url = f"http://{args.host}:{args.port}/"
    print(url)
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
