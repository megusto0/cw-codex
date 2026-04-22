import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import WindowCurveChart from '../components/WindowCurveChart';
import { fetchJson } from '../api';
import type { WindowRecord } from '../types';

export default function CaseExplorerPage() {
  const navigate = useNavigate();
  const [windows, setWindows] = useState<WindowRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [patientFilter, setPatientFilter] = useState<string>('all');
  const [labelFilter, setLabelFilter] = useState<string>('all');
  const [segmentFilter, setSegmentFilter] = useState<string>('all');

  useEffect(() => {
    fetchJson<WindowRecord[]>('/api/windows?limit=800')
      .then((records) => {
        setWindows(records);
        const firstUsable = records.find((item) => item.usable_for_memory) ?? records[0];
        setSelectedId(firstUsable?.window_id ?? null);
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  const filtered = useMemo(() => {
    return windows.filter((record) => {
      if (patientFilter !== 'all' && record.patient_id !== patientFilter) return false;
      if (labelFilter !== 'all' && record.label !== labelFilter) return false;
      if (segmentFilter !== 'all' && record.meal_segment !== segmentFilter) return false;
      return true;
    });
  }, [windows, patientFilter, labelFilter, segmentFilter]);

  const selected = filtered.find((record) => record.window_id === selectedId) ?? filtered[0] ?? null;

  useEffect(() => {
    if (selected && selected.window_id !== selectedId) {
      setSelectedId(selected.window_id);
    }
  }, [selected, selectedId]);

  if (error) {
    return <section className="page"><div className="panel">Failed to load windows: {error}</div></section>;
  }

  if (!windows.length) {
    return <section className="page"><div className="panel">Loading meal windows...</div></section>;
  }

  const patients = Array.from(new Set(windows.map((item) => item.patient_id))).sort();
  const labels = Array.from(new Set(windows.map((item) => item.label))).sort();
  const segments = Array.from(new Set(windows.map((item) => item.meal_segment))).sort();

  return (
    <section className="page">
      <header className="page-header">
        <div>
          <span className="eyebrow">Case Explorer</span>
          <h2>Browse Meal Windows</h2>
          <p>Choose a retrospective meal window, inspect its curve and context, then send it to associative retrieval.</p>
        </div>
      </header>

      <section className="filters panel">
        <label>
          Patient
          <select value={patientFilter} onChange={(event) => setPatientFilter(event.target.value)}>
            <option value="all">All patients</option>
            {patients.map((patient) => (
              <option key={patient} value={patient}>
                Patient {patient}
              </option>
            ))}
          </select>
        </label>

        <label>
          Label
          <select value={labelFilter} onChange={(event) => setLabelFilter(event.target.value)}>
            <option value="all">All labels</option>
            {labels.map((label) => (
              <option key={label} value={label}>
                {label.replace(/_/g, ' ')}
              </option>
            ))}
          </select>
        </label>

        <label>
          Time of day
          <select value={segmentFilter} onChange={(event) => setSegmentFilter(event.target.value)}>
            <option value="all">All segments</option>
            {segments.map((segment) => (
              <option key={segment} value={segment}>
                {segment}
              </option>
            ))}
          </select>
        </label>
      </section>

      <section className="case-layout">
        <article className="panel scroll-panel">
          <div className="case-list">
            {filtered.slice(0, 150).map((record) => (
              <button
                key={record.window_id}
                type="button"
                className={`case-list-item${selected?.window_id === record.window_id ? ' selected' : ''}`}
                onClick={() => setSelectedId(record.window_id)}
              >
                <div>
                  <strong>Patient {record.patient_id}</strong>
                  <span>{new Date(record.meal_time).toLocaleString()}</span>
                </div>
                <div>
                  <span className="badge neutral">{record.label_display}</span>
                  <span className={`badge ${record.usable_for_memory ? 'accent' : 'muted'}`}>
                    {record.usable_for_memory ? record.split : 'excluded'}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </article>

        {selected ? (
          <article className="panel">
            <div className="detail-header">
              <div>
                <span className="eyebrow">Selected Window</span>
                <h3>{selected.label_display}</h3>
                <p>{selected.label_reason}</p>
              </div>
              <button
                className="primary-button"
                disabled={!selected.usable_for_memory}
                onClick={() => navigate(`/retrieval?windowId=${selected.window_id}`)}
              >
                Use as query
              </button>
            </div>

            <WindowCurveChart
              minutes={selected.full_curve_minutes}
              series={[
                {
                  key: 'query',
                  label: 'Meal window',
                  color: '#1f8a70',
                  values: selected.full_curve_values,
                },
              ]}
            />

            <div className="detail-grid">
              <div className="key-value"><span>Carbs</span><strong>{selected.carbs.toFixed(1)} g</strong></div>
              <div className="key-value"><span>Bolus</span><strong>{selected.bolus.toFixed(1)} U</strong></div>
              <div className="key-value"><span>Baseline</span><strong>{selected.baseline_glucose.toFixed(0)} mg/dL</strong></div>
              <div className="key-value"><span>Trend 30m</span><strong>{selected.trend_30m.toFixed(1)}</strong></div>
              <div className="key-value"><span>Trend 90m</span><strong>{selected.trend_90m.toFixed(1)}</strong></div>
              <div className="key-value"><span>Heart rate</span><strong>{selected.heart_rate_missing ? 'missing' : `${selected.hr_mean.toFixed(0)} bpm`}</strong></div>
              <div className="key-value"><span>Pre coverage</span><strong>{(selected.pre_coverage * 100).toFixed(0)}%</strong></div>
              <div className="key-value"><span>Post coverage</span><strong>{(selected.post_coverage * 100).toFixed(0)}%</strong></div>
            </div>

            {!selected.usable_for_memory && selected.exclusion_reason ? (
              <div className="callout muted">
                This window is excluded from retrieval because of {selected.exclusion_reason.replace(/_/g, ' ')}.
              </div>
            ) : null}
          </article>
        ) : null}
      </section>
    </section>
  );
}

