import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import WindowCurveChart from '../components/WindowCurveChart';
import { fetchJson } from '../api';
import type { PrototypeRecord } from '../types';

export default function PrototypesPage() {
  const [prototypes, setPrototypes] = useState<PrototypeRecord[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchJson<PrototypeRecord[]>('/api/prototypes')
      .then(setPrototypes)
      .catch((reason: Error) => setError(reason.message));
  }, []);

  if (error) {
    return <section className="page"><div className="panel">Failed to load prototypes: {error}</div></section>;
  }

  if (!prototypes.length) {
    return <section className="page"><div className="panel">Loading prototypes...</div></section>;
  }

  return (
    <section className="page">
      <header className="page-header">
        <div>
          <span className="eyebrow">Prototype Memory</span>
          <h2>Representative Remembered Patterns</h2>
          <p>Each prototype summarizes a historical response family. It is descriptive memory, not recommended therapy.</p>
        </div>
      </header>

      <section className="prototype-grid">
        {prototypes.map((prototype) => (
          <article key={prototype.label} className="panel">
            <div className="detail-header">
              <div>
                <span className="eyebrow">Prototype</span>
                <h3>{prototype.label_display}</h3>
              </div>
              <span className="badge accent">Purity {(prototype.purity * 100).toFixed(0)}%</span>
            </div>

            <WindowCurveChart
              minutes={prototype.mean_curve_minutes}
              series={[
                {
                  key: prototype.label,
                  label: prototype.label_display,
                  color: '#1f8a70',
                  values: prototype.mean_curve_values,
                },
              ]}
              height={220}
            />

            <div className="detail-grid compact">
              <div className="key-value"><span>Support size</span><strong>{prototype.support_size}</strong></div>
              <div className="key-value"><span>Typical carbs</span><strong>{prototype.typical_context.carbs.toFixed(1)} g</strong></div>
              <div className="key-value"><span>Typical bolus</span><strong>{prototype.typical_context.bolus.toFixed(1)} U</strong></div>
              <div className="key-value"><span>Baseline</span><strong>{prototype.typical_context.baseline_glucose.toFixed(0)} mg/dL</strong></div>
              <div className="key-value"><span>30m trend</span><strong>{prototype.typical_context.trend_30m.toFixed(1)}</strong></div>
              <div className="key-value"><span>Main segment</span><strong>{prototype.typical_context.meal_segment_mode}</strong></div>
            </div>

            <div className="stack-list">
              {prototype.representative_window_ids.map((windowId) => (
                <Link key={windowId} to={`/retrieval?windowId=${windowId}`} className="text-link">
                  Inspect representative case {windowId}
                </Link>
              ))}
            </div>
          </article>
        ))}
      </section>
    </section>
  );
}

