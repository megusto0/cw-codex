import { useEffect, useState } from 'react';
import MetricCard from '../components/MetricCard';
import { fetchJson } from '../api';
import type { DashboardData } from '../types';

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export default function DashboardPage() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchJson<DashboardData>('/api/dashboard')
      .then(setData)
      .catch((reason: Error) => setError(reason.message));
  }, []);

  if (error) {
    return <section className="page"><div className="panel">Failed to load dashboard: {error}</div></section>;
  }

  if (!data) {
    return <section className="page"><div className="panel">Loading dashboard...</div></section>;
  }

  return (
    <section className="page">
      <header className="hero panel">
        <div className="hero-copy">
          <span className="eyebrow">Retrospective Coursework Demo</span>
          <h2>{data.title}</h2>
          <p>{data.subtitle}</p>
          <div className="callout">{data.disclaimer}</div>
        </div>
        <div className="hero-stats">
          <MetricCard label="Top-1 Retrieval" value={formatPercent(data.headline_metrics.top1_accuracy)} />
          <MetricCard label="Top-3 Retrieval" value={formatPercent(data.headline_metrics.top3_hit_rate)} />
          <MetricCard label="Memory Size" value={String(data.memory_size)} />
        </div>
      </header>

      <section className="metric-grid">
        <MetricCard label="Patients" value={String(data.patients_count)} />
        <MetricCard label="Extracted Windows" value={String(data.total_meal_windows)} />
        <MetricCard label="Usable Windows" value={String(data.usable_meal_windows)} />
        <MetricCard label="Feature Dimension" value={String(data.feature_dimension)} />
        <MetricCard label="MRR" value={data.headline_metrics.mean_reciprocal_rank.toFixed(3)} />
        <MetricCard label="Prototype Purity" value={data.headline_metrics.prototype_purity.toFixed(3)} />
      </section>

      <section className="two-column">
        <article className="panel">
          <span className="eyebrow">Headline Result</span>
          <p className="lead-text">{data.headline_summary}</p>
          <div className="distribution-list">
            {Object.entries(data.label_distribution).map(([label, count]) => (
              <div key={label} className="distribution-row">
                <span>{label.replace(/_/g, ' ')}</span>
                <div>
                  <div
                    className="distribution-bar"
                    style={{ width: `${(count / Math.max(...Object.values(data.label_distribution))) * 100}%` }}
                  />
                </div>
                <strong>{count}</strong>
              </div>
            ))}
          </div>
        </article>

        <article className="panel">
          <span className="eyebrow">Filtering Transparency</span>
          <div className="stack-list">
            {Object.entries(data.exclusion_reasons).map(([reason, count]) => (
              <div key={reason} className="key-value">
                <span>{reason.replace(/_/g, ' ')}</span>
                <strong>{count}</strong>
              </div>
            ))}
          </div>
        </article>
      </section>

      <section className="two-column">
        <article className="panel">
          <span className="eyebrow">Visual Ideas Reused</span>
          <ul className="plain-list">
            {data.reused_visual_ideas.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </article>

        <article className="panel">
          <span className="eyebrow">What Was Not Reused</span>
          <ul className="plain-list">
            {data.not_reused_from_glucoscope.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </article>
      </section>
    </section>
  );
}

