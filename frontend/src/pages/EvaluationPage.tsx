import { useEffect, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import MetricCard from '../components/MetricCard';
import { fetchJson } from '../api';
import type { EvaluationData } from '../types';

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export default function EvaluationPage() {
  const [data, setData] = useState<EvaluationData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchJson<EvaluationData>('/api/evaluation')
      .then(setData)
      .catch((reason: Error) => setError(reason.message));
  }, []);

  const baselineRows = useMemo(() => {
    if (!data) return [];
    return Object.entries(data.baselines).map(([name, metrics]) => ({
      name: name.replace(/_/g, ' '),
      balancedAccuracy: Number((metrics.balanced_accuracy * 100).toFixed(1)),
      macroF1: Number((metrics.macro_f1 * 100).toFixed(1)),
    }));
  }, [data]);

  if (error) {
    return <section className="page"><div className="panel">Failed to load evaluation: {error}</div></section>;
  }

  if (!data) {
    return <section className="page"><div className="panel">Loading evaluation...</div></section>;
  }

  return (
    <section className="page">
      <header className="page-header">
        <div>
          <span className="eyebrow">Evaluation</span>
          <h2>Retrieval Quality, Baselines, and Failure Modes</h2>
          <p>The emphasis is interpretability and robustness rather than chasing a better classifier number.</p>
        </div>
      </header>

      <section className="metric-grid">
        <MetricCard label="Top-1 Retrieval" value={formatPercent(data.retrieval_metrics.top1_accuracy)} />
        <MetricCard label="Top-3 Hit Rate" value={formatPercent(data.retrieval_metrics.top3_hit_rate)} />
        <MetricCard label="MRR" value={data.retrieval_metrics.mean_reciprocal_rank.toFixed(3)} />
        <MetricCard label="Energy Drop" value={data.diagnostics.average_energy_drop.toFixed(3)} />
        <MetricCard label="Entropy" value={data.diagnostics.average_attention_entropy.toFixed(3)} />
        <MetricCard label="Cross-Patient Top-1" value={formatPercent(data.diagnostics.cross_patient_top1_rate)} />
      </section>

      <section className="two-column">
        <article className="panel">
          <span className="eyebrow">Baseline Comparison</span>
          <div style={{ width: '100%', height: 280 }}>
            <ResponsiveContainer>
              <BarChart data={baselineRows}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(36, 56, 54, 0.09)" />
                <XAxis dataKey="name" tickLine={false} axisLine={false} interval={0} angle={-14} textAnchor="end" height={64} />
                <YAxis tickLine={false} axisLine={false} />
                <Tooltip />
                <Legend />
                <Bar dataKey="balancedAccuracy" fill="#1f8a70" radius={[8, 8, 0, 0]} />
                <Bar dataKey="macroF1" fill="#d97706" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="panel">
          <span className="eyebrow">Noise Robustness</span>
          <div style={{ width: '100%', height: 280 }}>
            <ResponsiveContainer>
              <LineChart data={data.noise_robustness}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(36, 56, 54, 0.09)" />
                <XAxis dataKey="level" tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} />
                <Tooltip />
                <Legend />
                <Line dataKey="top1_accuracy" name="Top-1" stroke="#1f8a70" strokeWidth={2.5} />
                <Line dataKey="top3_hit_rate" name="Top-3" stroke="#d97706" strokeWidth={2.5} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>
      </section>

      <article className="panel">
        <span className="eyebrow">Per-Patient Retrieval</span>
        <div style={{ width: '100%', height: 280 }}>
          <ResponsiveContainer>
            <BarChart data={data.per_patient}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(36, 56, 54, 0.09)" />
              <XAxis dataKey="patient_id" tickLine={false} axisLine={false} />
              <YAxis tickLine={false} axisLine={false} />
              <Tooltip />
              <Legend />
              <Bar dataKey="top1_accuracy" name="Top-1" fill="#1f8a70" radius={[8, 8, 0, 0]} />
              <Bar dataKey="top3_hit_rate" name="Top-3" fill="#d97706" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </article>

      <section className="two-column">
        <article className="panel">
          <span className="eyebrow">Selected Successes</span>
          <div className="stack-list">
            {data.qualitative_examples.successes.map((example) => (
              <div key={example.window_id} className="step-card">
                <strong>{example.window_id}</strong>
                <span>{example.label.replace(/_/g, ' ')}</span>
                <span>Top-1 correct, gap {example.top_weight_gap.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </article>

        <article className="panel">
          <span className="eyebrow">Selected Failures</span>
          <div className="stack-list">
            {data.qualitative_examples.failures.map((example) => (
              <div key={example.window_id} className="step-card">
                <strong>{example.window_id}</strong>
                <span>True label: {example.label.replace(/_/g, ' ')}</span>
                <span>Top retrieved label: {example.top_labels[0].replace(/_/g, ' ')}</span>
              </div>
            ))}
          </div>
        </article>
      </section>

      <article className="panel">
        <span className="eyebrow">Limitations</span>
        <ul className="plain-list">
          {data.limitations.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </article>
    </section>
  );
}

