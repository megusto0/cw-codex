import { useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import WindowCurveChart from '../components/WindowCurveChart';
import { postJson } from '../api';
import type { RetrievalResponse } from '../types';

export default function RetrievalPage() {
  const [searchParams] = useSearchParams();
  const windowId = searchParams.get('windowId');
  const [data, setData] = useState<RetrievalResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!windowId) return;
    setData(null);
    setError(null);
    postJson<RetrievalResponse>('/api/memory/retrieve', { window_id: windowId })
      .then(setData)
      .catch((reason: Error) => setError(reason.message));
  }, [windowId]);

  const prototypeData = useMemo(() => {
    if (!data) return [];
    return Object.entries(data.prototype_distribution).map(([label, value]) => ({
      label: label.replace(/_/g, ' '),
      weight: Number((value * 100).toFixed(1)),
    }));
  }, [data]);

  if (!windowId) {
    return (
      <section className="page">
        <div className="panel">
          Pick a query from the <Link to="/cases">Case Explorer</Link> to inspect associative retrieval.
        </div>
      </section>
    );
  }

  if (error) {
    return <section className="page"><div className="panel">Failed to run retrieval: {error}</div></section>;
  }

  if (!data) {
    return <section className="page"><div className="panel">Running associative retrieval...</div></section>;
  }

  const overlaySeries = [
    {
      key: 'query',
      label: 'Query meal window',
      color: '#1f8a70',
      values: data.query_window.full_curve_values,
    },
    ...data.top_k_memories.slice(0, 3).map((memory, index) => ({
      key: `match_${index}`,
      label: `Retrieved ${index + 1}`,
      color: ['#d97706', '#2563eb', '#8b5cf6'][index],
      values: memory.window.full_curve_values,
    })),
  ];

  const energyData = data.recalled_steps.map((step) => ({
    step: `Step ${step.step}`,
    energy: step.energy,
    entropy: step.entropy,
  }));

  return (
    <section className="page">
      <header className="page-header">
        <div>
          <span className="eyebrow">Similar Cases</span>
          <h2>Associative Retrieval</h2>
          <p>{data.plain_language_explanation_text}</p>
        </div>
      </header>

      <section className="two-column">
        <article className="panel">
          <span className="eyebrow">Query Meal Window</span>
          <div className="detail-grid compact">
            <div className="key-value"><span>Patient</span><strong>{data.query_window.patient_id}</strong></div>
            <div className="key-value"><span>Label</span><strong>{data.query_window.label_display}</strong></div>
            <div className="key-value"><span>Carbs</span><strong>{data.query_window.carbs.toFixed(1)} g</strong></div>
            <div className="key-value"><span>Bolus</span><strong>{data.query_window.bolus.toFixed(1)} U</strong></div>
            <div className="key-value"><span>Baseline</span><strong>{data.query_window.baseline_glucose.toFixed(0)} mg/dL</strong></div>
            <div className="key-value"><span>Segment</span><strong>{data.query_window.meal_segment}</strong></div>
          </div>
          <WindowCurveChart
            minutes={data.query_window.full_curve_minutes}
            series={[
              {
                key: 'query',
                label: 'Query',
                color: '#1f8a70',
                values: data.query_window.full_curve_values,
              },
            ]}
            height={240}
          />
        </article>

        <article className="panel">
          <span className="eyebrow">Prototype Match</span>
          <div style={{ width: '100%', height: 260 }}>
            <ResponsiveContainer>
              <BarChart data={prototypeData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(36, 56, 54, 0.09)" />
                <XAxis dataKey="label" tickLine={false} axisLine={false} interval={0} angle={-14} textAnchor="end" height={60} />
                <YAxis tickLine={false} axisLine={false} />
                <Tooltip />
                <Bar dataKey="weight" fill="#d97706" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="callout">
            Prototype weights summarize which remembered response family the recalled vector most resembles after Hopfield-style recall.
          </div>
        </article>
      </section>

      <article className="panel">
        <span className="eyebrow">Overlay of Query and Retrieved Similar Cases</span>
        <WindowCurveChart minutes={data.query_window.full_curve_minutes} series={overlaySeries} height={320} />
      </article>

      <section className="two-column">
        <article className="panel">
          <span className="eyebrow">Energy Trajectory</span>
          <div style={{ width: '100%', height: 260 }}>
            <ResponsiveContainer>
              <LineChart data={energyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(36, 56, 54, 0.09)" />
                <XAxis dataKey="step" tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} />
                <Tooltip />
                <Line dataKey="energy" stroke="#1f8a70" strokeWidth={2.5} dot />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="panel">
          <span className="eyebrow">Recall Steps</span>
          <div className="stack-list">
            {data.recalled_steps.map((step) => (
              <div key={step.step} className="step-card">
                <strong>Step {step.step}</strong>
                <span>Energy {step.energy.toFixed(3)}</span>
                <span>Entropy {step.entropy.toFixed(3)}</span>
                <span>Dominant memory {step.dominant_memory.window_id}</span>
              </div>
            ))}
          </div>
        </article>
      </section>

      <section className="retrieval-grid">
        {data.top_k_memories.map((memory, index) => (
          <article key={memory.window_id} className="panel">
            <div className="retrieved-header">
              <div>
                <span className="eyebrow">Retrieved {index + 1}</span>
                <h3>{memory.window.label_display}</h3>
                <p>{memory.explanation_text}</p>
              </div>
              <div className="stacked-metrics">
                <span className="badge accent">Similarity {memory.similarity.toFixed(2)}</span>
                <span className="badge neutral">Weight {(memory.weight * 100).toFixed(1)}%</span>
                <span className={`badge ${memory.same_patient ? 'neutral' : 'muted'}`}>
                  {memory.same_patient ? 'same patient' : 'cross patient'}
                </span>
              </div>
            </div>

            <div className="detail-grid compact">
              <div className="key-value"><span>Carbs</span><strong>{memory.window.carbs.toFixed(1)} g</strong></div>
              <div className="key-value"><span>Bolus</span><strong>{memory.window.bolus.toFixed(1)} U</strong></div>
              <div className="key-value"><span>Baseline</span><strong>{memory.window.baseline_glucose.toFixed(0)}</strong></div>
              <div className="key-value"><span>Meal segment</span><strong>{memory.window.meal_segment}</strong></div>
            </div>

            <div className="stack-list">
              {memory.top_blocks.map(([block, similarity]) => (
                <div key={block} className="key-value">
                  <span>{block.replace(/_/g, ' ')}</span>
                  <strong>{(similarity * 100).toFixed(0)}%</strong>
                </div>
              ))}
            </div>
          </article>
        ))}
      </section>
    </section>
  );
}

