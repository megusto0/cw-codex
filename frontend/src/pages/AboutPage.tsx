import { useEffect, useState } from 'react';
import { fetchJson } from '../api';
import type { AboutData } from '../types';

export default function AboutPage() {
  const [data, setData] = useState<AboutData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchJson<AboutData>('/api/about')
      .then(setData)
      .catch((reason: Error) => setError(reason.message));
  }, []);

  if (error) {
    return <section className="page"><div className="panel">Failed to load methodology: {error}</div></section>;
  }

  if (!data) {
    return <section className="page"><div className="panel">Loading methodology...</div></section>;
  }

  return (
    <section className="page">
      <header className="page-header">
        <div>
          <span className="eyebrow">About / Methodology</span>
          <h2>{data.title}</h2>
          <p>{data.plain_language}</p>
        </div>
      </header>

      <section className="two-column">
        <article className="panel">
          <span className="eyebrow">Why Memory-Based</span>
          <ul className="plain-list">
            {data.why_memory_based.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </article>

        <article className="panel">
          <span className="eyebrow">Feature Blocks</span>
          <ul className="plain-list">
            {data.vector_construction.feature_blocks.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
          <div className="callout">Feature dimension: {data.vector_construction.feature_dimension}</div>
        </article>
      </section>

      <article className="panel">
        <span className="eyebrow">Hopfield Retrieval</span>
        <div className="equation-grid">
          {Object.entries(data.hopfield_equations).map(([label, equation]) => (
            <div key={label} className="equation-card">
              <strong>{label.replace(/_/g, ' ')}</strong>
              <code>{equation}</code>
            </div>
          ))}
        </div>
      </article>

      <article className="panel">
        <span className="eyebrow">Not a Clinical System</span>
        <ul className="plain-list">
          {data.limitations.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </article>
    </section>
  );
}

