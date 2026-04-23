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
    return (
      <section className="page">
        <div className="panel empty-panel">Не удалось загрузить методологическое описание: {error}</div>
      </section>
    );
  }

  if (!data) {
    return (
      <section className="page">
        <div className="panel empty-panel">Загрузка методологии…</div>
      </section>
    );
  }

  return (
    <section className="page">
      <header className="page-header page-header-compact">
        <div className="section-title">
          <span className="eyebrow">Методология</span>
          <h2>{data.title}</h2>
          <p>{data.intro}</p>
        </div>
      </header>

      <div className="two-column">
        {data.sections.map((section) => (
          <article key={section.title} className="panel">
            <div className="section-title">
              <span className="eyebrow">{section.title}</span>
            </div>
            <ul className="plain-list">
              {section.body.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </article>
        ))}
      </div>

      <article className="panel">
        <div className="section-title">
          <span className="eyebrow">Краткие формулы</span>
          <p>Формулы приведены только для тех этапов, которые действительно нужны для защиты методики.</p>
        </div>
        <div className="formula-grid">
          {data.formulas.map((item) => (
            <div key={item.title} className="formula-card">
              <strong>{item.title}</strong>
              <code>{item.formula}</code>
            </div>
          ))}
        </div>
      </article>
    </section>
  );
}
