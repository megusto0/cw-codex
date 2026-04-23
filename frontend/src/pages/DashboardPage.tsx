import { useEffect, useState } from 'react';
import MetricCard from '../components/MetricCard';
import ModelSwitch from '../components/ModelSwitch';
import { fetchModelJson } from '../api';
import { useModelPreference } from '../model-context';
import { formatCompactNumber, formatPercent } from '../presentation';
import type { OverviewData } from '../types';

function OverviewError({ message }: { message: string }) {
  return (
    <section className="page">
      <div className="panel empty-panel">Не удалось загрузить обзор исследования: {message}</div>
    </section>
  );
}

export default function DashboardPage() {
  const { model } = useModelPreference();
  const [data, setData] = useState<OverviewData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setError(null);
    fetchModelJson<OverviewData>('/api/dashboard', model)
      .then(setData)
      .catch((reason: Error) => setError(reason.message));
  }, [model]);

  if (error) {
    return <OverviewError message={error} />;
  }

  if (!data) {
    return (
      <section className="page">
        <div className="panel empty-panel">Загрузка обзорной сводки…</div>
      </section>
    );
  }

  const distributionMax = Math.max(...data.chart.data.map((item) => item.value), 1);
  const exclusionEntries = Object.entries(data.exclusions);
  const exclusionMax = Math.max(...exclusionEntries.map(([, count]) => count), 1);

  return (
    <section className="page">
      <header className="page-header page-header-compact">
        <div className="section-title">
          <span className="eyebrow">Ретроспективный исследовательский прототип</span>
          <h2>{data.title}</h2>
          <p>{data.subtitle}</p>
          <div className="callout risk">{data.disclaimer}</div>
        </div>
        <div className="page-switch sticky-switch">
          <ModelSwitch />
        </div>
      </header>

      <section className="metric-grid metric-grid-4">
        <MetricCard label="Top-1 retrieval" value={formatPercent(data.headline_metrics.top1_accuracy)} />
        <MetricCard label="Top-3 hit rate" value={formatPercent(data.headline_metrics.top3_hit_rate)} />
        <MetricCard label="MRR" value={formatCompactNumber(data.headline_metrics.mean_reciprocal_rank, 3)} />
        <MetricCard
          label={data.headline_metrics.representation_label}
          value={String(data.headline_metrics.representation_size)}
          hint={
            data.headline_metrics.noise_stability !== null && data.headline_metrics.noise_stability !== undefined
              ? `noise stability ${formatPercent(data.headline_metrics.noise_stability)}`
              : undefined
          }
        />
      </section>

      <section className="dataset-strip panel">
        <div>
          <span>Пациенты</span>
          <strong>{data.dataset_strip.patients}</strong>
        </div>
        <div>
          <span>Выделенные окна</span>
          <strong>{data.dataset_strip.extracted_windows}</strong>
        </div>
        <div>
          <span>Пригодные окна</span>
          <strong>{data.dataset_strip.usable_windows}</strong>
        </div>
        <div>
          <span>Память / train</span>
          <strong>{data.dataset_strip.memory_windows}</strong>
        </div>
      </section>

      <section className="two-column">
        <article className="panel">
          <div className="section-title">
            <span className="eyebrow">Текущая модель</span>
            <h3>{data.selected_model.label}</h3>
            <p>{data.selected_model.scientific_description}</p>
          </div>

          <div className="compact-table">
            <div className="compact-table-header">
              <span>Модель</span>
              <span>Top-1</span>
              <span>Top-3</span>
              <span>MRR</span>
            </div>
            {data.model_comparison.map((item) => (
              <div
                key={item.key}
                className={`compact-table-row${item.key === data.selected_model.key ? ' selected' : ''}`}
              >
                <span>{item.label}</span>
                <span>{formatPercent(item.top1_accuracy)}</span>
                <span>{formatPercent(item.top3_hit_rate)}</span>
                <span>{formatCompactNumber(item.mean_reciprocal_rank, 3)}</span>
              </div>
            ))}
          </div>
        </article>

        <article className="panel">
          <div className="section-title">
            <span className="eyebrow">{data.chart.title}</span>
            <p>Тонкая сводка по структуре ретроспективных исследовательских меток.</p>
          </div>

          <div className="bar-list">
            {data.chart.data.map((item) => (
              <div key={item.key} className="bar-row">
                <span>{item.label}</span>
                <div className="bar-track">
                  <div className="bar-fill" style={{ width: `${(item.value / distributionMax) * 100}%` }} />
                </div>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>
        </article>
      </section>

      <article className="panel">
        <div className="section-title">
          <span className="eyebrow">Краткая интерпретация</span>
          <p>{data.interpretation}</p>
        </div>
      </article>

      <details className="panel disclosure-panel">
        <summary>Ограничения и прозрачность фильтрации</summary>
        <div className="disclosure-content">
          <div className="two-column">
            <div>
              <h3>Методологические ограничения</h3>
              <ul className="plain-list">
                {data.limitations.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
            <div>
              <h3>Исключения окон</h3>
              <div className="bar-list">
                {exclusionEntries.map(([reason, count]) => (
                  <div key={reason} className="bar-row">
                    <span>{reason}</span>
                    <div className="bar-track">
                      <div className="bar-fill muted" style={{ width: `${(count / exclusionMax) * 100}%` }} />
                    </div>
                    <strong>{count}</strong>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </details>
    </section>
  );
}
