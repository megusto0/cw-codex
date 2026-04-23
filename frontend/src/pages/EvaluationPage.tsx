import { useEffect, useMemo, useState } from 'react';
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
import ModelSwitch from '../components/ModelSwitch';
import { fetchModelJson } from '../api';
import { useModelPreference } from '../model-context';
import { CHART_COLORS, formatCompactNumber, formatFamily, formatMetricValue, formatPercent } from '../presentation';
import type { EvaluationData } from '../types';

function noiseModeLabel(value: string | undefined) {
  if (value === 'feature_mask') {
    return 'Маскирование признаков';
  }
  return 'Гауссов шум';
}

export default function EvaluationPage() {
  const { model } = useModelPreference();
  const [data, setData] = useState<EvaluationData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setError(null);
    fetchModelJson<EvaluationData>('/api/evaluation', model)
      .then(setData)
      .catch((reason: Error) => setError(reason.message));
  }, [model]);

  const selectedRow = useMemo(
    () => data?.comparison_rows.find((item) => item.key === data.selected_model) ?? null,
    [data],
  );

  const selectedNoiseSeries = useMemo(() => {
    if (!data) {
      return null;
    }
    const series =
      data.stability_chart.series.find((item) => item.key === data.selected_model) ??
      data.stability_chart.series[0] ??
      null;
    if (!series || !series.points.length) {
      return null;
    }
    const dominantMode = series.points[0]?.mode;
    return {
      label: series.label,
      mode: dominantMode,
      points: series.points
        .filter((item) => item.mode === dominantMode)
        .map((item) => ({
          level: item.level,
          top1_accuracy: item.top1_accuracy,
          top3_hit_rate: item.top3_hit_rate,
        })),
    };
  }, [data]);

  if (error) {
    return (
      <section className="page">
        <div className="panel empty-panel">Не удалось загрузить сравнительные результаты: {error}</div>
      </section>
    );
  }

  if (!data) {
    return (
      <section className="page">
        <div className="panel empty-panel">Загрузка сравнительного анализа…</div>
      </section>
    );
  }

  return (
    <section className="page">
      <header className="page-header page-header-compact">
        <div className="section-title">
          <span className="eyebrow">Сравнение retrieval-подходов</span>
          <h2>{data.title}</h2>
          <p>{data.subtitle}</p>
          <div className="callout info">{data.disclaimer}</div>
        </div>
        <div className="page-switch">
          <ModelSwitch />
        </div>
      </header>

      <article className="panel">
        <div className="section-title">
          <span className="eyebrow">Основные метрики</span>
          <p>
            Таблица показывает top-1 same-label retrieval, top-3 hit rate, MRR и noise stability. Именно
            эти четыре величины используются как основной сравнительный набор.
          </p>
        </div>

        <div className="results-table">
          <div className="results-table-header">
            <span>Модель</span>
            <span>Семейство</span>
            <span>Top-1</span>
            <span>Top-3</span>
            <span>MRR</span>
            <span>Noise stability</span>
          </div>
          {data.comparison_rows.map((row) => (
            <div
              key={row.key}
              className={`results-table-row${row.key === data.selected_model ? ' selected' : ''}`}
            >
              <span>{row.label}</span>
              <span>{formatFamily(row.family)}</span>
              <span>{formatPercent(row.top1_accuracy)}</span>
              <span>{formatPercent(row.top3_hit_rate)}</span>
              <span>{formatCompactNumber(row.mean_reciprocal_rank, 3)}</span>
              <span>{formatPercent(row.noise_stability)}</span>
            </div>
          ))}
        </div>

        {selectedRow?.notes ? <div className="callout info">{selectedRow.notes}</div> : null}
      </article>

      <section className="two-column">
        <article className="panel">
          <div className="section-title">
            <span className="eyebrow">Сводный график</span>
            <p>Визуальное сравнение top-1, top-3 и noise stability для всех доступных методов.</p>
          </div>
          <div className="chart-shell" style={{ height: 320 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.comparison_chart.data}>
                <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                <XAxis dataKey="label" tickLine={false} axisLine={false} interval={0} angle={-12} textAnchor="end" height={70} />
                <YAxis tickLine={false} axisLine={false} />
                <Tooltip />
                <Bar dataKey="top1_accuracy" name="Top-1" fill={CHART_COLORS.green} radius={[6, 6, 0, 0]} />
                <Bar dataKey="top3_hit_rate" name="Top-3" fill={CHART_COLORS.accent} radius={[6, 6, 0, 0]} />
                <Bar dataKey="noise_stability" name="Noise stability" fill={CHART_COLORS.blue} radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="panel">
          <div className="section-title">
            <span className="eyebrow">Устойчивость выбранной модели</span>
            <p>
              Для выбранной модели показывается деградация retrieval при одном типе искажения. Это
              локальная проверка устойчивости, а не клиническая оценка.
            </p>
          </div>
          {selectedNoiseSeries ? (
            <>
              <div className="chart-shell" style={{ height: 320 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={selectedNoiseSeries.points}>
                    <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                    <XAxis dataKey="level" tickLine={false} axisLine={false} />
                    <YAxis tickLine={false} axisLine={false} />
                    <Tooltip />
                    <Line dataKey="top1_accuracy" name="Top-1" stroke={CHART_COLORS.green} strokeWidth={2.2} dot={false} />
                    <Line dataKey="top3_hit_rate" name="Top-3" stroke={CHART_COLORS.accent} strokeWidth={2.2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <p className="chart-caption">
                Показан режим: {noiseModeLabel(selectedNoiseSeries.mode)}. Модель: {selectedNoiseSeries.label}.
              </p>
            </>
          ) : (
            <div className="empty-panel">Для выбранной модели нет сохранённой кривой устойчивости.</div>
          )}
        </article>
      </section>

      <section className="two-column">
        <article className="panel">
          <div className="section-title">
            <span className="eyebrow">Прототипные структуры</span>
            <p>
              Блок показывает, как каждая нейросетевая семья агрегирует локальные области памяти или карты.
            </p>
          </div>
          <div className="prototype-strip">
            {data.prototype_block.map((block) => (
              <div key={block.model} className="prototype-card">
                <strong>{block.label}</strong>
                <ul className="plain-list compact">
                  {block.items.map((item) => (
                    <li key={`${block.model}-${item.title}`}>
                      <span>{item.title}</span>
                      <span>
                        support {item.support}, purity {formatCompactNumber(item.purity, 2)}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </article>

        <article className="panel">
          <div className="section-title">
            <span className="eyebrow">Краткий вывод</span>
            <p>{data.conclusion}</p>
          </div>
        </article>
      </section>

      <details className="panel disclosure-panel">
        <summary>Дополнительные метрики</summary>
        <div className="disclosure-content">
          <div className="two-column">
            <div>
              <h3>Нейросетевые модели</h3>
              <div className="results-table small">
                <div className="results-table-header">
                  <span>Модель</span>
                  <span>Top-5</span>
                  <span>same-patient</span>
                  <span>cross-patient</span>
                  <span>Macro-F1</span>
                </div>
                {data.additional_metrics.models.map((row) => (
                  <div key={String(row.key)} className="results-table-row">
                    <span>{String(row.label)}</span>
                    <span>{formatMetricValue(row.top5_hit_rate as number | null | undefined, 3)}</span>
                    <span>{formatMetricValue(row.same_patient_top1 as number | null | undefined, 3)}</span>
                    <span>{formatMetricValue(row.cross_patient_top1 as number | null | undefined, 3)}</span>
                    <span>{formatMetricValue(row.macro_f1 as number | null | undefined, 3)}</span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h3>Базовые методы</h3>
              <div className="results-table small">
                <div className="results-table-header">
                  <span>Метод</span>
                  <span>Top-5</span>
                  <span>Macro-F1</span>
                  <span>Balanced acc.</span>
                  <span>Label purity</span>
                </div>
                {data.additional_metrics.baselines.map((row) => (
                  <div key={String(row.key)} className="results-table-row">
                    <span>{String(row.label)}</span>
                    <span>{formatMetricValue(row.top5_hit_rate as number | null | undefined, 3)}</span>
                    <span>{formatMetricValue(row.macro_f1 as number | null | undefined, 3)}</span>
                    <span>{formatMetricValue(row.balanced_accuracy as number | null | undefined, 3)}</span>
                    <span>{formatMetricValue(row.label_purity_top5 as number | null | undefined, 3)}</span>
                  </div>
                ))}
              </div>
              {data.additional_metrics.unavailable.length ? (
                <div className="callout warning">
                  Недоступные baseline-методы: {data.additional_metrics.unavailable.map((item) => String(item.label)).join(', ')}.
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </details>
    </section>
  );
}
