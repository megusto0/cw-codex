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

function formatMeanStd(mean?: number | null, std?: number | null) {
  if (mean === null || mean === undefined || std === null || std === undefined) {
    return '—';
  }
  return `${formatCompactNumber(mean, 3)} ± ${formatCompactNumber(std, 3)}`;
}

interface CompactFailureExample {
  query_id?: string;
  true_label_display?: string;
  retrieved_label_display?: string;
  top1_patient_relation?: string;
  short_explanation?: string;
  why_difficult?: string;
}

interface FailureBlock {
  top1_failures?: CompactFailureExample[];
  ambiguous_cases?: CompactFailureExample[];
  same_patient_dominant?: CompactFailureExample[];
  cross_patient_meaningful?: CompactFailureExample[];
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

  const neuralRows = useMemo(() => data?.comparison_rows.filter((row) => row.family === 'neural') ?? [], [data]);
  const baselineRows = useMemo(() => data?.comparison_rows.filter((row) => row.family === 'baseline') ?? [], [data]);
  const selectedRobustnessRows = useMemo(
    () => data?.robustness_summary?.rows.filter((row) => row.model_key === data.selected_model) ?? [],
    [data],
  );
  const primaryRobustnessRows = useMemo(
    () =>
      data?.robustness_summary?.rows.filter(
        (row) => row.mode === 'feature_mask' && Math.abs(row.level - 0.1) < 1e-6 && ['hopfield', 'siamese_temporal', 'som'].includes(row.model_key),
      ) ?? [],
    [data],
  );
  const selectedFailureBlock = useMemo(() => {
    if (!data?.failure_analysis) {
      return null;
    }
    return (data.failure_analysis as Record<string, FailureBlock>)[data.selected_model] ?? null;
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
            Основной блок показывает только retrieval-метрики: Top-1, Top-3 и MRR. Устойчивость,
            baseline-методы и неудачные случаи вынесены ниже, чтобы не смешивать разные типы диагностики.
          </p>
        </div>

        <div className="results-table">
          <div className="results-table-header">
            <span>Модель</span>
            <span>Семейство</span>
            <span>Top-1</span>
            <span>Top-3</span>
            <span>MRR</span>
          </div>
          {neuralRows.map((row) => (
            <div
              key={row.key}
              className={`results-table-row${row.key === data.selected_model ? ' selected' : ''}`}
            >
              <span>{row.label}</span>
              <span>{formatFamily(row.family)}</span>
              <span>{formatPercent(row.top1_accuracy)}</span>
              <span>{formatPercent(row.top3_hit_rate)}</span>
              <span>{formatCompactNumber(row.mean_reciprocal_rank, 3)}</span>
            </div>
          ))}
        </div>

        {selectedRow?.notes ? <div className="callout info">{selectedRow.notes}</div> : null}
      </article>

      <section className="two-column">
        <article className="panel">
          <div className="section-title">
            <span className="eyebrow">Сводный график</span>
            <p>Визуальное сравнение Top-1, Top-3 и MRR для трёх top-level neural retrieval моделей.</p>
          </div>
          <div className="chart-shell" style={{ height: 320 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.comparison_chart.data.filter((row) => row.family === 'neural')}>
                <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                <XAxis dataKey="label" tickLine={false} axisLine={false} interval={0} angle={-12} textAnchor="end" height={70} />
                <YAxis tickLine={false} axisLine={false} />
                <Tooltip />
                <Bar dataKey="top1_accuracy" name="Top-1" fill={CHART_COLORS.green} radius={[6, 6, 0, 0]} />
                <Bar dataKey="top3_hit_rate" name="Top-3" fill={CHART_COLORS.accent} radius={[6, 6, 0, 0]} />
                <Bar dataKey="mean_reciprocal_rank" name="MRR" fill={CHART_COLORS.blue} radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="panel">
          <div className="section-title">
            <span className="eyebrow">Устойчивость</span>
            <p>
              Коррупционный тест показывает, насколько Top-k retrieval сохраняется после маскирования
              признаков или численного возмущения входа.
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
              {selectedRobustnessRows.length ? (
                <div className="results-table small">
                  <div className="results-table-header">
                    <span>Искажение</span>
                    <span>Top-1 после</span>
                    <span>Снижение Top-1</span>
                    <span>Top-3 после</span>
                    <span>Снижение Top-3</span>
                  </div>
                  {selectedRobustnessRows.map((row) => (
                    <div key={`${row.mode}-${row.level}`} className="results-table-row">
                      <span>{row.label}</span>
                      <span>{formatPercent(row.top1_corrupted)}</span>
                      <span>{formatPercent(row.top1_drop)}</span>
                      <span>{formatPercent(row.top3_corrupted)}</span>
                      <span>{formatPercent(row.top3_drop)}</span>
                    </div>
                  ))}
                </div>
              ) : null}
            </>
          ) : (
            <div className="empty-panel">Для выбранной модели нет сохранённой кривой устойчивости.</div>
          )}
        </article>
      </section>

      <section className="two-column">
        <article className="panel">
          <div className="section-title">
            <span className="eyebrow">Базовые методы</span>
            <p>Baselines пересчитаны на том же 12-пациентном query pool и остаются evaluation-only.</p>
          </div>
          <div className="results-table small">
            <div className="results-table-header">
              <span>Метод</span>
              <span>Top-1</span>
              <span>Top-3</span>
              <span>MRR</span>
            </div>
            {baselineRows.map((row) => (
              <div key={row.key} className="results-table-row">
                <span>{row.label}</span>
                <span>{formatPercent(row.top1_accuracy)}</span>
                <span>{formatPercent(row.top3_hit_rate)}</span>
                <span>{formatCompactNumber(row.mean_reciprocal_rank, 3)}</span>
              </div>
            ))}
          </div>
          {data.additional_metrics.unavailable.length ? (
            <div className="callout warning">
              Недоступные baseline-методы: {data.additional_metrics.unavailable.map((item) => String(item.label)).join(', ')}.
            </div>
          ) : null}
        </article>

        <article className="panel">
          <div className="section-title">
            <span className="eyebrow">Сохранение Top-1 при 10% маскировании</span>
            <p>{data.robustness_summary?.definition}</p>
          </div>
          <div className="results-table small">
            <div className="results-table-header">
              <span>Модель</span>
              <span>Чистый Top-1</span>
              <span>Top-1 после</span>
              <span>Сохранение</span>
            </div>
            {primaryRobustnessRows.map((row) => (
              <div key={row.model_key} className="results-table-row">
                <span>{row.model_label}</span>
                <span>{formatPercent(row.top1_clean)}</span>
                <span>{formatPercent(row.top1_corrupted)}</span>
                <span>{formatPercent(row.top1_retention)}</span>
              </div>
            ))}
          </div>
        </article>
      </section>

      <section className="two-column">
        <article className="panel">
          <div className="section-title">
            <span className="eyebrow">Устойчивость по seed’ам</span>
            <p>{data.seed_stability?.note ?? 'Артефакт seed stability пока не рассчитан.'}</p>
          </div>
          <div className="results-table small">
            <div className="results-table-header">
              <span>Модель</span>
              <span>Top-1</span>
              <span>Top-3</span>
              <span>MRR</span>
            </div>
            {(data.seed_stability?.models ?? []).map((row) => (
              <div key={row.key} className="results-table-row">
                <span>{row.label}</span>
                <span>{formatMeanStd(row.summary.top1_accuracy.mean, row.summary.top1_accuracy.std)}</span>
                <span>{formatMeanStd(row.summary.top3_hit_rate.mean, row.summary.top3_hit_rate.std)}</span>
                <span>{formatMeanStd(row.summary.mean_reciprocal_rank.mean, row.summary.mean_reciprocal_rank.std)}</span>
              </div>
            ))}
          </div>
        </article>

        <article className="panel">
          <div className="section-title">
            <span className="eyebrow">Обобщение между пациентами</span>
            <p>Доли same/cross-patient показывают структуру извлечения, а не клиническую переносимость.</p>
          </div>
          <div className="results-table small">
            <div className="results-table-header">
              <span>Модель</span>
              <span>Same-patient</span>
              <span>Cross-patient</span>
            </div>
            {(data.patient_generalization ?? []).map((row) => (
              <div key={String(row.key)} className="results-table-row">
                <span>{String(row.label)}</span>
                <span>{formatPercent(row.same_patient_top1_rate as number | null | undefined)}</span>
                <span>{formatPercent(row.cross_patient_top1_rate as number | null | undefined)}</span>
              </div>
            ))}
          </div>
        </article>
      </section>

      <details className="panel disclosure-panel">
        <summary>Неудачные и неоднозначные случаи выбранной модели</summary>
        <div className="disclosure-content">
          {selectedFailureBlock ? (
            <div className="results-table small">
              <div className="results-table-header">
                <span>Тип</span>
                <span>Окно</span>
                <span>Истинная метка</span>
                <span>Top-1</span>
                <span>Причина сложности</span>
              </div>
              {[...(selectedFailureBlock.top1_failures ?? []), ...(selectedFailureBlock.ambiguous_cases ?? [])].slice(0, 6).map((item, index) => (
                <div key={`${item.query_id}-${index}`} className="results-table-row">
                  <span>{index < (selectedFailureBlock.top1_failures?.length ?? 0) ? 'Ошибка Top-1' : 'Неоднозначность'}</span>
                  <span>{item.query_id}</span>
                  <span>{item.true_label_display}</span>
                  <span>{item.retrieved_label_display}</span>
                  <span>{item.why_difficult}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-panel">Для выбранной модели нет компактного failure-analysis артефакта.</div>
          )}
        </div>
      </details>

      <article className="panel">
        <div className="section-title">
          <span className="eyebrow">Краткий вывод</span>
          <p>{data.conclusion}</p>
        </div>
      </article>

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
            </div>
          </div>
        </div>
      </details>
    </section>
  );
}
