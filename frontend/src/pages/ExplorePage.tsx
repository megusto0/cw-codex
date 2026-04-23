import { Fragment, useEffect, useMemo, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import MetricCard from '../components/MetricCard';
import ModelSwitch from '../components/ModelSwitch';
import SomGridChart from '../components/SomGridChart';
import WindowCurveChart from '../components/WindowCurveChart';
import { fetchJson, retrieveWindow } from '../api';
import { useModelPreference } from '../model-context';
import {
  CHART_COLORS,
  formatCompactNumber,
  formatDateTime,
  formatLabel,
  formatMetricValue,
  formatRelation,
  formatSegment,
  formatSplit,
  truncateReason,
} from '../presentation';
import type { RetrieveResponse, WindowRecord } from '../types';

interface FiltersState {
  patientId: string;
  label: string;
  split: string;
  mealSegment: string;
  query: string;
}

const INITIAL_FILTERS: FiltersState = {
  patientId: 'all',
  label: 'all',
  split: 'all',
  mealSegment: 'all',
  query: '',
};

function buildMetricCards(data: RetrieveResponse) {
  if (data.model.key === 'hopfield') {
    return [
      { label: 'Top-1 similarity', value: formatMetricValue(data.primary_metrics.top1_similarity) },
      { label: 'Уровень уверенности', value: formatMetricValue(data.primary_metrics.confidence_level) },
      { label: 'Gap top-1/top-2', value: formatMetricValue(data.primary_metrics.top12_gap) },
      { label: 'Снижение энергии', value: formatMetricValue(data.primary_metrics.energy_drop) },
    ];
  }

  if (data.model.key === 'siamese_temporal') {
    return [
      { label: 'Косинусное сходство', value: formatMetricValue(data.primary_metrics.top1_similarity) },
      { label: 'Gap top-1/top-2', value: formatMetricValue(data.primary_metrics.top12_gap) },
      { label: 'Чистота окрестности', value: formatMetricValue(data.primary_metrics.neighborhood_purity) },
      { label: 'Top-1 сосед', value: formatRelation(data.primary_metrics.patient_relation as string | null | undefined) },
    ];
  }

  return [
    { label: 'BMU confidence', value: formatMetricValue(data.primary_metrics.bmu_confidence) },
    { label: 'Cluster purity', value: formatMetricValue(data.primary_metrics.cluster_purity) },
    { label: 'Quantization error', value: formatMetricValue(data.primary_metrics.quantization_error) },
    { label: 'Topographic error', value: formatMetricValue(data.primary_metrics.topographic_error) },
  ];
}

function renderFeatureBlockList(blocks: unknown) {
  const entries = Object.entries((blocks as Record<string, unknown>) ?? {})
    .filter(([, value]) => typeof value === 'number')
    .sort((a, b) => Number(b[1]) - Number(a[1]));

  if (!entries.length) {
    return <div className="empty-panel">Детализация по блокам признаков отсутствует.</div>;
  }

  const maxValue = Math.max(...entries.map(([, value]) => Number(value)), 1);
  return (
    <div className="bar-list compact">
      {entries.map(([key, value]) => (
        <div key={key} className="bar-row">
          <span>{key}</span>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${(Number(value) / maxValue) * 100}%` }} />
          </div>
          <strong>{formatCompactNumber(Number(value), 3)}</strong>
        </div>
      ))}
    </div>
  );
}

function renderModelChart(data: RetrieveResponse) {
  if (data.chart_payload.kind === 'curve_overlay') {
    return (
      <WindowCurveChart
        minutes={data.chart_payload.minutes}
        series={data.chart_payload.series}
        height={260}
      />
    );
  }

  if (data.chart_payload.kind === 'som_grid') {
    return (
      <SomGridChart
        gridHeight={data.chart_payload.grid_height}
        gridWidth={data.chart_payload.grid_width}
        activeCell={data.chart_payload.active_cell}
        cells={data.chart_payload.cells}
      />
    );
  }

  return (
    <div className="chart-shell" style={{ height: 260 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data.chart_payload.points}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
          <XAxis dataKey="step" tickLine={false} axisLine={false} />
          <YAxis tickLine={false} axisLine={false} />
          <Tooltip />
          <Line dataKey="energy" name="Энергия" stroke={CHART_COLORS.accent} strokeWidth={2.2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function chartCaption(modelKey: string) {
  if (modelKey === 'hopfield') {
    return 'Траектория энергии показывает, как ассоциативная память стабилизирует состояние запроса.';
  }
  if (modelKey === 'siamese_temporal') {
    return 'Наложение кривых показывает, какие исторические окна оказались ближайшими в эмбеддинговом пространстве.';
  }
  return 'Карта Кохонена показывает локальную топологическую область, в которую попадает выбранное окно.';
}

function neighborMetricLabel(modelKey: string) {
  return modelKey === 'som' ? 'Map distance' : 'Similarity';
}

export default function ExplorePage() {
  const { model } = useModelPreference();
  const [catalog, setCatalog] = useState<WindowRecord[]>([]);
  const [filters, setFilters] = useState<FiltersState>(INITIAL_FILTERS);
  const [selectedWindowId, setSelectedWindowId] = useState<string | null>(null);
  const [expandedNeighborId, setExpandedNeighborId] = useState<string | null>(null);
  const [result, setResult] = useState<RetrieveResponse | null>(null);
  const [loadingCatalog, setLoadingCatalog] = useState(true);
  const [loadingResult, setLoadingResult] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoadingCatalog(true);
    fetchJson<WindowRecord[]>('/api/windows?limit=1500')
      .then((items) => {
        setCatalog(items);
      })
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setLoadingCatalog(false));
  }, []);

  const patients = useMemo(
    () => Array.from(new Set(catalog.map((item) => item.patient_id))).sort(),
    [catalog],
  );
  const labels = useMemo(
    () => Array.from(new Set(catalog.map((item) => item.label))).sort(),
    [catalog],
  );

  const filteredWindows = useMemo(() => {
    const query = filters.query.trim().toLowerCase();
    return catalog.filter((item) => {
      if (filters.patientId !== 'all' && item.patient_id !== filters.patientId) {
        return false;
      }
      if (filters.label !== 'all' && item.label !== filters.label) {
        return false;
      }
      if (filters.split !== 'all' && item.split !== filters.split) {
        return false;
      }
      if (filters.mealSegment !== 'all' && item.meal_segment !== filters.mealSegment) {
        return false;
      }
      if (query && !item.window_id.toLowerCase().includes(query)) {
        return false;
      }
      return true;
    });
  }, [catalog, filters]);

  useEffect(() => {
    if (!filteredWindows.length) {
      setSelectedWindowId(null);
      setResult(null);
      return;
    }
    if (!selectedWindowId || !filteredWindows.some((item) => item.window_id === selectedWindowId)) {
      setSelectedWindowId(filteredWindows[0].window_id);
    }
  }, [filteredWindows, selectedWindowId]);

  const selectedWindow = useMemo(
    () => catalog.find((item) => item.window_id === selectedWindowId) ?? null,
    [catalog, selectedWindowId],
  );

  useEffect(() => {
    setExpandedNeighborId(null);

    if (!selectedWindow) {
      setResult(null);
      return;
    }

    if (!selectedWindow.usable_for_memory) {
      setResult(null);
      return;
    }

    setLoadingResult(true);
    setError(null);
    retrieveWindow<RetrieveResponse>(model, selectedWindow.window_id, 5)
      .then(setResult)
      .catch((reason: Error) => {
        setResult(null);
        setError(reason.message);
      })
      .finally(() => setLoadingResult(false));
  }, [model, selectedWindow]);

  if (error && !catalog.length) {
    return (
      <section className="page">
        <div className="panel empty-panel">Не удалось загрузить данные для поиска случаев: {error}</div>
      </section>
    );
  }

  const metricCards = result ? buildMetricCards(result) : [];
  const featureBlocks = (result?.advanced.feature_block_similarity ?? {}) as Record<string, unknown>;

  return (
    <section className="page">
      <header className="page-header page-header-compact">
        <div className="section-title">
          <span className="eyebrow">Поиск сходных случаев</span>
          <h2>Для выбранного окна что показывает текущая модель?</h2>
          <p>Слева выбирается постпрандиальное окно, справа отображается retrieval-результат выбранной модели.</p>
        </div>
        <div className="page-switch sticky-switch">
          <ModelSwitch />
        </div>
      </header>

      <section className="explore-layout">
        <aside className="panel explorer-panel">
          <div className="section-title">
            <span className="eyebrow">Выбор окна</span>
            <p>
              Компактная таблица предназначена для выбора окна; подробности выводятся только для
              текущего запроса и раскрытого соседа.
            </p>
          </div>

          <div className="filters">
            <label className="field">
              <span>Пациент</span>
              <select
                value={filters.patientId}
                onChange={(event) => setFilters((state) => ({ ...state, patientId: event.target.value }))}
              >
                <option value="all">Все</option>
                {patients.map((patient) => (
                  <option key={patient} value={patient}>
                    {patient}
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span>Метка</span>
              <select
                value={filters.label}
                onChange={(event) => setFilters((state) => ({ ...state, label: event.target.value }))}
              >
                <option value="all">Все</option>
                {labels.map((label) => (
                  <option key={label} value={label}>
                    {formatLabel(label)}
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span>Split</span>
              <select
                value={filters.split}
                onChange={(event) => setFilters((state) => ({ ...state, split: event.target.value }))}
              >
                <option value="all">Все</option>
                <option value="train">Память</option>
                <option value="val">Валидация</option>
                <option value="test">Отложенная проверка</option>
                <option value="excluded">Исключено</option>
              </select>
            </label>

            <label className="field">
              <span>Сегмент приёма пищи</span>
              <select
                value={filters.mealSegment}
                onChange={(event) => setFilters((state) => ({ ...state, mealSegment: event.target.value }))}
              >
                <option value="all">Все</option>
                <option value="breakfast">Завтрак</option>
                <option value="lunch">Обед</option>
                <option value="dinner">Ужин</option>
                <option value="overnight">Ночной интервал</option>
              </select>
            </label>

            <label className="field field-wide">
              <span>Поиск по window_id</span>
              <input
                type="text"
                value={filters.query}
                onChange={(event) => setFilters((state) => ({ ...state, query: event.target.value }))}
                placeholder="например, 559-20220127170000-178"
              />
            </label>
          </div>

          <div className="table-caption">
            Найдено окон: <strong>{filteredWindows.length}</strong>
          </div>

          <div className="window-list">
            {loadingCatalog ? (
              <div className="empty-panel">Загрузка списка окон…</div>
            ) : filteredWindows.length ? (
              filteredWindows.map((item) => (
                <button
                  key={item.window_id}
                  type="button"
                  className={`window-row${selectedWindowId === item.window_id ? ' selected' : ''}`}
                  onClick={() => setSelectedWindowId(item.window_id)}
                >
                  <span>{item.patient_id}</span>
                  <span>{formatDateTime(item.meal_time)}</span>
                  <span>{formatLabel(item.label)}</span>
                  <span>{item.carbs.toFixed(0)} г</span>
                  <span>{item.baseline_glucose.toFixed(0)}</span>
                  <span>{formatSplit(item.split)}</span>
                </button>
              ))
            ) : (
              <div className="empty-panel">Под выбранные фильтры окна не найдены.</div>
            )}
          </div>
        </aside>

        <div className="detail-stack">
          <article className="panel">
            {selectedWindow ? (
              <>
                <div className="section-title">
                  <span className="eyebrow">Запросное окно</span>
                  <h3>{selectedWindow.window_id}</h3>
                  <p>
                    Пациент {selectedWindow.patient_id}, {formatDateTime(selectedWindow.meal_time)}.
                  </p>
                </div>

                <div className="chip-list">
                  <span className="badge neutral">{formatLabel(selectedWindow.label)}</span>
                  <span className="badge neutral">{formatSegment(selectedWindow.meal_segment)}</span>
                  <span className="badge neutral">carbs {selectedWindow.carbs.toFixed(0)} г</span>
                  <span className="badge neutral">bolus {selectedWindow.bolus.toFixed(1)}</span>
                  <span className="badge neutral">baseline {selectedWindow.baseline_glucose.toFixed(0)}</span>
                  <span className="badge neutral">trend {selectedWindow.trend_30m.toFixed(0)}</span>
                  <span className="badge neutral">
                    HR {selectedWindow.heart_rate_missing ? 'неполный' : 'доступен'}
                  </span>
                </div>

                <WindowCurveChart
                  minutes={selectedWindow.full_curve_minutes}
                  series={[
                    {
                      key: 'query',
                      label: 'Запрос',
                      values: selectedWindow.full_curve_values,
                    },
                  ]}
                  height={240}
                />

                {!selectedWindow.usable_for_memory ? (
                  <div className="callout warning">
                    Это окно исключено из retrieval-анализа и не может быть использовано как запрос для
                    сравнения моделей.
                  </div>
                ) : null}
              </>
            ) : (
              <div className="empty-panel">Выберите окно в левой панели.</div>
            )}
          </article>

          <article className="panel">
            {!selectedWindow ? (
              <div className="empty-panel">Результат появится после выбора окна.</div>
            ) : !selectedWindow.usable_for_memory ? (
              <div className="empty-panel">
                Для исключённых окон retrieval не рассчитывается; используйте фильтр split или выберите другое окно.
              </div>
            ) : loadingResult ? (
              <div className="empty-panel">Вычисление retrieval-результата…</div>
            ) : result ? (
              <>
                <div className="section-title">
                  <span className="eyebrow">{result.model.label}</span>
                  <p>{result.summary_text}</p>
                </div>

                <section className="metric-grid metric-grid-4">
                  {metricCards.map((item) => (
                    <MetricCard key={item.label} label={item.label} value={String(item.value)} />
                  ))}
                </section>

                <div className="chart-block">
                  {renderModelChart(result)}
                  <p className="chart-caption">{chartCaption(result.model.key)}</p>
                </div>

                <div className="section-title">
                  <span className="eyebrow">Извлечённые похожие случаи</span>
                  <p>Почему эта модель выбрала именно эти случаи: соседство определяется текущим механизмом сходства.</p>
                </div>

                <div className="results-table neighbors">
                  <div className="results-table-header">
                    <span>Rank</span>
                    <span>Метка</span>
                    <span>Пациент</span>
                    <span>{neighborMetricLabel(result.model.key)}</span>
                    <span>Связь</span>
                    <span>Причина</span>
                  </div>
                  {result.neighbors.map((neighbor) => (
                    <Fragment key={neighbor.window_id}>
                      <button
                        type="button"
                        className={`results-table-row action-row${expandedNeighborId === neighbor.window_id ? ' selected' : ''}`}
                        onClick={() =>
                          setExpandedNeighborId((current) => (current === neighbor.window_id ? null : neighbor.window_id))
                        }
                      >
                        <span>{neighbor.rank}</span>
                        <span>{neighbor.label_display ?? formatLabel(neighbor.label)}</span>
                        <span>{neighbor.patient_id}</span>
                        <span>
                          {result.model.key === 'som'
                            ? formatMetricValue(neighbor.map_distance)
                            : formatMetricValue(neighbor.similarity)}
                        </span>
                        <span>
                          <span className={`badge ${neighbor.same_patient ? 'green' : 'blue'}`}>{neighbor.relation_badge}</span>
                        </span>
                        <span>{truncateReason(neighbor.reason)}</span>
                      </button>

                      {expandedNeighborId === neighbor.window_id ? (
                        <div className="neighbor-detail">
                          <WindowCurveChart
                            minutes={neighbor.window.full_curve_minutes}
                            series={[
                              {
                                key: 'query',
                                label: 'Запрос',
                                values: selectedWindow.full_curve_values,
                              },
                              {
                                key: 'neighbor',
                                label: `Сосед ${neighbor.rank}`,
                                values: neighbor.window.full_curve_values,
                              },
                            ]}
                            height={180}
                          />

                          <div className="two-column">
                            <div>
                              <h3>Краткая интерпретация</h3>
                              <p>{neighbor.reason}</p>
                            </div>
                            <div>
                              <h3>Сходство по блокам признаков</h3>
                              {renderFeatureBlockList(featureBlocks[neighbor.window_id])}
                            </div>
                          </div>
                        </div>
                      ) : null}
                    </Fragment>
                  ))}
                </div>

                <details className="disclosure-panel inline">
                  <summary>Дополнительная диагностика</summary>
                  <div className="disclosure-content">
                    <div className="two-column">
                      <div>
                        <h3>Блоки признаков</h3>
                        {renderFeatureBlockList(featureBlocks[result.neighbors[0]?.window_id ?? ''])}
                      </div>
                      <div>
                        <h3>Локальные опоры модели</h3>
                        {renderFeatureBlockList(
                          result.advanced.prototype_affinities ??
                            result.advanced.prototype_distribution ??
                            result.advanced.label_distribution,
                        )}
                        {'embedding_notes' in result.advanced && result.advanced.embedding_notes ? (
                          <p className="section-copy">{String(result.advanced.embedding_notes)}</p>
                        ) : null}
                      </div>
                    </div>
                  </div>
                </details>
              </>
            ) : (
              <div className="empty-panel">
                Retrieval-результат для текущего окна отсутствует. {error ? `Сообщение: ${error}` : ''}
              </div>
            )}
          </article>
        </div>
      </section>
    </section>
  );
}
