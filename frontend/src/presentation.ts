export const CHART_COLORS = {
  accent: '#d4956a',
  green: '#5cb888',
  yellow: '#c9a84c',
  orange: '#d48a4c',
  red: '#c75f5f',
  blue: '#5b8ec7',
  cyan: '#6aafb8',
  purple: '#9a7cc7',
  textSecondary: '#8a8a8a',
  grid: 'rgba(255,255,255,0.08)',
  tooltipBackground: '#1e1e1e',
  tooltipBorder: 'rgba(255,255,255,0.10)',
};

const LABELS: Record<string, string> = {
  controlled_response: 'Контролируемый отклик',
  postprandial_spike: 'Постпрандиальный пик',
  late_low: 'Позднее снижение',
  unstable_response: 'Нестабильный отклик',
  ambiguous: 'Неоднозначный отклик',
  custom_query: 'Пользовательский запрос',
};

const SEGMENTS: Record<string, string> = {
  breakfast: 'Завтрак',
  lunch: 'Обед',
  dinner: 'Ужин',
  overnight: 'Ночной интервал',
};

const SPLITS: Record<string, string> = {
  train: 'Память',
  val: 'Валидация',
  test: 'Отложенная проверка',
  excluded: 'Исключено',
  query: 'Запрос',
};

const FAMILIES: Record<string, string> = {
  neural: 'Нейросетевые модели',
  baseline: 'Базовые методы',
};

export function formatPercent(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '—';
  }
  return `${(value * 100).toFixed(1)}%`;
}

export function formatLabel(value?: string | null) {
  if (!value) {
    return '—';
  }
  return LABELS[value] ?? value.replace(/_/g, ' ');
}

export function formatSegment(value?: string | null) {
  if (!value) {
    return '—';
  }
  return SEGMENTS[value] ?? value;
}

export function formatSplit(value?: string | null) {
  if (!value) {
    return '—';
  }
  return SPLITS[value] ?? value;
}

export function formatDateTime(value: string) {
  return new Date(value).toLocaleString('ru-RU', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function formatFamily(value: string) {
  return FAMILIES[value] ?? value;
}

export function formatMetricValue(value: string | number | null | undefined, digits = 3) {
  if (value === null || value === undefined) {
    return '—';
  }
  if (typeof value === 'string') {
    return value;
  }
  return Number.isInteger(value) ? String(value) : value.toFixed(digits);
}

export function formatCompactNumber(value: number | null | undefined, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '—';
  }
  return Number.isInteger(value) ? String(value) : value.toFixed(digits);
}

export function formatRelation(value: 'same' | 'cross' | string | null | undefined) {
  if (value === 'same') {
    return 'Тот же пациент';
  }
  if (value === 'cross') {
    return 'Другой пациент';
  }
  return '—';
}

export function truncateReason(value: string, maxLength = 110) {
  if (!value) {
    return 'Краткое объяснение отсутствует.';
  }
  const firstSentence = value.split('. ')[0]?.trim();
  if (firstSentence && firstSentence.length <= maxLength) {
    return firstSentence.endsWith('.') ? firstSentence : `${firstSentence}.`;
  }
  return value.length > maxLength ? `${value.slice(0, maxLength - 1).trim()}…` : value;
}
