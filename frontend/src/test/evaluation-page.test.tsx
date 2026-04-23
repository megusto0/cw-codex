import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ModelProvider } from '../model-context';
import EvaluationPage from '../pages/EvaluationPage';

function makeEvaluationPayload(selectedModel: 'hopfield' | 'siamese_temporal' | 'som') {
  return {
    title: 'Сравнение моделей',
    subtitle: 'Сравнение нейросетевых семейств в задаче поиска сходных постпрандиальных CGM-окон',
    disclaimer: 'Сравнение моделей носит исследовательский характер и не предназначено для клинической интерпретации.',
    selected_model: selectedModel,
    comparison_rows: [
      {
        key: 'hopfield',
        label: 'Память Хопфилда',
        family: 'neural',
        available: true,
        top1_accuracy: 0.41,
        top3_hit_rate: 0.62,
        mean_reciprocal_rank: 0.51,
        noise_stability: 0.99,
        secondary_metrics: { average_energy_drop: 0.12 },
        additional_metrics: { top5_hit_rate: 0.66, macro_f1: 0.17 },
        noise_points: [
          { mode: 'gaussian_noise', level: 0.0, top1_accuracy: 0.41, top3_hit_rate: 0.62 },
          { mode: 'gaussian_noise', level: 0.1, top1_accuracy: 0.39, top3_hit_rate: 0.59 },
        ],
        notes: 'Ассоциативная память полезна для интерпретации recall-траектории.',
      },
      {
        key: 'siamese_temporal',
        label: 'Сиамская temporal-модель',
        family: 'neural',
        available: true,
        top1_accuracy: 0.53,
        top3_hit_rate: 0.66,
        mean_reciprocal_rank: 0.61,
        noise_stability: 0.94,
        secondary_metrics: { neighborhood_purity: 0.46 },
        additional_metrics: { top5_hit_rate: 0.75, macro_f1: 0.31 },
        noise_points: [
          { mode: 'gaussian_noise', level: 0.0, top1_accuracy: 0.53, top3_hit_rate: 0.66 },
          { mode: 'gaussian_noise', level: 0.1, top1_accuracy: 0.45, top3_hit_rate: 0.66 },
        ],
        notes: 'Temporal-энкодер лучше удерживает retrieval-качество в эмбеддинговом пространстве.',
      },
      {
        key: 'som',
        label: 'Карта Кохонена',
        family: 'neural',
        available: true,
        top1_accuracy: 0.45,
        top3_hit_rate: 0.60,
        mean_reciprocal_rank: 0.54,
        noise_stability: 0.9,
        secondary_metrics: { quantization_error: 0.23 },
        additional_metrics: { top5_hit_rate: 0.68, macro_f1: 0.22 },
        noise_points: [
          { mode: 'gaussian_noise', level: 0.0, top1_accuracy: 0.45, top3_hit_rate: 0.6 },
          { mode: 'gaussian_noise', level: 0.1, top1_accuracy: 0.4, top3_hit_rate: 0.57 },
        ],
        notes: 'Карта самоорганизации удобна для анализа локальной структуры.',
      },
      {
        key: 'cosine_knn',
        label: 'cosine kNN',
        family: 'baseline',
        available: true,
        top1_accuracy: 0.36,
        top3_hit_rate: 0.54,
        mean_reciprocal_rank: 0.44,
        noise_stability: 0.84,
        secondary_metrics: {},
        additional_metrics: { top5_hit_rate: 0.61, macro_f1: 0.14, balanced_accuracy: 0.18, label_purity_top5: 0.34 },
      },
    ],
    comparison_chart: {
      kind: 'primary_metrics',
      data: [
        { label: 'Память Хопфилда', top1_accuracy: 0.41, top3_hit_rate: 0.62, mean_reciprocal_rank: 0.51, noise_stability: 0.99, family: 'neural' },
        { label: 'Сиамская temporal-модель', top1_accuracy: 0.53, top3_hit_rate: 0.66, mean_reciprocal_rank: 0.61, noise_stability: 0.94, family: 'neural' },
        { label: 'Карта Кохонена', top1_accuracy: 0.45, top3_hit_rate: 0.6, mean_reciprocal_rank: 0.54, noise_stability: 0.9, family: 'neural' },
      ],
    },
    stability_chart: {
      kind: 'noise_robustness',
      series: [
        {
          key: 'hopfield',
          label: 'Память Хопфилда',
          family: 'neural',
          points: [
            { mode: 'gaussian_noise', level: 0.0, top1_accuracy: 0.41, top3_hit_rate: 0.62 },
            { mode: 'gaussian_noise', level: 0.1, top1_accuracy: 0.39, top3_hit_rate: 0.59 },
          ],
        },
        {
          key: 'siamese_temporal',
          label: 'Сиамская temporal-модель',
          family: 'neural',
          points: [
            { mode: 'gaussian_noise', level: 0.0, top1_accuracy: 0.53, top3_hit_rate: 0.66 },
            { mode: 'gaussian_noise', level: 0.1, top1_accuracy: 0.45, top3_hit_rate: 0.66 },
          ],
        },
        {
          key: 'som',
          label: 'Карта Кохонена',
          family: 'neural',
          points: [
            { mode: 'gaussian_noise', level: 0.0, top1_accuracy: 0.45, top3_hit_rate: 0.6 },
            { mode: 'gaussian_noise', level: 0.1, top1_accuracy: 0.4, top3_hit_rate: 0.57 },
          ],
        },
      ],
    },
    prototype_block: [
      { model: 'hopfield', label: 'Память Хопфилда', items: [{ title: 'Постпрандиальный пик', support: 20, purity: 0.75 }] },
      { model: 'siamese_temporal', label: 'Сиамская temporal-модель', items: [{ title: 'Постпрандиальный пик', support: 22, purity: 0.8 }] },
      { model: 'som', label: 'Карта Кохонена', items: [{ title: 'Контролируемый отклик', support: 18, purity: 0.7 }] },
    ],
    additional_metrics: {
      models: [
        { key: 'hopfield', label: 'Память Хопфилда', top5_hit_rate: 0.66, same_patient_top1: 0.3, cross_patient_top1: 0.7, macro_f1: 0.17 },
        { key: 'siamese_temporal', label: 'Сиамская temporal-модель', top5_hit_rate: 0.75, same_patient_top1: 0.45, cross_patient_top1: 0.55, macro_f1: 0.31 },
      ],
      baselines: [
        { key: 'cosine_knn', label: 'cosine kNN', top5_hit_rate: 0.61, macro_f1: 0.14, balanced_accuracy: 0.18, label_purity_top5: 0.34 },
      ],
      unavailable: [],
    },
    conclusion: 'На данной малой выборке модели демонстрируют разные сильные стороны и не дают оснований для клинических выводов.',
  };
}

describe('EvaluationPage', () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/evaluation?model=siamese_temporal')) {
        return new Response(JSON.stringify(makeEvaluationPayload('siamese_temporal')), { status: 200 });
      }
      if (url.includes('/api/evaluation?model=som')) {
        return new Response(JSON.stringify(makeEvaluationPayload('som')), { status: 200 });
      }
      return new Response(JSON.stringify(makeEvaluationPayload('hopfield')), { status: 200 });
    });
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('renders and refetches when switching models', async () => {
    render(
      <MemoryRouter initialEntries={['/evaluation?model=hopfield']}>
        <ModelProvider>
          <EvaluationPage />
        </ModelProvider>
      </MemoryRouter>,
    );

    expect(await screen.findByText('Сравнение моделей')).toBeInTheDocument();

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/evaluation?model=hopfield'), expect.any(Object));
    });

    expect(screen.getAllByText('Память Хопфилда').length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole('tab', { name: 'Карта Кохонена' }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/evaluation?model=som'), expect.any(Object));
    });

    expect(screen.getByText('Карта самоорганизации удобна для анализа локальной структуры.')).toBeInTheDocument();
  });
});
