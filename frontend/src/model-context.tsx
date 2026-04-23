import {
  createContext,
  type PropsWithChildren,
  useContext,
  useEffect,
  useMemo,
} from 'react';
import { useSearchParams } from 'react-router-dom';
import type { ModelKey } from './types';

export interface ModelOption {
  key: ModelKey;
  label: string;
  scientificDescription: string;
}

const MODEL_STORAGE_KEY = 'cw-codex:model';

export const MODEL_OPTIONS: ModelOption[] = [
  {
    key: 'hopfield',
    label: 'Память Хопфилда',
    scientificDescription: 'Ассоциативная память с итеративным восстановлением состояния и энергетической диагностикой.',
  },
  {
    key: 'siamese_temporal',
    label: 'Сиамская temporal-модель',
    scientificDescription: 'Нейросетевая temporal-модель метрического пространства для поиска сходных постпрандиальных окон.',
  },
  {
    key: 'som',
    label: 'Карта Кохонена',
    scientificDescription: 'Карта самоорганизации для топологического анализа структуры данных и локального retrieval.',
  },
];

function normalizeModel(value: string | null | undefined): ModelKey {
  if (value === 'siamese' || value === 'siamese_temporal') {
    return 'siamese_temporal';
  }
  if (value === 'som') {
    return 'som';
  }
  return 'hopfield';
}

interface ModelContextValue {
  model: ModelKey;
  setModel: (value: ModelKey) => void;
  options: ModelOption[];
  linkWithModel: (path: string, params?: Record<string, string | number | null | undefined>) => string;
}

const ModelContext = createContext<ModelContextValue | null>(null);

function readInitialModel(searchParams: URLSearchParams): ModelKey {
  const fromUrl = searchParams.get('model');
  if (fromUrl) {
    return normalizeModel(fromUrl);
  }
  if (typeof window === 'undefined') {
    return 'hopfield';
  }
  return normalizeModel(window.localStorage.getItem(MODEL_STORAGE_KEY));
}

export function ModelProvider({ children }: PropsWithChildren) {
  const [searchParams, setSearchParams] = useSearchParams();
  const model = normalizeModel(searchParams.get('model') ?? readInitialModel(searchParams));

  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(MODEL_STORAGE_KEY, model);
    }
    if (searchParams.get('model') !== model) {
      const next = new URLSearchParams(searchParams);
      next.set('model', model);
      setSearchParams(next, { replace: true });
    }
  }, [model, searchParams, setSearchParams]);

  const value = useMemo<ModelContextValue>(
    () => ({
      model,
      setModel: (nextModel) => {
        const next = new URLSearchParams(searchParams);
        next.set('model', nextModel);
        setSearchParams(next, { replace: true });
        if (typeof window !== 'undefined') {
          window.localStorage.setItem(MODEL_STORAGE_KEY, nextModel);
        }
      },
      options: MODEL_OPTIONS,
      linkWithModel: (path, params = {}) => {
        const [pathname, rawQuery = ''] = path.split('?');
        const next = new URLSearchParams(rawQuery);
        next.set('model', model);
        Object.entries(params).forEach(([key, value]) => {
          if (value === null || value === undefined || value === '') {
            next.delete(key);
            return;
          }
          next.set(key, String(value));
        });
        const query = next.toString();
        return query ? `${pathname}?${query}` : pathname;
      },
    }),
    [model, searchParams, setSearchParams],
  );

  return <ModelContext.Provider value={value}>{children}</ModelContext.Provider>;
}

export function useModelPreference() {
  const value = useContext(ModelContext);
  if (!value) {
    throw new Error('useModelPreference must be used inside ModelProvider');
  }
  return value;
}
