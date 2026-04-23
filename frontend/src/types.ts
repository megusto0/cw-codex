export type ModelKey = 'hopfield' | 'siamese_temporal' | 'som';

export interface ModelDescriptor {
  key: ModelKey;
  label: string;
  scientific_description: string;
  short_description?: string;
  representation_name?: string;
  prototype_name?: string;
  similarity_name?: string;
  supports_iterative_recall?: boolean;
}

export interface WindowRecord {
  window_id: string;
  patient_id: string;
  meal_time: string;
  meal_type: string;
  meal_segment: string;
  meal_segment_display?: string;
  carbs: number;
  bolus: number;
  baseline_glucose: number;
  trend_30m: number;
  trend_90m: number;
  hr_mean: number;
  heart_rate_missing: number | boolean;
  label: string;
  label_display?: string;
  label_reason?: string;
  split: string;
  split_display?: string;
  usable_for_memory: boolean;
  exclusion_reason_display?: string | null;
  full_curve_minutes: number[];
  full_curve_values: Array<number | null>;
}

export interface ModelsResponse {
  default_model: ModelKey;
  models: ModelDescriptor[];
}

export interface OverviewData {
  title: string;
  subtitle: string;
  disclaimer: string;
  selected_model: ModelDescriptor;
  available_models: ModelDescriptor[];
  headline_metrics: {
    top1_accuracy: number;
    top3_hit_rate: number;
    mean_reciprocal_rank: number;
    representation_size: number;
    representation_label: string;
    representation_value?: string;
    noise_stability?: number | null;
  };
  dataset_strip: {
    patients: number;
    extracted_windows: number;
    usable_windows: number;
    memory_windows: number;
  };
  model_comparison: Array<{
    key: ModelKey;
    label: string;
    scientific_description: string;
    top1_accuracy: number;
    top3_hit_rate: number;
    mean_reciprocal_rank: number;
    noise_stability?: number | null;
    secondary_metrics: Record<string, number | null>;
    additional_metrics: Record<string, number | null | undefined>;
  }>;
  chart: {
    kind: 'label_distribution';
    title: string;
    data: Array<{ key: string; label: string; value: number }>;
  };
  interpretation: string;
  limitations: string[];
  exclusions: Record<string, number>;
}

export interface RetrieveNeighbor {
  rank: number;
  window_id: string;
  label: string;
  label_display?: string;
  patient_id: string;
  similarity: number;
  same_patient: boolean;
  relation_badge: string;
  reason: string;
  map_distance?: number | null;
  window: WindowRecord;
}

export interface RetrieveResponse {
  model: ModelDescriptor;
  query_window: WindowRecord;
  summary_text: string;
  primary_metrics: Record<string, string | number | null>;
  neighbors: RetrieveNeighbor[];
  chart_payload:
    | {
        kind: 'energy_trajectory';
        points: Array<{ step: number; energy: number }>;
      }
    | {
        kind: 'curve_overlay';
        minutes: number[];
        series: Array<{ key: string; label: string; values: Array<number | null> }>;
      }
    | {
        kind: 'som_grid';
        grid_height: number;
        grid_width: number;
        active_cell: number;
        cells: Array<{
          cell_index: number;
          row: number;
          col: number;
          count: number;
          dominant_label?: string | null;
          dominant_label_display?: string;
          purity?: number;
        }>;
      };
  advanced: Record<string, unknown>;
}

export interface EvaluationRow {
  key: string;
  label: string;
  family: 'neural' | 'baseline';
  available: boolean;
  top1_accuracy: number | null;
  top3_hit_rate: number | null;
  mean_reciprocal_rank: number | null;
  noise_stability: number | null;
  corruption_retention_top1_10?: number | null;
  secondary_metrics: Record<string, number | null>;
  additional_metrics?: Record<string, number | null | undefined>;
  noise_points?: Array<{
    mode: string;
    level: number;
    top1_accuracy: number;
    top3_hit_rate: number;
  }>;
  notes?: string;
}

export interface EvaluationData {
  title: string;
  subtitle: string;
  disclaimer: string;
  selected_model: ModelKey;
  comparison_rows: EvaluationRow[];
  comparison_chart: {
    kind: 'primary_metrics';
    data: Array<{
      label: string;
      top1_accuracy: number | null;
      top3_hit_rate: number | null;
      mean_reciprocal_rank: number | null;
      noise_stability: number | null;
      corruption_retention_top1_10?: number | null;
      family: 'neural' | 'baseline';
    }>;
  };
  stability_chart: {
    kind: 'noise_robustness';
    series: Array<{
      key: string;
      label: string;
      family: 'neural' | 'baseline';
      points: Array<{
        mode: string;
        level: number;
        top1_accuracy: number;
        top3_hit_rate: number;
      }>;
    }>;
  };
  prototype_block: Array<{
    model: string;
    label: string;
    items: Array<{
      title: string;
      support: number;
      purity: number;
    }>;
  }>;
  additional_metrics: {
    models: Array<Record<string, string | number | null | undefined>>;
    baselines: Array<Record<string, string | number | null | undefined>>;
    unavailable: Array<Record<string, unknown>>;
  };
  robustness_summary?: {
    definition: string;
    rows: Array<{
      model_key: string;
      model_label: string;
      mode: string;
      mode_display: string;
      level: number;
      label: string;
      top1_clean: number;
      top3_clean: number;
      top1_corrupted: number;
      top3_corrupted: number;
      top1_drop: number;
      top3_drop: number;
      top1_retention?: number | null;
      top3_retention?: number | null;
    }>;
  };
  seed_stability?: {
    note: string;
    models: Array<{
      key: string;
      label: string;
      status: string;
      reason?: string | null;
      summary: {
        top1_accuracy: { mean: number | null; std: number | null };
        top3_hit_rate: { mean: number | null; std: number | null };
        mean_reciprocal_rank: { mean: number | null; std: number | null };
      };
    }>;
  } | null;
  failure_analysis?: Record<string, unknown>;
  patient_generalization?: Array<Record<string, string | number | null | undefined>>;
  conclusion: string;
}

export interface AboutData {
  title: string;
  intro: string;
  sections: Array<{
    title: string;
    body: string[];
  }>;
  formulas: Array<{
    title: string;
    formula: string;
  }>;
  available_models: ModelDescriptor[];
}
