export interface DashboardData {
  title: string;
  subtitle: string;
  disclaimer: string;
  patients_count: number;
  total_meal_windows: number;
  usable_meal_windows: number;
  memory_size: number;
  feature_dimension: number;
  headline_metrics: Record<string, number>;
  headline_summary: string;
  label_distribution: Record<string, number>;
  split_distribution: Record<string, number>;
  exclusion_reasons: Record<string, number>;
  reused_visual_ideas: string[];
  not_reused_from_glucoscope: string[];
}

export interface WindowRecord {
  window_id: string;
  patient_id: string;
  meal_time: string;
  meal_type: string;
  meal_segment: string;
  carbs: number;
  bolus: number;
  has_bolus: number;
  carbs_per_unit: number;
  active_basal: number;
  baseline_glucose: number;
  trend_30m: number;
  trend_90m: number;
  premeal_mean: number;
  premeal_std: number;
  premeal_cv: number;
  hr_mean: number;
  hr_std: number;
  hr_min: number;
  hr_max: number;
  heart_rate_missing: number;
  response_peak: number;
  response_nadir: number;
  rise_from_baseline: number;
  post_range: number;
  post_cv: number;
  post_tir: number;
  label: string;
  label_display: string;
  label_reason: string;
  pre_coverage: number;
  post_coverage: number;
  split: string;
  usable_for_memory: boolean;
  exclusion_reason?: string | null;
  full_curve_minutes: number[];
  full_curve_values: Array<number | null>;
  full_curve_missingness: number[];
  premeal_values: number[];
  premeal_delta: number[];
  premeal_missingness: number[];
  memory_index?: number | null;
}

export interface RetrievalStep {
  step: number;
  vector: number[];
  energy: number;
  entropy: number;
  top_weight_gap: number;
  dominant_memory: {
    window_id: string;
    label: string;
    patient_id: string;
  };
}

export interface RetrievedMemory {
  index: number;
  window_id: string;
  label: string;
  patient_id: string;
  similarity: number;
  weight: number;
  same_patient: boolean;
  window: WindowRecord;
  feature_block_similarity: Record<string, number>;
  top_blocks: Array<[string, number]>;
  explanation_text: string;
}

export interface RetrievalResponse {
  query_window: WindowRecord;
  query_vector: number[];
  recalled_vector: number[];
  recalled_steps: RetrievalStep[];
  top_k_memories: RetrievedMemory[];
  similarities: number[];
  weights: number[];
  energy_values: number[];
  prototype_distribution: Record<string, number>;
  plain_language_explanation_text: string;
}

export interface PrototypeRecord {
  label: string;
  label_display: string;
  support_size: number;
  purity: number;
  vector: number[];
  mean_curve_minutes: number[];
  mean_curve_values: number[];
  representative_window_ids: string[];
  representative_windows?: WindowRecord[];
  typical_context: {
    carbs: number;
    bolus: number;
    baseline_glucose: number;
    trend_30m: number;
    trend_90m: number;
    meal_segment_mode: string;
  };
}

export interface BaselineMetrics {
  accuracy: number;
  balanced_accuracy: number;
  macro_f1: number;
  weighted_f1: number;
  confusion_matrix: {
    labels: string[];
    matrix: number[][];
  };
}

export interface NoisePoint {
  mode: string;
  level: number;
  top1_accuracy: number;
  top3_hit_rate: number;
}

export interface QualitativeExample {
  window_id: string;
  patient_id: string;
  label: string;
  top_ids: string[];
  top_labels: string[];
  top_patients: string[];
  top1_correct: boolean;
  top3_hit: boolean;
  top5_hit: boolean;
  mrr: number;
  label_purity_top5: number;
  same_patient_rate: number;
  cross_patient_top1: boolean;
  energy_before: number;
  energy_after: number;
  energy_drop: number;
  attention_entropy: number;
  top_weight_gap: number;
  predicted_label: string;
  prototype_label: string;
  prototype_distribution: Record<string, number>;
}

export interface EvaluationData {
  retrieval_metrics: Record<string, number>;
  diagnostics: Record<string, number>;
  baselines: Record<string, BaselineMetrics>;
  per_patient: Array<{
    patient_id: string;
    top1_accuracy: number;
    top3_hit_rate: number;
    mrr: number;
    same_patient_rate: number;
    count: number;
  }>;
  noise_robustness: NoisePoint[];
  qualitative_examples: {
    successes: QualitativeExample[];
    failures: QualitativeExample[];
  };
  limitations: string[];
}

export interface AboutData {
  title: string;
  plain_language: string;
  why_memory_based: string[];
  vector_construction: {
    feature_blocks: string[];
    feature_dimension: number;
  };
  hopfield_equations: Record<string, string>;
  limitations: string[];
}

