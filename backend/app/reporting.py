from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from .config import Settings


def _load_json(path: Path) -> dict[str, Any] | list[dict[str, Any]] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pickle(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def _share(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _score(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def _compute_noise_stability(points: list[dict[str, Any]]) -> float | None:
    if not points:
        return None

    by_mode: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        by_mode.setdefault(str(point["mode"]), []).append(point)

    retentions: list[float] = []
    for mode_points in by_mode.values():
        ordered = sorted(mode_points, key=lambda item: float(item["level"]))
        clean = ordered[0]
        clean_score = (float(clean["top1_accuracy"]) + float(clean["top3_hit_rate"])) / 2.0
        if clean_score <= 0:
            continue
        for point in ordered[1:]:
            score = (float(point["top1_accuracy"]) + float(point["top3_hit_rate"])) / 2.0
            retentions.append(score / clean_score)
    if not retentions:
        return None
    return sum(retentions) / len(retentions)


def _corruption_retention_top1(points: list[dict[str, Any]]) -> float | None:
    if not points:
        return None
    feature_mask = [point for point in points if point.get("mode") == "feature_mask"]
    if not feature_mask:
        return None
    clean = next((point for point in feature_mask if float(point["level"]) == 0.0), None)
    corrupted = next((point for point in feature_mask if abs(float(point["level"]) - 0.1) < 1e-6), None)
    if not clean or not corrupted:
        return None
    clean_top1 = float(clean["top1_accuracy"])
    if clean_top1 <= 0:
        return None
    return float(corrupted["top1_accuracy"]) / clean_top1


def _best_supported_prototype(prototypes: dict[str, dict[str, Any]] | None) -> dict[str, Any] | None:
    if not prototypes:
        return None
    return max(prototypes.values(), key=lambda item: item.get("support_size", 0))


def _best_som_cell(bundle: dict[str, Any] | None) -> dict[str, Any] | None:
    if not bundle:
        return None
    cells = list((bundle.get("cell_stats") or {}).values())
    if not cells:
        return None
    return max(cells, key=lambda item: item.get("count", 0))


def _example_line(title: str, example: dict[str, Any] | None) -> str:
    if not example:
        return f"- {title}: характерный пример в текущем срезе не выделен."

    label = example.get("label_display") or example.get("label") or "не указано"
    predicted = example.get("predicted_label_display") or example.get("prototype_label") or "не указано"
    window_id = example.get("window_id", "не указано")
    summary = example.get("summary_text") or example.get("reason") or "Краткое объяснение отсутствует."
    return f"- {title}: {window_id}, истинная метка «{label}», ближайшее решение «{predicted}». {summary}"


def _comparison_row_text(row: dict[str, Any]) -> str:
    return (
        f"| {row['label']} | {_share(row.get('top1_accuracy'))} | {_share(row.get('top3_hit_rate'))} | "
        f"{_score(row.get('mean_reciprocal_rank'))} | {_share(row.get('corruption_retention_top1_10'))} |"
    )


def generate_coursework_report(
    bundle: dict[str, Any],
    settings: Settings,
) -> str:
    dashboard = bundle["dashboard"]
    hopfield_eval = bundle["evaluation"]
    hopfield_metrics = hopfield_eval["retrieval_metrics"]
    hopfield_noise = _compute_noise_stability(hopfield_eval.get("noise_robustness", []))
    hopfield_retention = _corruption_retention_top1(hopfield_eval.get("noise_robustness", []))

    siamese_metrics = _load_json(settings.siamese_metrics_path) or {}
    siamese_retrieval = (siamese_metrics.get("retrieval_metrics") or {})
    siamese_noise = _compute_noise_stability(siamese_metrics.get("noise_robustness", []))
    siamese_retention = _corruption_retention_top1(siamese_metrics.get("noise_robustness", []))

    som_metrics = _load_json(settings.som_metrics_path) or {}
    som_retrieval = (som_metrics.get("retrieval_metrics") or {})
    som_noise = _compute_noise_stability(som_metrics.get("noise_robustness", []))
    som_retention = _corruption_retention_top1(som_metrics.get("noise_robustness", []))

    baseline_rows = _load_json(settings.comparison_metrics_path) or []
    seed_stability = _load_json(settings.seed_stability_path) or {}
    siamese_prototypes = _load_json(settings.siamese_prototypes_path) or {}
    som_bundle = _load_pickle(settings.som_runtime_bundle_path) or {}

    neural_rows = [
        {
            "label": "Память Хопфилда",
            "top1_accuracy": hopfield_metrics.get("top1_accuracy"),
            "top3_hit_rate": hopfield_metrics.get("top3_hit_rate"),
            "mean_reciprocal_rank": hopfield_metrics.get("mean_reciprocal_rank"),
            "noise_stability": hopfield_noise,
            "corruption_retention_top1_10": hopfield_retention,
        },
        {
            "label": "Сиамская temporal-модель",
            "top1_accuracy": siamese_retrieval.get("top1_accuracy"),
            "top3_hit_rate": siamese_retrieval.get("top3_hit_rate"),
            "mean_reciprocal_rank": siamese_retrieval.get("mean_reciprocal_rank"),
            "noise_stability": siamese_noise,
            "corruption_retention_top1_10": siamese_retention,
        },
        {
            "label": "Карта Кохонена",
            "top1_accuracy": som_retrieval.get("top1_accuracy"),
            "top3_hit_rate": som_retrieval.get("top3_hit_rate"),
            "mean_reciprocal_rank": som_retrieval.get("mean_reciprocal_rank"),
            "noise_stability": som_noise,
            "corruption_retention_top1_10": som_retention,
        },
    ]

    baseline_lines: list[str] = []
    if isinstance(baseline_rows, list):
        for row in baseline_rows:
            if not row.get("available", True):
                baseline_lines.append(f"- {row['label']}: метрика не рассчитана в текущем артефактном срезе.")
                continue
            baseline_lines.append(
                f"- {row['label']}: top-1 {_share(row.get('top1_accuracy'))}, top-3 {_share(row.get('top3_hit_rate'))}, "
                f"MRR {_score(row.get('mean_reciprocal_rank'))}."
            )

    seed_lines: list[str] = []
    for row in seed_stability.get("models", []):
        if row.get("status") != "ok":
            seed_lines.append(f"- {row.get('label', row.get('key'))}: недоступно ({row.get('reason', 'причина не указана')}).")
            continue
        summary = row["summary"]
        seed_lines.append(
            f"- {row['label']}: Top-1 {_score(summary['top1_accuracy']['mean'])}±{_score(summary['top1_accuracy']['std'])}, "
            f"Top-3 {_score(summary['top3_hit_rate']['mean'])}±{_score(summary['top3_hit_rate']['std'])}, "
            f"MRR {_score(summary['mean_reciprocal_rank']['mean'])}±{_score(summary['mean_reciprocal_rank']['std'])}."
        )

    hopfield_proto = _best_supported_prototype(bundle["prototypes"])
    siamese_proto = _best_supported_prototype(siamese_prototypes if isinstance(siamese_prototypes, dict) else None)
    som_cell = _best_som_cell(som_bundle)

    hopfield_success = (hopfield_eval.get("qualitative_examples") or {}).get("successes", [None])[0]
    hopfield_failure = (hopfield_eval.get("qualitative_examples") or {}).get("failures", [None])[0]
    siamese_success = (siamese_metrics.get("qualitative_examples") or {}).get("successes", [None])[0]
    siamese_failure = (siamese_metrics.get("qualitative_examples") or {}).get("failures", [None])[0]
    som_success = (som_metrics.get("qualitative_examples") or {}).get("successes", [None])[0]
    som_failure = (som_metrics.get("qualitative_examples") or {}).get("failures", [None])[0]

    lines = [
        "# Сравнение нейросетевых подходов к поиску сходных постпрандиальных CGM-окон на выборке OhioT1DM",
        "",
        "## 1. Цель проекта",
        (
            "Проект исследует, как разные нейросетевые архитектуры работают в задаче поиска сходных "
            "постпрандиальных CGM-окон на малой выборке OhioT1DM. Основной вопрос формулируется не как "
            "поиск универсального победителя, а как сравнение поведения разных retrieval-подходов и "
            "проверка того, поддерживает ли доступный объём данных убедительный вывод о превосходстве одной модели."
        ),
        (
            "Система не является медицинским изделием, не формирует клинические рекомендации и не "
            "предназначена для выбора дозы инсулина. Полученные результаты следует интерпретировать как "
            "исследовательское сравнение retrieval-подходов, а не как доказательство клинической применимости."
        ),
        "",
        "## 2. Исходные данные OhioT1DM",
        f"- Пациенты: {dashboard['patients_count']}",
        f"- Выделенные окна приёма пищи: {dashboard['total_meal_windows']}",
        f"- Пригодные окна для retrieval-анализа: {dashboard['usable_meal_windows']}",
        f"- Окна памяти / train: {dashboard['memory_size']}",
        f"- Размерность исходного признакового пространства: {dashboard['feature_dimension']}",
        (
            "Используются CGM, контекст приёма пищи, болюс, базальный инсулин, временной контекст, "
            "идентификатор пациента и статистики частоты сердечных сокращений при наличии таких данных."
        ),
        "",
        "## 3. Формирование постпрандиальных окон",
        (
            "Каждое окно включает предпищевой CGM-фрагмент от -90 до 0 минут и постпрандиальный отклик "
            "от 0 до +180 минут. Окна с конфликтующими соседними приёмами пищи, отсутствующим baseline или "
            "недостаточным покрытием CGM исключаются с явной фиксацией причины."
        ),
        (
            "Для честного сравнения используется единый протокол split по всем моделям: одинаковый пул "
            "пригодных окон, единый train/test режим и явный запрет self-retrieval leakage."
        ),
        "",
        "## 4. Кодирование признаков",
        (
            "Общее признаковое представление объединяет форму предпищевого CGM, delta-from-baseline, "
            "маску пропусков, meal context, временной контекст, patient context и heart-rate context. "
            "Именно это общее представление используется как исходная основа для всех сравнений."
        ),
        "",
        "## 5. Сравниваемые модели",
        "### 5.1. Память Хопфилда",
        (
            "Память Хопфилда используется как ассоциативный retrieval-baseline. Запросное окно проходит "
            "итеративное восстановление состояния, после чего анализируются top-k совпадения, траектория "
            "энергии и распределение внимания по памяти."
        ),
        "### 5.2. Сиамская temporal-модель",
        (
            "Сиамская модель реализует временной энкодер 1D-CNN для предпищевой последовательности CGM, "
            "малоглубинный MLP для табличного контекста и общее эмбеддинговое пространство. Retrieval "
            "выполняется по косинусному сходству, а метки используются только для организации "
            "метрического пространства, а не для перевода проекта в классификационный benchmark."
        ),
        "### 5.3. Карта Кохонена",
        (
            "Карта Кохонена применяется как модель топологической самоорганизации. Она полезна не только "
            "для neighborhood-based retrieval, но и для визуального анализа локальной структуры признакового пространства."
        ),
        "",
        "## 6. Метрики и protocol fairness",
        (
            "Основными retrieval-метриками являются top-1 same-label retrieval, top-3 hit rate и MRR. "
            "Устойчивость оценивается отдельно через контролируемое маскирование признаков и численное "
            "возмущение входа; классификационные показатели остаются вторичными."
        ),
        (
            "На странице сравнения дополнительно показаны evaluation-only baselines: cosine kNN, DTW kNN, "
            "Soft-DTW kNN и nearest prototype. Они не формируют основную историю интерфейса и нужны только "
            "для умеренной калибровки результатов. Cache baseline-методов пересчитан для текущего 12-пациентного среза."
        ),
        (
            "Коррупционная устойчивость задаётся явно: модель оценивается на тех же query-окнах после "
            "10% и 20% маскирования признаков, а также малых и средних численных возмущений. В отчёте "
            "основным коротким индикатором служит сохранение Top-1 при 10% маскировании."
        ),
        "",
        "## 7. Экспериментальные результаты",
        "| Модель | Top-1 | Top-3 | MRR | Сохранение Top-1 при 10% маскировании |",
        "| --- | --- | --- | --- | --- |",
        *[_comparison_row_text(row) for row in neural_rows],
        "",
        "### Коррупционная устойчивость",
        f"- Память Хопфилда: сохранение Top-1 при 10% маскировании {_share(hopfield_retention)}.",
        f"- Сиамская temporal-модель: сохранение Top-1 при 10% маскировании {_share(siamese_retention)}.",
        f"- Карта Кохонена: сохранение Top-1 при 10% маскировании {_share(som_retention)}.",
        "",
        "### Устойчивость trainable режимов по seed’ам",
        *(seed_lines or ["- Seed-stability артефакт пока не рассчитан."]),
        "",
        "### Evaluation-only baselines",
        *baseline_lines,
        "",
        (
            "На текущем срезе Siamese temporal-модель и карта Кохонена демонстрируют более высокие retrieval-метрики, "
            "чем память Хопфилда. При этом победитель зависит от критерия: Siamese лучше интерпретируется как "
            "обученное метрическое пространство, тогда как SOM показывает наиболее сильный top-3 и MRR на этом split."
        ),
        "",
        "## 8. Прототипы и интерпретация представлений",
    ]

    if hopfield_proto:
        lines.append(
            f"- Память Хопфилда: наиболее поддержанный прототип «{hopfield_proto['label_display']}» "
            f"(support {hopfield_proto['support_size']}, purity {hopfield_proto['purity']:.2f}). "
            f"{hopfield_proto['interpretation_text']}"
        )
    if siamese_proto:
        lines.append(
            f"- Сиамская temporal-модель: наиболее поддержанный эмбеддинговый прототип «{siamese_proto['label_display']}» "
            f"(support {siamese_proto['support_size']}, purity {siamese_proto['purity']:.2f}). "
            f"{siamese_proto['interpretation_text']}"
        )
    if som_cell:
        lines.append(
            f"- Карта Кохонена: наиболее плотная локальная область — ячейка ({som_cell['row']}, {som_cell['col']}) "
            f"с dominant label «{som_cell['dominant_label_display']}», support {som_cell['count']} и purity {som_cell['purity']:.2f}. "
            "В отличие от двух других моделей, SOM описывает не классический прототип класса, а локальную топологическую область карты."
        )

    lines.extend(
        [
            "",
            "## 9. Анализ успешных и неудачных случаев",
            "### 9.1. Выраженные успешные совпадения",
            _example_line("Память Хопфилда", hopfield_success),
            _example_line("Сиамская temporal-модель", siamese_success),
            _example_line("Карта Кохонена", som_success),
            "",
            "### 9.2. Неудачные и неоднозначные совпадения",
            _example_line("Память Хопфилда", hopfield_failure),
            _example_line("Сиамская temporal-модель", siamese_failure),
            _example_line("Карта Кохонена", som_failure),
            "",
            "## 10. Почему retrieval-подход методологически оправдан на малой выборке",
            (
                "При ограниченном числе пациентов retrieval позволяет обсуждать не абстрактную точность модели, "
                "а конкретные соответствия между запросом и историческими окнами памяти. Такой подход лучше "
                "подходит для малой выборки, поскольку поддерживает case-to-case интерпретацию, анализ неудач "
                "и проверку структуры признакового пространства без завышенных заявлений о генерализации."
            ),
            "",
            "## 11. Почему проект не является задачей бесконечного подбора классификаторов",
            (
                "Интерфейс намеренно ограничен тремя top-level нейросетевыми режимами: память Хопфилда, "
                "Siamese temporal-модель и карта Кохонена. Базовые методы вынесены только на страницу "
                "сравнения, а близкие по смыслу ablation-варианты не превращаются в отдельные экранные режимы. "
                "Это сохраняет проект как coursework по сравнительному neural retrieval, а не как classifier zoo."
            ),
            "",
            "## 12. Методологические ограничения",
        ]
    )

    for limitation in hopfield_eval["limitations"]:
        lines.append(f"- {limitation}")

    lines.extend(
        [
            "- Различия между моделями наблюдаются на одной и той же малой выборке и не должны трактоваться как клиническое превосходство.",
            "- Величины для trainable моделей чувствительны к split, случайной инициализации и объёму доступных примеров памяти.",
            "",
            "## 13. Заключение",
            (
                "Доступные данные не дают оснований утверждать, что одна модель является универсально лучшей. "
                "Память Хопфилда остаётся наиболее наглядной с точки зрения ассоциативной интерпретации и "
                "энергетической диагностики; Siamese temporal-модель показывает сильное качество retrieval в "
                "обученном метрическом пространстве; карта Кохонена лучше всего проявляет топологическую "
                "структуру данных и локальные neighbourhood-связи. Поэтому итоговый вывод корректнее формулировать "
                "как сравнение сильных сторон разных neural retrieval families, а не как поиск клинического победителя."
            ),
        ]
    )

    return "\n".join(lines)
