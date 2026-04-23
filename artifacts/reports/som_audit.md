# Аудит оценки карты Кохонена

## Что проверено
- Запросные окна: 129
- Train-memory окна: 576
- Top-k ranking пересчитан из SOM weights, BMU, расстояния на карте и локального feature similarity.
- Проверено, что retrieved окна принадлежат train-memory и не совпадают с query identity.
- Проверена корректность same-patient / cross-patient маркера.
- Проверено, что модуль SOM не использует Siamese ranking artifacts.

## Результаты проверок
- Ошибки ранжирования: 0
- Ошибки leakage/self-retrieval: 0
- Ошибки same/cross patient: 0
- Зависимость от Siamese artifacts в SOM module: нет

## Сравнение сохранённых и пересчитанных метрик
| Метрика | Сохранено | Пересчитано | Абсолютная разница |
| --- | --- | --- | --- |
| top1_accuracy | 0.481 | 0.481 | 0.000 |
| top3_hit_rate | 0.775 | 0.775 | 0.000 |
| mean_reciprocal_rank | 0.629 | 0.629 | 0.000 |

## Вывод
Сильный результат SOM сохраняется после аудита: пересчитанные top-k метрики совпадают с сохранёнными, а признаков self-retrieval leakage или подмены Siamese ranking не обнаружено.