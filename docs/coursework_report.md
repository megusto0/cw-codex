# RL Therapy Lab - Hopfield Postprandial Memory

## 1. Motivation
This coursework project studies whether a modern Hopfield-style associative memory can retrieve interpretable similar postprandial cases from a small OhioT1DM-derived meal-window dataset.

## 2. Dataset
- Patients: 6
- Total meal windows extracted: 1303
- Usable retrospective windows: 474

## 3. Meal-window Extraction
Each meal window uses a -90 to 0 minute pre-meal CGM context and a 0 to +180 minute post-meal response window. Windows with overlapping meals or insufficient CGM coverage are excluded and tracked transparently.

## 4. Feature Encoding
The vector combines pre-meal CGM shape, delta-from-baseline, missingness markers, meal context, time context, patient identity, and heart-rate statistics when available.

## 5. Hopfield Associative Memory Method
Memory vectors from the train split are stored in a continuous Hopfield-style retrieval matrix. A held-out query is recalled iteratively using similarity-weighted updates and energy diagnostics.

## 6. Prototypes
- Controlled response: support 78, purity 0.67, typical carbs 31.0 g.
- Postprandial spike: support 182, purity 0.67, typical carbs 38.0 g.
- Late low: support 21, purity 0.40, typical carbs 35.0 g.
- Unstable response: support 3, purity 0.13, typical carbs 18.0 g.

## 7. Baselines
The main retrieval comparison uses cosine kNN, nearest prototype matching, patient-majority labeling, and an optional logistic-regression classifier.

## 8. Experiments
Held-out evaluation is chronological per patient. The report emphasizes retrieval quality, prototype quality, and robustness instead of building a large classifier zoo.

## 9. Results
- Hopfield top-1 same-label accuracy: 0.411
- Hopfield top-3 hit rate: 0.616
- Hopfield top-5 hit rate: 0.658
- Mean reciprocal rank: 0.505
- Average energy drop after recall: 0.123

## 10. Failure Analysis
- Query 588-20211024195800-257 (Postprandial spike) was pulled toward Controlled response with weight gap 0.020.
- Query 570-20220117194400-139 (Postprandial spike) was pulled toward Controlled response with weight gap 0.020.
- Query 588-20211021072000-245 (Postprandial spike) was pulled toward Ambiguous response with weight gap 0.013.

## 11. Interface Overview
The frontend provides a dashboard, case explorer, retrieval page, prototype gallery, evaluation page, and methodology page focused on similar-case interpretation.

## 12. Limitations
- The dataset contains only six OhioT1DM participants, so cross-patient generalization is limited.
- Research labels are deterministic retrospective categories, not clinical truth.
- The memory vectors encode pre-meal context and do not claim to forecast treatment outcomes.
- Heart-rate coverage varies by patient, so wearable context is informative but incomplete.

## 13. Conclusion
The project demonstrates that associative retrieval can remain valuable even when classification metrics are only moderate, because the remembered cases, prototype structure, and robustness diagnostics are directly inspectable.

## Appendix: Qualitative Successes
- Query 575-20220108191100-274 (Postprandial spike) retrieved the same label at rank 1 with weight gap 0.020.
- Query 570-20220120115400-148 (Postprandial spike) retrieved the same label at rank 1 with weight gap 0.011.
- Query 559-20220127170000-178 (Postprandial spike) retrieved the same label at rank 1 with weight gap 0.008.