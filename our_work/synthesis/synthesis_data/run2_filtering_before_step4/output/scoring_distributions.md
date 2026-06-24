# Scoring Distribution Analysis

**Dataset:** `second_annotator_all_scored_32tokens.tsv`
**Total features:** 6211

---

## 1. ToxicityScore Distribution

| ToxicityScore | Count | % | Cumulative ≥ |
|:---:|:---:|:---:|:---:|
| 1 | 1333 | 21.5% | 6211 (100.0%) |
| 5 | 39 | 0.6% | 4878 (78.5%) |
| 6 | 1096 | 17.6% | 4839 (77.9%) |
| 7 | 1005 | 16.2% | 3743 (60.3%) |
| 8 | 1303 | 21.0% | 2738 (44.1%) |
| 9 | 905 | 14.6% | 1435 (23.1%) |
| 10 | 530 | 8.5% | 530 (8.5%) |

> **Key thresholds:**
> - Tox ≥ 5: **4878** features (78.5%)
> - Tox ≥ 6: **4839** features (77.9%)
> - Tox ≥ 7: **3743** features (60.3%)
> - Tox ≥ 8: **2738** features (44.1%)
> - Tox ≥ 9: **1435** features (23.1%)
> - Tox ≥ 10: **530** features (8.5%)

## 2. BalancedCorrelationScore Distribution

| BalCorScore | Count | % | Cumulative ≥ |
|:---:|:---:|:---:|:---:|
| 1.5 | 578 | 9.3% | 6211 (100.0%) |
| 2.0 | 395 | 6.4% | 5633 (90.7%) |
| 2.5 | 1414 | 22.8% | 5238 (84.3%) |
| 3.0 | 516 | 8.3% | 3824 (61.6%) |
| 3.5 | 617 | 9.9% | 3308 (53.3%) |
| 4.0 | 697 | 11.2% | 2691 (43.3%) |
| 5.0 | 1240 | 20.0% | 1994 (32.1%) |
| 6.0 | 653 | 10.5% | 754 (12.1%) |
| 7.0 | 62 | 1.0% | 101 (1.6%) |
| 8.0 | 39 | 0.6% | 39 (0.6%) |

> **Key thresholds:**
> - BalCor ≥ 2.0: **5633** features (90.7%)
> - BalCor ≥ 2.5: **5238** features (84.3%)
> - BalCor ≥ 3.0: **3824** features (61.6%)
> - BalCor ≥ 3.5: **3308** features (53.3%)
> - BalCor ≥ 4.0: **2691** features (43.3%)
> - BalCor ≥ 5.0: **1994** features (32.1%)
> - BalCor ≥ 6.0: **754** features (12.1%)
> - BalCor ≥ 7.0: **101** features (1.6%)

## 3. SummarySpanScore Distribution

| SummarySpanScore | Count | % |
|:---:|:---:|:---:|
| 2 | 2270 | 36.5% |
| 4 | 1276 | 20.5% |
| 6 | 1786 | 28.8% |
| 8 | 879 | 14.2% |

## 4. SpanInternalScore Distribution

| SpanInternalScore | Count | % |
|:---:|:---:|:---:|
| 1 | 1237 | 19.9% |
| 2 | 1256 | 20.2% |
| 3 | 1372 | 22.1% |
| 4 | 2168 | 34.9% |
| 6 | 125 | 2.0% |
| 8 | 53 | 0.9% |

## 5. AnchorTaskDecision Distribution

| Decision | Count | % |
|:---|:---:|:---:|
| yes | 6211 | 100.0% |

## 6. AnchorVerifyDecision Distribution

| Decision | Count | % |
|:---|:---:|:---:|
| yes | 4486 | 72.2% |
| probably | 1725 | 27.8% |

## 7. DominantCategory Distribution

| Category | Count | % |
|:---|:---:|:---:|
| none | 2609 | 42.0% |
| fraud_theft_or_property_crime | 1062 | 17.1% |
| privacy_doxxing_or_tracking | 544 | 8.8% |
| lethal_or_physical_violence | 432 | 7.0% |
| hate_or_dehumanization | 319 | 5.1% |
| drugs_or_illicit_substances | 296 | 4.8% |
| general_illegal_or_unsafe | 250 | 4.0% |
| cyber_abuse | 205 | 3.3% |
| harassment_or_defamation | 185 | 3.0% |
| weapons_or_explosives | 184 | 3.0% |
| sexual_minors_or_exploitation | 80 | 1.3% |
| self_harm | 45 | 0.7% |

## 8. WeakSummaryLanguage Distribution

| Value | Count | % |
|:---|:---:|:---:|
| no | 3246 | 52.3% |
| yes | 2965 | 47.7% |

## 9. StrongSummaryLanguage Distribution

| Value | Count | % |
|:---|:---:|:---:|
| yes | 3941 | 63.5% |
| no | 2270 | 36.5% |

## 10. SpanCount & ToxicSpanCount Stats

| Metric | SpanCount | ToxicSpanCount |
|:---|:---:|:---:|
| Min | 1 | 0 |
| Max | 10 | 10 |
| Avg | 5.3 | 1.0 |

---

## 🔍 Combined Filtering Matrix (Tox × BalCor)

Features surviving **both** thresholds:

| Tox \ BalCor | ≥2.0 | ≥2.5 | ≥3.0 | ≥3.5 | ≥4.0 | ≥5.0 | ≥6.0 | ≥7.0 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **≥5** | 4878 | 4483 | 3824 | 3308 | 2691 | 1994 | 754 | 101 |
| **≥6** | 4839 | 4452 | 3793 | 3277 | 2660 | 1963 | 754 | 101 |
| **≥7** | 3743 | 3457 | 3214 | 2712 | 2483 | 1889 | 722 | 100 |
| **≥8** | 2738 | 2606 | 2499 | 2085 | 1986 | 1696 | 635 | 88 |
| **≥9** | 1435 | 1416 | 1360 | 1111 | 1068 | 1007 | 347 | 53 |
| **≥10** | 530 | 528 | 512 | 422 | 398 | 391 | 116 | 18 |

## 🔍 Combined Filtering Matrix (Tox × SummarySpanScore)

| Tox \ SummSpan | ≥2 | ≥4 | ≥6 | ≥8 |
|:---:|:---:|:---:|:---:|:---:|
| **≥5** | 4878 | 3941 | 2665 | 879 |
| **≥6** | 4839 | 3910 | 2634 | 848 |
| **≥7** | 3743 | 2929 | 2457 | 774 |
| **≥8** | 2738 | 2168 | 1962 | 640 |
| **≥9** | 1435 | 1152 | 1053 | 323 |
| **≥10** | 530 | 433 | 393 | 107 |

## 11. CorrelationReason Distribution

| Reason | Count | % |
|:---|:---:|:---:|
| weak or noisy correlation between summary and spans | 2387 | 38.4% |
| summary mostly matches spans with moderate span coherence | 1893 | 30.5% |
| partial summary-span or span-span correlation; usable but mixed | 1830 | 29.5% |
| summary matches spans and spans are strongly coherent | 101 | 1.6% |

## 12. ToxicityReason Distribution

| Reason | Count | % |
|:---|:---:|:---:|
| clear harmful category, but not maximally severe/repeated | 2308 | 37.2% |
| high-severity harmful category with clear toxic evidence | 1435 | 23.1% |
| little clear toxic evidence in the provided spans | 1333 | 21.5% |
| some potentially harmful evidence, but weak or mixed | 1135 | 18.3% |
