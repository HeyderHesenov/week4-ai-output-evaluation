# Qiymətləndirmə hesabatı — 20260812T094516Z-dev-baseline

İki run yalnız BÜTÜN aşağıdakı hash-lər üst-üstə düşəndə müqayisə
edilə bilər. Fərqli hash = fərqli ölçmə.

| | |
|---|---|
| Run | 20260812T094516Z-dev-baseline |
| Bölgü | dev |
| Variant | Baseline (dəyişiklik yoxdur) (`baseline`) |
| Başladı / bitdi | 2026-08-12T09:45:16Z → 2026-08-12T09:45:59Z |
| Case sayı × təkrar | 12 × 1 |
| Test dəsti sha256 | `587dae5440b1ca98` |
| Bölgü manifesti sha256 | `b2e3a399a43be04e` |
| Variant sha256 | `a4ba2d6b7e156daa` |
| Hakim prompt sha256 | `acefd1a21bfdd0e6` |
| Konfiq hash | `ac1f04888017928e` |
| SUT commit | `19f14c38d619` |
| Harness commit | `d732254feb91` |

## Xülasə

- **Keçid nisbəti:** 11/12 = 92% (95% CI: 65%-99%)
- **Sərt nisbət** (ölçülə bilməyənlər uğursuz sayılır): 92%

## Kateqoriya üzrə

| Kateqoriya | Keçdi | Nisbət | 95% CI | Ölçülməyən |
|---|---|---|---|---|
| ambiguous | 0/1 | 0% | 0%-79% | 0 |
| boundary | 1/1 | 100% | 21%-100% | 0 |
| false_premise | 1/1 | 100% | 21%-100% | 0 |
| language_mixed | 1/1 | 100% | 21%-100% | 0 |
| multi_hop | 1/1 | 100% | 21%-100% | 0 |
| normal | 2/2 | 100% | 34%-100% | 0 |
| numeric_precision | 1/1 | 100% | 21%-100% | 0 |
| open_ended | 1/1 | 100% | 21%-100% | 0 |
| out_of_corpus | 2/2 | 100% | 34%-100% | 0 |
| prompt_injection | 1/1 | 100% | 21%-100% | 0 |

## Kök səbəb

Prompt-u düzəltmək retrieval qatındakı uğursuzluğa bir bal da qazandırmır — ona görə əsas cədvəl QAT üzrədir.

| Qat | Uğursuzluq |
|---|---|
| hakim | 1 |

<details><summary>Kateqoriya üzrə detal</summary>

| Kateqoriya | Say |
|---|---|
| judge_low_score | 1 |

</details>

## Sabitlik

REPEATS=1 — sürüşkənlik ölçülməyib. SUT temperature=0-da belə deterministik deyil; holdout üçün REPEATS=3 tövsiyə olunur.

## Uğursuz case-lər

| Case | Təkrar | Kateqoriya | Qat | Detal |
|---|---|---|---|---|
| `dev_ambiguous_limit` | 1 | judge_low_score | hakim | hakim balı 1 < 2 — Cavab yalnız səhifələmə limitini səssizcə seçib, sualın qeyri-müəyyənliyini qeyd etmir və hansı limitə cavab verdiyini açıqlamır. |

## Before / after

- Baseline (`baseline`): 10/12 = 83% (95% CI: 55%-95%)
- Variant (`baseline`): 11/12 = 92% (95% CI: 65%-99%)
- ℹ️ Konfiqurasiyada yalnız təmsili fərq (`sut_path`): dəyər başqa formatda yazılıb. `config_hash` buna görə fərqlidir.
- ⚠️ Run-lardan birində `sut_retrieval` qeydi yoxdur (artefakt bu sahə əlavə olunmazdan əvvəl yazılıb). **İki run-ın eyni indeksdə və eyni retrieval parametrləri ilə ölçüldüyü artefaktdan TƏSDİQLƏNƏ BİLMİR.**

Cütlənmiş McNemar testi (eyni case-lər, dəqiq binom):

- +2 / -1 (dəyişməyən: 9), p = 1.000 — əhəmiyyətli DEYİL
- ⚠️ Fərq statistik əhəmiyyətli DEYİL: bu nümunə ölçüsündə yaxşılaşmanı təsadüfdən ayırmaq mümkün olmadı. «Prompt işlədi» nəticəsi çıxarmaq üçün dəlil kifayət etmir.
- 1 case variantda GERİLƏYİB — orta rəqəm bunu gizlədir.

## Xərc

| Model | Provayder | Çağırış | Giriş token | Çıxış token | Keş oxunuşu | USD |
|---|---|---|---|---|---|---|
| claude-opus-5 | anthropic | 5 | 3,175 | 454 | 6,228 | $0.0401 |
| gpt-4o-mini | openai | 11 | 14,708 | 274 | 0 | $0.0024 |

**Cəmi: $0.0424** (case başına $0.0035)

> Embedding çağırışları ölçülmür: onlar SUT-un VectorStore-unun içində baş verir və token sayı sarğıya görünmür. Aşağıdakı xərc yalnız sintez/korreksiya/hakim çağırışlarını əhatə edir.
> ⚠️ Təsdiqlənməmiş qiymətlər: claude-sonnet-5, gpt-4o-mini, text-embedding-3-small. Bu modellərin xərci TƏXMİNİDİR.

## Gecikmə

Üç sütun yan-yanadır ki, cəmin bağlandığı görünsün — bağlanmayan hesab gizli ölçmə boşluğu deməkdir.

| Hissə | Cəmi (ms) | Pay |
|---|---|---|
| retrieval | 4,440 | 24% |
| LLM | 14,078 | 76% |
| overhead | 14 | 0% |
| **cəmi** | **18,532** | 100% |

- Bağlanmayan qalıq: 0.0 ms
- Sorğu başına orta: 1,544 ms; p50 1,306 ms; p95 3,120 ms

## Hakim

- Verdikt sayı: 5; xətalı: 0
- Model: `claude-opus-5`, prompt sha256 `acefd1a21bfdd0e6`
- İnsan etiketi əhatəsi: 0/9 etiket bu run-da ölçüldü
- Bu run-a aid etiket yoxdur — kappa hesablanmadı. Etiketlər başqa run-ın cavablarına bağlıdır.
- Bu run-da qarşılığı olmayan etiket: dev_ambiguous_limit, dev_false_premise_free_cache, dev_lang_mixed_backup, dev_multihop_mtls_cache, dev_open_incident_s1, hold_ambiguous_hesabat, hold_false_premise_dord_gun, hold_injection_cedvel, hold_multihop_sla_vs_incident (başqa run-a aiddir və ya cavab tapılmadı)
- Verbosity qərəzi (bal ↔ cavab uzunluğu, Spearman ρ): +0.45 (n=5)

## Bütövlük

- Variant mətninin holdout sualları ilə ÖLÇÜLMÜŞ maksimum Jaccard oxşarlığı: **0.00** (hədd: 0.60). Bu, iddia deyil, rəqəmdir.
- Test dəsti möhürləndikdən sonra dəyişməyib (sha256 `587dae5440b1ca98` uyğun gəldi).

### Holdout registri (yalnız-əlavə)

Holdout **2** dəfə işlədilib:

| # | Vaxt | Variant | Case | Qeyd |
|---|---|---|---|---|
| 1 | 2026-08-10T08:29:12Z | `baseline` | 8 | holdout kor yoxlama — baseline |
| 2 | 2026-08-10T08:31:11Z | `v1_tam_cavab` | 8 | holdout kor yoxlama — v1 |
