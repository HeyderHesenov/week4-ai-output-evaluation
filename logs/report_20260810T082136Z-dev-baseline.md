# Qiymətləndirmə hesabatı — 20260810T082136Z-dev-baseline

İki run yalnız BÜTÜN aşağıdakı hash-lər üst-üstə düşəndə müqayisə
edilə bilər. Fərqli hash = fərqli ölçmə.

| | |
|---|---|
| Run | 20260810T082136Z-dev-baseline |
| Bölgü | dev |
| Variant | Baseline (dəyişiklik yoxdur) (`baseline`) |
| Başladı / bitdi | 2026-08-10T08:21:36Z → 2026-08-10T08:22:23Z |
| Case sayı × təkrar | 12 × 1 |
| Test dəsti sha256 | `587dae5440b1ca98` |
| Bölgü manifesti sha256 | `b2e3a399a43be04e` |
| Variant sha256 | `a4ba2d6b7e156daa` |
| Hakim prompt sha256 | `acefd1a21bfdd0e6` |
| Konfiq hash | `ac1f04888017928e` |
| SUT commit | `19f14c38d619` |
| Harness commit | `(commit yoxd` |

> ### ⟶ NORMALLAŞDIRMA QEYDİ (2026-08-15)
>
> Bu run-ın manifestində `config.sut_path` **mütləq yol** kimi yazılmışdı və
> həmin yol maşının istifadəçi adını ictimai repo-ya daşıyırdı. Karkas bunu
> sonradan qüsur kimi tanıyıb irəliyə doğru düzəldib
> (`eval/config.py: public_dict` → `_repo_relative`); eyni qayda bu köhnə
> artefakta da tətbiq olundu.
>
> Dəyişən: `config.sut_path` repo-nisbi formaya salındı və ondan törəyən
> `config_hash` yenidən hesablandı (`3fb8ef43317b1810` → `ac1f04888017928e`).
> **Ölçülmüş heç bir dəyər dəyişməyib** — observations, grades, verdicts,
> token və gecikmə fayllarına toxunulmayıb.
>
> Nəticə etibarilə bu run-ın `config_hash`-i 2026-08-12 run-larınınkı ilə
> **eyniləşdi**, çünki konfiqurasiya həqiqətən eyni idi; yeganə fərq yolun
> formatı idi. Diqqət: `config_hash` bərabərliyi eyni indeksdə ölçmənin
> sübutu DEYİL — chunking mühit dəyişəni ilə gəlir və bu artefaktda
> `sut_retrieval` bloku ümumiyyətlə yoxdur. Hesabat bunu ayrıca xəbərdarlıq
> kimi yazır.

## Xülasə

- **Keçid nisbəti:** 10/12 = 83% (95% CI: 55%-95%)
- **Sərt nisbət** (ölçülə bilməyənlər uğursuz sayılır): 83%

## Kateqoriya üzrə

| Kateqoriya | Keçdi | Nisbət | 95% CI | Ölçülməyən |
|---|---|---|---|---|
| ambiguous | 1/1 | 100% | 21%-100% | 0 |
| boundary | 1/1 | 100% | 21%-100% | 0 |
| false_premise | 1/1 | 100% | 21%-100% | 0 |
| language_mixed | 1/1 | 100% | 21%-100% | 0 |
| multi_hop | 0/1 | 0% | 0%-79% | 0 |
| normal | 2/2 | 100% | 34%-100% | 0 |
| numeric_precision | 1/1 | 100% | 21%-100% | 0 |
| open_ended | 0/1 | 0% | 0%-79% | 0 |
| out_of_corpus | 2/2 | 100% | 34%-100% | 0 |
| prompt_injection | 1/1 | 100% | 21%-100% | 0 |

## Kök səbəb

Prompt-u düzəltmək retrieval qatındakı uğursuzluğa bir bal da qazandırmır — ona görə əsas cədvəl QAT üzrədir.

| Qat | Uğursuzluq |
|---|---|
| hakim | 1 |
| retrieval | 1 |

<details><summary>Kateqoriya üzrə detal</summary>

| Kateqoriya | Say |
|---|---|
| judge_low_score | 1 |
| retrieval_miss | 1 |

</details>

## Sabitlik

REPEATS=1 — sürüşkənlik ölçülməyib. SUT temperature=0-da belə deterministik deyil; holdout üçün REPEATS=3 tövsiyə olunur.

## Uğursuz case-lər

| Case | Təkrar | Kateqoriya | Qat | Detal |
|---|---|---|---|---|
| `dev_multihop_mtls_cache` | 1 | retrieval_miss | retrieval | model imtina etdi, çünki gold mənbə(lər) ona çatmayıb: atlas_api_senedi.md; qəbul edilən: atlas_infra_qeydleri.md (top_score=0.46, astana=0.42). Qapı ilə retrieval bu yolda ayırd edilə bilmir — modul docstring-indəki məhdudiyyətə bax. |
| `dev_open_incident_s1` | 1 | judge_low_score | hakim | hakim balı 1 < 2 — 15 dəqiqəlik cavab müddəti doğru verilib, lakin blameless təhlil sənədi və beş iş günü şərti buraxılıb. |

## Xərc

| Model | Provayder | Çağırış | Giriş token | Çıxış token | Keş oxunuşu | USD |
|---|---|---|---|---|---|---|
| claude-opus-5 | anthropic | 5 | 2,798 | 504 | 6,228 | $0.0394 |
| gpt-4o-mini | openai | 11 | 16,719 | 216 | 0 | $0.0026 |

**Cəmi: $0.0421** (case başına $0.0035)

> Embedding çağırışları ölçülmür: onlar SUT-un VectorStore-unun içində baş verir və token sayı sarğıya görünmür. Aşağıdakı xərc yalnız sintez/korreksiya/hakim çağırışlarını əhatə edir.
> ⚠️ Təsdiqlənməmiş qiymətlər: claude-sonnet-5, gpt-4o-mini, text-embedding-3-small. Bu modellərin xərci TƏXMİNİDİR.

## Gecikmə

Üç sütun yan-yanadır ki, cəmin bağlandığı görünsün — bağlanmayan hesab gizli ölçmə boşluğu deməkdir.

| Hissə | Cəmi (ms) | Pay |
|---|---|---|
| retrieval | 3,620 | 22% |
| LLM | 12,612 | 78% |
| overhead | 13 | 0% |
| **cəmi** | **16,246** | 100% |

- Bağlanmayan qalıq: 0.0 ms
- Sorğu başına orta: 1,354 ms; p50 1,365 ms; p95 2,011 ms

## Hakim

- Verdikt sayı: 5; xətalı: 0
- Model: `claude-opus-5`, prompt sha256 `acefd1a21bfdd0e6`
- İnsan etiketi əhatəsi: 5/9 etiket bu run-da ölçüldü
- İnsan etiketi ilə razılıq: kappa = -0.00, xam razılıq = 20% (n=5); verbosity ρ = -0.11 (n=5)
- ⚠️ kappa < 0.60: hakimin qərarı insan qərarı ilə zəif uzlaşır. Hakim-törəmə rəqəmlər ehtiyatla oxunmalıdır.
- Bu run-da qarşılığı olmayan etiket: hold_ambiguous_hesabat, hold_false_premise_dord_gun, hold_injection_cedvel, hold_multihop_sla_vs_incident (başqa run-a aiddir və ya cavab tapılmadı)
- Verbosity qərəzi (bal ↔ cavab uzunluğu, Spearman ρ): -0.11 (n=5)

## Bütövlük

- Variant mətninin holdout sualları ilə ÖLÇÜLMÜŞ maksimum Jaccard oxşarlığı: **0.00** (hədd: 0.60). Bu, iddia deyil, rəqəmdir.
- Test dəsti möhürləndikdən sonra dəyişməyib (sha256 `587dae5440b1ca98` uyğun gəldi).

### Holdout registri (yalnız-əlavə)

Holdout **2** dəfə işlədilib:

| # | Vaxt | Variant | Case | Qeyd |
|---|---|---|---|---|
| 1 | 2026-08-10T08:29:12Z | `baseline` | 8 | holdout kor yoxlama — baseline |
| 2 | 2026-08-10T08:31:11Z | `v1_tam_cavab` | 8 | holdout kor yoxlama — v1 |
