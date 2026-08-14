# Qiymətləndirmə hesabatı — 20260810T082743Z-dev-v1_tam_cavab

İki run yalnız BÜTÜN aşağıdakı hash-lər üst-üstə düşəndə müqayisə
edilə bilər. Fərqli hash = fərqli ölçmə.

| | |
|---|---|
| Run | 20260810T082743Z-dev-v1_tam_cavab |
| Bölgü | dev |
| Variant | v1 — çoxhissəli suallarda tam cavab (`v1_tam_cavab`) |
| Başladı / bitdi | 2026-08-10T08:27:43Z → 2026-08-10T08:28:30Z |
| Case sayı × təkrar | 12 × 1 |
| Test dəsti sha256 | `587dae5440b1ca98` |
| Bölgü manifesti sha256 | `b2e3a399a43be04e` |
| Variant sha256 | `07d51428fea9ee55` |
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

- **Keçid nisbəti:** 11/12 = 92% (95% CI: 65%-99%)
- **Sərt nisbət** (ölçülə bilməyənlər uğursuz sayılır): 92%

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
| open_ended | 1/1 | 100% | 21%-100% | 0 |
| out_of_corpus | 2/2 | 100% | 34%-100% | 0 |
| prompt_injection | 1/1 | 100% | 21%-100% | 0 |

## Kök səbəb

Prompt-u düzəltmək retrieval qatındakı uğursuzluğa bir bal da qazandırmır — ona görə əsas cədvəl QAT üzrədir.

| Qat | Uğursuzluq |
|---|---|
| retrieval | 1 |

<details><summary>Kateqoriya üzrə detal</summary>

| Kateqoriya | Say |
|---|---|
| retrieval_miss | 1 |

</details>

## Sabitlik

REPEATS=1 — sürüşkənlik ölçülməyib. SUT temperature=0-da belə deterministik deyil; holdout üçün REPEATS=3 tövsiyə olunur.

## Uğursuz case-lər

| Case | Təkrar | Kateqoriya | Qat | Detal |
|---|---|---|---|---|
| `dev_multihop_mtls_cache` | 1 | retrieval_miss | retrieval | model imtina etdi, çünki gold mənbə(lər) ona çatmayıb: atlas_api_senedi.md; qəbul edilən: atlas_infra_qeydleri.md (top_score=0.46, astana=0.42). Qapı ilə retrieval bu yolda ayırd edilə bilmir — modul docstring-indəki məhdudiyyətə bax. |

## Before / after

- Baseline (`baseline`): 10/12 = 83% (95% CI: 55%-95%)
- Variant (`v1_tam_cavab`): 11/12 = 92% (95% CI: 65%-99%)

Cütlənmiş McNemar testi (eyni case-lər, dəqiq binom):

- +1 / -0 (dəyişməyən: 11), p = 1.000 — əhəmiyyətli DEYİL
- ⚠️ Fərq statistik əhəmiyyətli DEYİL: bu nümunə ölçüsündə yaxşılaşmanı təsadüfdən ayırmaq mümkün olmadı. «Prompt işlədi» nəticəsi çıxarmaq üçün dəlil kifayət etmir.

## Xərc

| Model | Provayder | Çağırış | Giriş token | Çıxış token | Keş oxunuşu | USD |
|---|---|---|---|---|---|---|
| claude-opus-5 | anthropic | 5 | 2,849 | 551 | 6,228 | $0.0409 |
| gpt-4o-mini | openai | 12 | 24,134 | 450 | 12,544 | $0.0058 |

**Cəmi: $0.0466** (case başına $0.0039)

> Embedding çağırışları ölçülmür: onlar SUT-un VectorStore-unun içində baş verir və token sayı sarğıya görünmür. Aşağıdakı xərc yalnız sintez/korreksiya/hakim çağırışlarını əhatə edir.
> ⚠️ Təsdiqlənməmiş qiymətlər: claude-sonnet-5, gpt-4o-mini, text-embedding-3-small. Bu modellərin xərci TƏXMİNİDİR.

## Gecikmə

Üç sütun yan-yanadır ki, cəmin bağlandığı görünsün — bağlanmayan hesab gizli ölçmə boşluğu deməkdir.

| Hissə | Cəmi (ms) | Pay |
|---|---|---|
| retrieval | 3,201 | 16% |
| LLM | 16,691 | 84% |
| overhead | 16 | 0% |
| **cəmi** | **19,908** | 100% |

- Bağlanmayan qalıq: 0.0 ms
- Sorğu başına orta: 1,659 ms; p50 1,214 ms; p95 3,971 ms

## Hakim

- Verdikt sayı: 5; xətalı: 0
- Model: `claude-opus-5`, prompt sha256 `acefd1a21bfdd0e6`
- İnsan etiketi əhatəsi: 0/9 etiket bu run-da ölçüldü
- Bu run-a aid etiket yoxdur — kappa hesablanmadı. Etiketlər başqa run-ın cavablarına bağlıdır.
- Bu run-da qarşılığı olmayan etiket: dev_ambiguous_limit, dev_false_premise_free_cache, dev_lang_mixed_backup, dev_multihop_mtls_cache, dev_open_incident_s1, hold_ambiguous_hesabat, hold_false_premise_dord_gun, hold_injection_cedvel, hold_multihop_sla_vs_incident (başqa run-a aiddir və ya cavab tapılmadı)
- Verbosity qərəzi (bal ↔ cavab uzunluğu, Spearman ρ): +0.00 (n=5)

## Bütövlük

- Variant mətninin holdout sualları ilə ÖLÇÜLMÜŞ maksimum Jaccard oxşarlığı: **0.11** (hədd: 0.60). Bu, iddia deyil, rəqəmdir.
- Test dəsti möhürləndikdən sonra dəyişməyib (sha256 `587dae5440b1ca98` uyğun gəldi).

### Holdout registri (yalnız-əlavə)

Holdout **2** dəfə işlədilib:

| # | Vaxt | Variant | Case | Qeyd |
|---|---|---|---|---|
| 1 | 2026-08-10T08:29:12Z | `baseline` | 8 | holdout kor yoxlama — baseline |
| 2 | 2026-08-10T08:31:11Z | `v1_tam_cavab` | 8 | holdout kor yoxlama — v1 |
