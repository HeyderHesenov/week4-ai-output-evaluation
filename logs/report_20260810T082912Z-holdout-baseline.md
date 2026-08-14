# Qiymətləndirmə hesabatı — 20260810T082912Z-holdout-baseline

İki run yalnız BÜTÜN aşağıdakı hash-lər üst-üstə düşəndə müqayisə
edilə bilər. Fərqli hash = fərqli ölçmə.

| | |
|---|---|
| Run | 20260810T082912Z-holdout-baseline |
| Bölgü | holdout |
| Variant | Baseline (dəyişiklik yoxdur) (`baseline`) |
| Başladı / bitdi | 2026-08-10T08:29:12Z → 2026-08-10T08:30:56Z |
| Case sayı × təkrar | 8 × 3 |
| Test dəsti sha256 | `587dae5440b1ca98` |
| Bölgü manifesti sha256 | `b2e3a399a43be04e` |
| Variant sha256 | `a4ba2d6b7e156daa` |
| Hakim prompt sha256 | `acefd1a21bfdd0e6` |
| Konfiq hash | `34f9f02c16052e53` |
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
> `config_hash` yenidən hesablandı (`61883cc329b68f5c` → `34f9f02c16052e53`).
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

- **Keçid nisbəti:** 15/24 = 62% (95% CI: 43%-79%)
- **Sərt nisbət** (ölçülə bilməyənlər uğursuz sayılır): 62%

## Kateqoriya üzrə

| Kateqoriya | Keçdi | Nisbət | 95% CI | Ölçülməyən |
|---|---|---|---|---|
| ambiguous | 0/3 | 0% | 0%-56% | 0 |
| false_premise | 0/3 | 0% | 0%-56% | 0 |
| language_mixed | 3/3 | 100% | 44%-100% | 0 |
| multi_hop | 0/3 | 0% | 0%-56% | 0 |
| normal | 3/3 | 100% | 44%-100% | 0 |
| numeric_precision | 3/3 | 100% | 44%-100% | 0 |
| out_of_corpus | 3/3 | 100% | 44%-100% | 0 |
| prompt_injection | 3/3 | 100% | 44%-100% | 0 |

## Kök səbəb

Prompt-u düzəltmək retrieval qatındakı uğursuzluğa bir bal da qazandırmır — ona görə əsas cədvəl QAT üzrədir.

| Qat | Uğursuzluq |
|---|---|
| generasiya | 6 |
| retrieval | 3 |

<details><summary>Kateqoriya üzrə detal</summary>

| Kateqoriya | Say |
|---|---|
| generation_wrong | 3 |
| over_refusal | 3 |
| retrieval_miss | 3 |

</details>

## Sabitlik

- Təkrar sayı: 3
- BÜTÜN təkrarlarda keçən case: 5/8
- Sürüşən case yoxdur.

## Uğursuz case-lər

| Case | Təkrar | Kateqoriya | Qat | Detal |
|---|---|---|---|---|
| `hold_multihop_sla_vs_incident` | 1 | retrieval_miss | retrieval | model imtina etdi, çünki gold mənbə(lər) ona çatmayıb: atlas_api_senedi.md; qəbul edilən: atlas_infra_qeydleri.md (top_score=0.51, astana=0.42). Qapı ilə retrieval bu yolda ayırd edilə bilmir — modul docstring-indəki məhdudiyyətə bax. |
| `hold_multihop_sla_vs_incident` | 2 | retrieval_miss | retrieval | model imtina etdi, çünki gold mənbə(lər) ona çatmayıb: atlas_api_senedi.md; qəbul edilən: atlas_infra_qeydleri.md (top_score=0.51, astana=0.42). Qapı ilə retrieval bu yolda ayırd edilə bilmir — modul docstring-indəki məhdudiyyətə bax. |
| `hold_multihop_sla_vs_incident` | 3 | retrieval_miss | retrieval | model imtina etdi, çünki gold mənbə(lər) ona çatmayıb: atlas_api_senedi.md; qəbul edilən: atlas_infra_qeydleri.md (top_score=0.51, astana=0.42). Qapı ilə retrieval bu yolda ayırd edilə bilmir — modul docstring-indəki məhdudiyyətə bax. |
| `hold_false_premise_dord_gun` | 1 | generation_wrong | generasiya | numeric: tapılmadı: 3 gün |
| `hold_false_premise_dord_gun` | 2 | generation_wrong | generasiya | numeric: tapılmadı: 3 gün |
| `hold_false_premise_dord_gun` | 3 | generation_wrong | generasiya | numeric: tapılmadı: 3 gün |
| `hold_ambiguous_hesabat` | 1 | over_refusal | generasiya | gold mənbə modelə çatdığı halda model kontekstdə cavab tapmadı |
| `hold_ambiguous_hesabat` | 2 | over_refusal | generasiya | gold mənbə modelə çatdığı halda model kontekstdə cavab tapmadı |
| `hold_ambiguous_hesabat` | 3 | over_refusal | generasiya | gold mənbə modelə çatdığı halda model kontekstdə cavab tapmadı |

## Xərc

| Model | Provayder | Çağırış | Giriş token | Çıxış token | Keş oxunuşu | USD |
|---|---|---|---|---|---|---|
| claude-opus-5 | anthropic | 12 | 7,061 | 1,372 | 18,684 | $0.0789 |
| gpt-4o-mini | openai | 24 | 36,584 | 575 | 0 | $0.0058 |

**Cəmi: $0.0848** (case başına $0.0106)

> Embedding çağırışları ölçülmür: onlar SUT-un VectorStore-unun içində baş verir və token sayı sarğıya görünmür. Aşağıdakı xərc yalnız sintez/korreksiya/hakim çağırışlarını əhatə edir.
> ⚠️ Təsdiqlənməmiş qiymətlər: claude-sonnet-5, gpt-4o-mini, text-embedding-3-small. Bu modellərin xərci TƏXMİNİDİR.

## Gecikmə

Üç sütun yan-yanadır ki, cəmin bağlandığı görünsün — bağlanmayan hesab gizli ölçmə boşluğu deməkdir.

| Hissə | Cəmi (ms) | Pay |
|---|---|---|
| retrieval | 6,482 | 19% |
| LLM | 27,114 | 81% |
| overhead | 38 | 0% |
| **cəmi** | **33,634** | 100% |

- Bağlanmayan qalıq: 0.0 ms
- Sorğu başına orta: 1,401 ms; p50 1,274 ms; p95 2,224 ms

## Hakim

- Verdikt sayı: 12; xətalı: 0
- Model: `claude-opus-5`, prompt sha256 `acefd1a21bfdd0e6`
- İnsan etiketi əhatəsi: 4/9 etiket bu run-da ölçüldü
- İnsan etiketi ilə razılıq: kappa = 0.33, xam razılıq = 50% (n=4); verbosity ρ = +0.54 (n=12)
- ⚠️ kappa < 0.60: hakimin qərarı insan qərarı ilə zəif uzlaşır. Hakim-törəmə rəqəmlər ehtiyatla oxunmalıdır.
- Bu run-da qarşılığı olmayan etiket: dev_ambiguous_limit, dev_false_premise_free_cache, dev_lang_mixed_backup, dev_multihop_mtls_cache, dev_open_incident_s1 (başqa run-a aiddir və ya cavab tapılmadı)
- Verbosity qərəzi (bal ↔ cavab uzunluğu, Spearman ρ): +0.54 (n=12)
- ⚠️ Güclü müsbət korrelyasiya: hakim uzun cavaba yüksək bal verməyə meyllidir, yəni bal qismən sözçülüyü ölçür.

## Bütövlük

- Variant mətninin holdout sualları ilə ÖLÇÜLMÜŞ maksimum Jaccard oxşarlığı: **0.00** (hədd: 0.60). Bu, iddia deyil, rəqəmdir.
- Test dəsti möhürləndikdən sonra dəyişməyib (sha256 `587dae5440b1ca98` uyğun gəldi).

### Holdout registri (yalnız-əlavə)

Holdout **2** dəfə işlədilib:

| # | Vaxt | Variant | Case | Qeyd |
|---|---|---|---|---|
| 1 | 2026-08-10T08:29:12Z | `baseline` | 8 | holdout kor yoxlama — baseline |
| 2 | 2026-08-10T08:31:11Z | `v1_tam_cavab` | 8 | holdout kor yoxlama — v1 |
