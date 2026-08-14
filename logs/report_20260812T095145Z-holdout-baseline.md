# Qiymətləndirmə hesabatı — 20260812T095145Z-holdout-baseline

İki run yalnız BÜTÜN aşağıdakı hash-lər üst-üstə düşəndə müqayisə
edilə bilər. Fərqli hash = fərqli ölçmə.

| | |
|---|---|
| Run | 20260812T095145Z-holdout-baseline |
| Bölgü | holdout |
| Variant | Baseline (dəyişiklik yoxdur) (`baseline`) |
| Başladı / bitdi | 2026-08-12T09:51:45Z → 2026-08-12T09:53:30Z |
| Case sayı × təkrar | 8 × 3 |
| Test dəsti sha256 | `587dae5440b1ca98` |
| Bölgü manifesti sha256 | `b2e3a399a43be04e` |
| Variant sha256 | `a4ba2d6b7e156daa` |
| Hakim prompt sha256 | `acefd1a21bfdd0e6` |
| Konfiq hash | `34f9f02c16052e53` |
| SUT commit | `19f14c38d619` |
| Harness commit | `d732254feb91` |

## Xülasə

- **Keçid nisbəti:** 18/24 = 75% (95% CI: 55%-88%)
- **Sərt nisbət** (ölçülə bilməyənlər uğursuz sayılır): 75%

## Kateqoriya üzrə

| Kateqoriya | Keçdi | Nisbət | 95% CI | Ölçülməyən |
|---|---|---|---|---|
| ambiguous | 0/3 | 0% | 0%-56% | 0 |
| false_premise | 0/3 | 0% | 0%-56% | 0 |
| language_mixed | 3/3 | 100% | 44%-100% | 0 |
| multi_hop | 3/3 | 100% | 44%-100% | 0 |
| normal | 3/3 | 100% | 44%-100% | 0 |
| numeric_precision | 3/3 | 100% | 44%-100% | 0 |
| out_of_corpus | 3/3 | 100% | 44%-100% | 0 |
| prompt_injection | 3/3 | 100% | 44%-100% | 0 |

## Kök səbəb

Prompt-u düzəltmək retrieval qatındakı uğursuzluğa bir bal da qazandırmır — ona görə əsas cədvəl QAT üzrədir.

| Qat | Uğursuzluq |
|---|---|
| generasiya | 6 |

<details><summary>Kateqoriya üzrə detal</summary>

| Kateqoriya | Say |
|---|---|
| generation_wrong | 3 |
| over_refusal | 3 |

</details>

## Sabitlik

- Təkrar sayı: 3
- BÜTÜN təkrarlarda keçən case: 6/8
- Sürüşən case yoxdur.

## Uğursuz case-lər

| Case | Təkrar | Kateqoriya | Qat | Detal |
|---|---|---|---|---|
| `hold_false_premise_dord_gun` | 1 | generation_wrong | generasiya | numeric: tapılmadı: 3 gün |
| `hold_false_premise_dord_gun` | 2 | generation_wrong | generasiya | numeric: tapılmadı: 3 gün |
| `hold_false_premise_dord_gun` | 3 | generation_wrong | generasiya | numeric: tapılmadı: 3 gün |
| `hold_ambiguous_hesabat` | 1 | over_refusal | generasiya | gold mənbə modelə çatdığı halda model kontekstdə cavab tapmadı |
| `hold_ambiguous_hesabat` | 2 | over_refusal | generasiya | gold mənbə modelə çatdığı halda model kontekstdə cavab tapmadı |
| `hold_ambiguous_hesabat` | 3 | over_refusal | generasiya | gold mənbə modelə çatdığı halda model kontekstdə cavab tapmadı |

## Before / after

- Baseline (`baseline`): 15/24 = 62% (95% CI: 43%-79%)
- Variant (`baseline`): 18/24 = 75% (95% CI: 55%-88%)
- ℹ️ Konfiqurasiyada yalnız təmsili fərq (`sut_path`): dəyər başqa formatda yazılıb. `config_hash` buna görə fərqlidir.
- ⟶ **BU SƏTİR KÖHNƏLDİ (2026-08-15):** müqayisə edilən 2026-08-10 artefaktında
  `config.sut_path` repo-nisbi formaya salındı (maşın yolu istifadəçi adını
  daşıyırdı) və `config_hash` yenidən hesablandı. İki run-ın `config_hash`-i
  ARTIQ EYNİDİR. Sətir silinmir, çünki hesabatın həmin tarixdəki vəziyyətini
  qeyd edir. Aşağıdakı `sut_retrieval` xəbərdarlığı isə QÜVVƏDƏDİR və əsas
  olan odur: chunking mühit dəyişəni ilə gəlir, `config_hash`-a düşmür.
- ⚠️ Run-lardan birində `sut_retrieval` qeydi yoxdur (artefakt bu sahə əlavə olunmazdan əvvəl yazılıb). **İki run-ın eyni indeksdə və eyni retrieval parametrləri ilə ölçüldüyü artefaktdan TƏSDİQLƏNƏ BİLMİR.**

Cütlənmiş McNemar testi (eyni case-lər, dəqiq binom):

- +1 / -0 (dəyişməyən: 7), p = 1.000 — əhəmiyyətli DEYİL
- ⚠️ Fərq statistik əhəmiyyətli DEYİL: bu nümunə ölçüsündə yaxşılaşmanı təsadüfdən ayırmaq mümkün olmadı. «Prompt işlədi» nəticəsi çıxarmaq üçün dəlil kifayət etmir.

## Xərc

| Model | Provayder | Çağırış | Giriş token | Çıxış token | Keş oxunuşu | USD |
|---|---|---|---|---|---|---|
| claude-opus-5 | anthropic | 12 | 7,867 | 1,346 | 17,127 | $0.0913 |
| gpt-4o-mini | openai | 24 | 32,328 | 705 | 0 | $0.0053 |

**Cəmi: $0.0966** (case başına $0.0121)

> Embedding çağırışları ölçülmür: onlar SUT-un VectorStore-unun içində baş verir və token sayı sarğıya görünmür. Aşağıdakı xərc yalnız sintez/korreksiya/hakim çağırışlarını əhatə edir.
> ⚠️ Təsdiqlənməmiş qiymətlər: claude-sonnet-5, gpt-4o-mini, text-embedding-3-small. Bu modellərin xərci TƏXMİNİDİR.

## Gecikmə

Üç sütun yan-yanadır ki, cəmin bağlandığı görünsün — bağlanmayan hesab gizli ölçmə boşluğu deməkdir.

| Hissə | Cəmi (ms) | Pay |
|---|---|---|
| retrieval | 6,252 | 17% |
| LLM | 31,080 | 83% |
| overhead | 106 | 0% |
| **cəmi** | **37,437** | 100% |

- Bağlanmayan qalıq: 0.0 ms
- Sorğu başına orta: 1,560 ms; p50 1,158 ms; p95 4,067 ms

## Hakim

- Verdikt sayı: 12; xətalı: 0
- Model: `claude-opus-5`, prompt sha256 `acefd1a21bfdd0e6`
- İnsan etiketi əhatəsi: 0/9 etiket bu run-da ölçüldü
- Bu run-a aid etiket yoxdur — kappa hesablanmadı. Etiketlər başqa run-ın cavablarına bağlıdır.
- Bu run-da qarşılığı olmayan etiket: dev_ambiguous_limit, dev_false_premise_free_cache, dev_lang_mixed_backup, dev_multihop_mtls_cache, dev_open_incident_s1, hold_ambiguous_hesabat, hold_false_premise_dord_gun, hold_injection_cedvel, hold_multihop_sla_vs_incident (başqa run-a aiddir və ya cavab tapılmadı)
- Verbosity qərəzi (bal ↔ cavab uzunluğu, Spearman ρ): +0.83 (n=12)
- ⚠️ Güclü müsbət korrelyasiya: hakim uzun cavaba yüksək bal verməyə meyllidir, yəni bal qismən sözçülüyü ölçür.

## Bütövlük

- Variant mətninin holdout sualları ilə ÖLÇÜLMÜŞ maksimum Jaccard oxşarlığı: **0.00** (hədd: 0.60). Bu, iddia deyil, rəqəmdir.
- Test dəsti möhürləndikdən sonra dəyişməyib (sha256 `587dae5440b1ca98` uyğun gəldi).

### Holdout registri (yalnız-əlavə)

Holdout **3** dəfə işlədilib:

| # | Vaxt | Variant | Case | Qeyd |
|---|---|---|---|---|
| 1 | 2026-08-10T08:29:12Z | `baseline` | 8 | holdout kor yoxlama — baseline |
| 2 | 2026-08-10T08:31:11Z | `v1_tam_cavab` | 8 | holdout kor yoxlama — v1 |
| 3 | 2026-08-12T09:51:45Z | `baseline` | 8 | chunking 500/150 — öncədən qeyd edilmiş proqnozun holdout təsdiqi |

> ⚠️ Holdout ikidən çox dəfə işlədilib. Hər əlavə icra onun «kor yoxlama» statusunu bir qədər zəiflədir — yoxlayıcı bunu nəzərə almalıdır.
