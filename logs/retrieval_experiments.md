# Chunking / embedding modeli / sorğu genişləndirmə — ÖLÇÜLDÜ (2026-08-12)

`logs/retrieval_sweep.md` qapı **parametrlərini** rədd etdi və üç istiqamət
təklif etdi. Bu sənəd həmin üçünü eyni metrika ilə ölçür. Alət:
`tools/retrieval_experiments.py`, xam nəticə:
`logs/probes/20260812T191757Z-eksperiment-chunking+embedding+genişləndirmə/`.

> ⟶ **RETRAKSİYA (2026-08-13): `logs/retrieval_experiments.json` xam nəticə
> DEYİL.** Bu sətir əvvəllər həmin fayla işarə edirdi. Fayl silinmir (ölçmə
> qeydi geriyə dönük redaktə olunmur), amma o, **13:38 icrasında donub** və
> aşağıdakı cədvəli dəstəkləmir: içindəki eksperiment adları geri götürülmüş
> dəstdəndir (`sorğu genişləndirmə`, `genişləndirmə + 300/90` — büdcə rejimi
> ayrılmamışdan əvvəl) və `retrieval_budget` sahəsi ümumiyyətlə yoxdur, yəni
> D2 düzəlişindən əvvəlkidir. Bu sənədin rəqəmlərinin sübutu yalnız yuxarıda
> göstərilən probe qovluğudur — `logs/retrieval_sweep.md`-dəki eyni
> vəziyyətlə eyni qayda.

## Metod

Qapı BÜTÜN eksperimentlərdə sabit saxlanılır (`0.42 / 0.10 / 0.35`). Səbəb:
burada ölçülən indeksin və ya sorğunun keyfiyyətidir; qapını eyni anda
tərpətsək, hansı dəyişikliyin nə verdiyi bilinməzdi.

**İndeks təcridi.** Hər eksperiment öz `PERSIST_DIR`-inə yazır. Mövcud
`storage/chroma` 2026-08-10 baseline run-larının ölçüldüyü indeksdir; onun
üstünə yazmaq həmin run-ları təkrar istehsal olunmaz edərdi. Alət
`--workdir` paylaşılan qovluğun içində olsa **imtina edir** (`exit 2`).

Korpus 2,574 tokendir, ona görə yenidən indeksləmə 3-small ilə $0.00005,
3-large ilə $0.00034 — eksperimentlərin hamısı praktiki olaraq pulsuzdur.

## Nəticə

| eksperiment | indeks chunk | örtük | sızma | orta qəbul |
|---|---|---|---|---|
| baseline (800/200) | 19 | 7/8 | 1/2 | 2.67 |
| **chunking 500/150** | 29 | **8/8** | **1/2** | 2.92 |
| **chunking 400/120** | 35 | **8/8** | **1/2** | 2.83 |
| chunking 300/90 | 47 | 8/8 | **2/2** ✗ | 3.00 |
| chunking 250/60 | 54 | 8/8 | **2/2** ✗ | 2.92 |
| embedding 3-large | 19 | 7/8 | 1/2 | 2.25 |
| 3-large + 300/90 | 47 | 8/8 | 1/2 | 3.08 |
| **sorğu genişləndirmə** | 19 | **8/8** | **1/2** | 2.83 |
| genişləndirmə + 300/90 | 47 | 8/8 | **2/2** ✗ | 3.33 |

Üç namizəd örtüyü tamamlayır və sızmanı artırmır: **chunking 500/150**,
**chunking 400/120**, **sorğu genişləndirmə**.

## Nə öyrənildi

**Chunking işləyir — amma çox xırdalamaq zərərlidir.** 800 → 500 və 400
örtüyü tamamlayır, sızmanı isə baseline səviyyəsində saxlayır. 300 və 250-də
örtük yenə 8/8-dir, lakin `dev_out_of_corpus_ceo` da chunk qəbul etməyə
başlayır: xırda chunk-lar mövzu kontekstini itirir və təsadüfi sual üçün də
oxşar görünür. Yəni «daha xırda = daha yaxşı» DEYİL, əyri var.

**Embedding modeli tək başına kömək etmir.** `text-embedding-3-large`
baseline chunking ilə yenə 7/8 verir — çatışan sənəd yenə qapıdan keçmir.
Onun tək faydası 300/90-ın yaratdığı sızmanı geri qaytarmasıdır, yəni model
xırda chunk-ların itirdiyi kontekstin bir hissəsini kompensasiya edir. Tokeni
6.5 dəfə baha olan modeli bunun üçün götürmək əsassızdır.

**Sorğu genişləndirmə indeksə toxunmadan işləyir.** Sual bağlayıcılardan
(`və`, `həmçinin`) bölünür, hər hissə üçün retrieval aparılır, nəticə
birləşdirilir; tam sual həmişə saxlanılır ki, bölgü səhv olanda nəticə
pisləşməsin. Mövcud indeksdə 8/8 verir.

## Dürüst məhdudiyyətlər — bunlar nəticəni zəiflədir

1. **n = 1.** Dev-də örtüyü pozan case CƏMİ BİRDİR
   (`dev_multihop_mtls_cache`). Bütün «8/8» nəticələri həmin tək case-in
   çevrilməsidir. Üç fərqli dəyişiklik eyni bir case-i düzəldir və bu sübut
   onları bir-birindən **ayırd edə bilmir**.
2. **Örtük zəruri şərtdir, kifayət deyil.** Sənədin kontekstə düşməsi cavabın
   düzgün olacağı demək deyil — SUT-da ikinci (grounding) qat və sintez
   mərhələsi də var. Uçdan-uca təsdiq üçün tam dev run lazımdır ($0.042).
3. **Sorğu genişləndirmə SUT-a KOD dəyişikliyi tələb edir.** Chunking və
   embedding modeli konfiqurasiyadır (env + yenidən ingest), amma
   çox-sorğulu retrieval Week 2 pipeline-ında yeni kod deməkdir. Burada
   ölçmə xaricdən aparılıb; istehsala köçürmək eyni şey deyil.

## Tövsiyə

**`CHUNK_SIZE=500`, `CHUNK_OVERLAP=150`.**

Səbəbləri: örtüyü tamamlayır; sızmanı artırmır; baseline-a ən yaxın olan
qazanan variantdır (800 → 500, halbuki 400 daha uzaq və eyni nəticəni verir);
təmiz konfiqurasiya dəyişikliyidir — SUT-un bir sətri belə dəyişmir, yalnız
yenidən ingest lazımdır; və qəbul edilən chunk sayını cəmi +0.25 artırır, yəni
kontekst şişmir.

Sorğu genişləndirmə də eyni nəticəni verir və indeksə toxunmur, amma kod
dəyişikliyi tələb edir — eyni qazancı daha ucuz yolla almaq mümkün olduğu üçün
ikinci sıradadır.

## Növbəti addım

Uçdan-uca dev run ($0.042) `CHUNK_SIZE=500` ilə: örtük qazancının həqiqətən
pass-rate-ə çevrildiyini göstərər. Nəticə müsbət olarsa, holdout **bir dəfə**
(ledger 2 → 3).

Manifest artıq indeks kimliyini də yazır (`sut_retrieval`: `chunk_size`,
`chunk_overlap`, `collection_name`, `persist_dir_name`), ona görə fərqli
chunking ilə işlədilən run artefaktdan tanınır.

## Təkrar istehsal

```bash
python tools/retrieval_experiments.py --workdir /tmp/idx-eksperiment
```

---

# Uçdan-uca dev təsdiqi — `CHUNK_SIZE=500` (2026-08-12)

Run: `20260812T094516Z-dev-baseline`, xərc **$0.0424**, indeks `chroma_c500`
(29 chunk, baseline indeksinə toxunulmayıb).

## Nəticə: 10/12 → 11/12, amma statistik əhəmiyyətli DEYİL

McNemar: **+2 / −1** (dəyişməyən 9), **p = 1.000**.

| case | 800/200 | 500/150 | |
|---|---|---|---|
| `dev_multihop_mtls_cache` | `retrieval_miss` | `ok` | ✅ **proqnoz edilən düzəliş** |
| `dev_open_incident_s1` | `judge_low_score` | `ok` | ✅ gözlənilməyən qazanc |
| `dev_ambiguous_limit` | `ok` | `judge_low_score` | ❌ geriləmə |

## Şərh — hansı hissəyə güvənmək olar

**Proqnoz doğru çıxdı və o hissə hakimdən ASILI DEYİL.** Sweep dedi ki,
500/150 ilə `atlas_api_senedi.md` kontekstə düşəcək; uçdan-uca run-da həmin
case `retrieval_miss`-dən `ok`-a keçdi. Bu təsnifat deterministikdir — hakimin
balına söykənmir. Yəni mexanizm təsdiqlənib.

**Qalan iki dəyişiklik isə məhz ölçü alətimizin ən zəif olduğu yerdədir.**
`dev_open_incident_s1` `open_ended`, `dev_ambiguous_limit` isə `ambiguous`
kateqoriyasındadır — hər ikisi hakim balı ilə qiymətləndirilir və
`logs/judge_bias.md` (kappa = 0.13) məhz bu iki kateqoriyanın rəqəmlərini
«ehtiyatla oxuyun» deyə işarələyib. Ona görə +1/−1 fərqini qazanc və ya itki
kimi oxumaq üçün əsas yoxdur; onlar bir-birini ödəyir və hər ikisi zəif
kalibrlənmiş alətlə ölçülüb.

Dürüst yekun: **bir deterministik qazanc təsdiqləndi, qalan hərəkət səs-küydür.**
p = 1.000 bunu onsuz da deyir.

## Artefakt boşluğu — müqayisə kodunda tapılan qüsur

Hesabatın ilk versiyası bu müqayisə üçün «yalnız təmsili fərq, ölçülən sistem
eynidir» yazdı. **Bu yanlış idi:** `CHUNK_SIZE` SUT-a env vasitəsilə gedir,
framework-un `config` blokuna düşmür, ona görə `config_hash` EYNİ qalır.
`_config_diff_lines` yalnız `config`-a baxırdı və indeks fərqini görmürdü.

Düzəldildi: müqayisə indi `sut_retrieval` blokunu da yoxlayır (o, pipeline-dan
GERİ oxunur, yəni SUT-un həqiqətən nə işlətdiyini göstərir). 2026-08-10
baseline-ında bu sahə olmadığı üçün hesabat indi açıq deyir ki, iki run-ın eyni
indeksdə ölçüldüyü **artefaktdan təsdiqlənə bilmir**. Reqressiya testi:
`tests/test_report.py::test_CHUNKING_ferqi_config_hash_eyni_olsa_da_tutulur`.

## Holdout üçün ÖNCƏDƏN QEYD EDİLMİŞ proqnoz

Holdout hələ işlədilməyib (ledger = 2). Aparılarsa, proqnoz **indi** yazılır ki,
sonradan uyğunlaşdırılmasın:

> `hold_multihop_sla_vs_incident` holdout-da `retrieval_miss` idi. Mexanizm
> deterministik olduğu üçün gözlənti onun `ok`-a keçməsidir. Hakim-əsaslı
> kateqoriyalarda (`ambiguous`, `open_ended`) isə istiqamətli proqnoz
> VERİLMİR — dev-də onlar bir-birini ödədi və alət zəif kalibrlənib.

Proqnoz tutmasa, bu, chunking-in işləmədiyinin dəlilidir və elə yazılacaq.

---

# Holdout təsdiqi — proqnoz TUTDU (2026-08-12)

Run: `20260812T095145Z-holdout-baseline`, 8 case × 3 təkrar, xərc **$0.0966**.
**Holdout registrində 3-cü icra.** Bu, birdəfəlik qapının istifadəsidir.

## Nəticə: 15/24 → 18/24 (62% → 75%), McNemar +1 / −0

| case | 800/200 (3 təkrar) | 500/150 (3 təkrar) |
|---|---|---|
| `hold_multihop_sla_vs_incident` | `retrieval_miss` ×3 | **`ok` ×3** ✅ |
| `hold_false_premise_dord_gun` | `generation_wrong` ×3 | `generation_wrong` ×3 |
| `hold_ambiguous_hesabat` | `over_refusal` ×3 | `over_refusal` ×3 |
| qalan 5 case | `ok` ×3 | `ok` ×3 |

**Öncədən qeyd edilmiş proqnoz hərfi-hərfinə tutdu:** yalnız
`hold_multihop_sla_vs_incident` dəyişdi, `retrieval_miss`-dən `ok`-a, və
**hər üç təkrarda**. Hakim-əsaslı kateqoriyalar üçün istiqamətli proqnoz
verilməmişdi — onlar da dəyişmədi. **Sıfır geriləmə.**

## p = 1.000-i necə oxumaq lazımdır

Test yenə «əhəmiyyətli deyil» deyir. Bu, effektin olmaması demək DEYİL — bu
nümunə ölçüsündə testin **konstruksiyaca gücsüz** olması deməkdir:

| uyğunsuz cüt | p (dəqiq binom, iki tərəfli) |
|---|---|
| 1 | 1.000 |
| 3 | 0.250 |
| 5 | 0.0625 |
| **6** | **0.0312** ← ilk dəfə p < 0.05 |

Holdout-da cəmi 8 case var və retrieval qatında uğursuz olan **birdir**.
Yəni p < 0.05 üçün 8 case-in 6-sı çevrilməli idi — bu, retrieval düzəlişindən
prinsipcə gözlənilə bilməzdi. p-dəyəri burada effekt haqqında deyil, nümunə
ölçüsü haqqında məlumat verir.

Dəlil p-də deyil, başqa yerdədir:

1. **Proqnoz uçdan-uca run-dan ƏVVƏL yazılmışdı** (bu sənəddə, yuxarıda) və
   konkret case adı ilə. Sonradan uyğunlaşdırma mümkün deyil.
2. **Hər üç təkrarda eynidir** — sikkə atma deyil.
3. **Təsnifat deterministikdir.** `retrieval_miss` hakimin balından asılı
   deyil; kappa = 0.13 xəbərdarlığı bu nəticəyə toxunmur.
4. **Sıfır geriləmə.** Dev-dəki `dev_ambiguous_limit` geriləməsi holdout-da
   təkrarlanmadı, yəni o, səs-küy idi.

## `v1_tam_cavab` ilə fərq — niyə bu dəfə tövsiyə MÜSBƏTDİR

`logs/before_after.md`-dəki prompt varianti dev-də 10/12 → 11/12 qaldırmış,
holdout-da isə **+0 / −0** vermişdi: qazanc keçmədi, ona görə tövsiyə
«göndərmə» idi.

Bu dəfə qazanc **keçdi** və məhz proqnoz edilən mexanizmlə. Fərq budur.

## Tövsiyə: `CHUNK_SIZE=500`, `CHUNK_OVERLAP=150` GÖNDƏRİLSİN

İstehsalata tətbiq: Week 2-nin `.env`-ində `CHUNK_SIZE=500`,
`CHUNK_OVERLAP=150`, sonra `python -m rag.cli ingest --path data/ --reset`.
SUT-un **bir sətri belə dəyişmir** — bu, təmiz konfiqurasiya dəyişikliyidir.

## Qalan uğursuzluqlar artıq retrieval qatında DEYİL

Holdout-da qalan iki uğursuzluq `generation_wrong` (yanlış müqəddiməni qəbul
edir) və `over_refusal`-dır. Hər ikisi generasiya qatındadır — chunking onlara
təsir etmir və etməməlidir. Növbəti dövr üçün istiqamət budur.

**Holdout registri artıq 3-dədir. Növbəti holdout icrası bu blind-check
iddiasını daha da zəiflədir — növbəti dövr dev-də qurulmalı və holdout yalnız
son təsdiq üçün saxlanılmalıdır.**

---

# ⟶ RETRAKSİYA (2026-08-12): genişləndirmə sətirləri ədalətli müqayisə deyil

Yuxarıdakı mətn silinmir. Geri götürülən **yalnız sorğu genişləndirməsinə**
aid hissədir.

## Nə pozulub

`sub_queries()` parçaları `?`-dən təmizləyir, sonra isə «tam sual artıq
siyahıdadırmı» yoxlamasını **təmizlənməmiş** sualla aparırdı. Şərt həmişə
doğru çıxırdı, yəni tam sual **hər dəfə** əlavə olunurdu — bağlayıcısı
olmayan sual üçün belə.

Ölçülmüş nəticə: genişləndirmə sətirləri **108** chunk çəkib, baseline isə
**48** — 2.25 dəfə artıq büdcə. Cədvəlin öz müqəddiməsi «qapı sabitdir ki,
yalnız bir şey dəyişsin» deyir, halbuki büdcə də dəyişirdi.

Daha kəskin ikinci səbəb: 12 dev sualının **9-unda bağlayıcı yoxdur**, yəni
onların «genişləndirməsi» eyni sualın `?`-siz nüsxəsi idi. Ona görə «8/8»
nəticəsinin sual bölgüsündən, yoxsa təkrarlanan embedding-dən gəldiyini
mövcud artefakt **ayırd edə bilmir**.

Geri götürülür:
- `sorğu genişləndirmə` və `genişləndirmə + 300/90` cədvəl sətirləri
- «Sorğu genişləndirmə indeksə toxunmadan işləyir» abzası
- «Tövsiyə» bölməsindəki «Sorğu genişləndirmə də eyni nəticəni verir» abzası

## Nə pozulmayıb — sərhəd açıq deyilir

Chunking sətirlərinin hamısı **büdcə-bərabərdir** (12 sorğu / 48 çəkilən
chunk, artefaktda yoxlanıla bilər), ona görə aşağıdakılar toxunulmur:

- `chunking 500/150` və qalan bütün chunking sətirləri
- `embedding 3-large` sətirləri
- **`CHUNK_SIZE=500` tövsiyəsi**
- Uçdan-uca dev run (`20260812T094516Z-dev-baseline`)
- Holdout təsdiqi və öncədən qeyd edilmiş proqnoz
- `logs/judge_bias.md`, `logs/before_after.md`

Sərhədini göstərməyən retraksiya geri götürdüyü iddia qədər etibarsızdır.

## Düzəliş

`sub_queries()` indi bölgünü **yalnız hər iki tərəf müstəqil sual kimi
oxunanda** qəbul edir; bölgü baş tutmasa nəticə **tək** sorğudur. Bu, həm
təkrarlanan nüsxəni, həm də «iki ismi birləşdirən `və`» halını həll edir
(«Backup və restore prosedurları nədir?» əvvəllər birinci ismi silirdi).
12 dev sualı üçün cəmi sorğu **27 → 16**.

Büdcə artıq hər sətirdə rəqəmlə yazılır (`retrieval_budget`) və baseline ilə
uyğun gəlməyən sətir `summary.md`-də işarələnir; alət `exit 3` qaytarır
(`--allow-unmatched-budget` ilə söndürülür). Sətir həmişə ölçülür və yazılır
— sübut məhv edilmir. Genişləndirmə indi **iki** sətirdə ölçülür: büdcə
bərabər və büdcə sərbəst, yəni əlavə büdcənin nə qazandırdığı görünür.

Yenidən ölçmənin nəticəsi aşağıdakı bölmədədir.

---

# YENİDƏN ÖLÇÜLDÜ (2026-08-12) — büdcə bərabərləşdirildi

Aşağıdakı cədvəl **iki dəfə, müstəqil olaraq** istehsal olunub: `111502Z` və
`191757Z` icraları eyni 9 sətri — chunk sayı, örtük, sızma, büdcə — hərfi-hərfinə
təkrarladı. İkinci icra artefaktdakı `--workdir` yolunun təmizlənməsi
(`eval.artifacts.argv_temizle`) üçün lazım idi; nəticənin dəyişməməsi isə
əlavə fayda oldu — ölçmənin təkrar istehsal olunduğunun sübutu. Saxlanılan
artefakt ikincisidir.

<!-- artefakt: 20260812T191757Z-eksperiment-chunking+embedding+genişləndirmə -->

| eksperiment | indeks chunk | örtük | sızma | retrieval büdcəsi |
|---|---|---|---|---|
| baseline (800/200) | 19 | 7/8 | 1/2 | 48 |
| chunking 500/150 | 29 | 8/8 | 1/2 | 48 |
| chunking 400/120 | 35 | 8/8 | 1/2 | 48 |
| chunking 300/90 | 47 | 8/8 | 2/2 | 48 |
| chunking 250/60 | 54 | 8/8 | 2/2 | 48 |
| embedding 3-large | 19 | 7/8 | 1/2 | 48 |
| 3-large + chunking 300/90 | 47 | 8/8 | 1/2 | 48 |
| genişləndirmə (büdcə bərabər) | 19 | 8/8 | 1/2 | 46 ↓ büdcə az |
| genişləndirmə (büdcə sərbəst) | 19 | 8/8 | 1/2 | 64 ✗ BÜDCƏ ARTIQ |

<!-- /artefakt -->

## Chunking nəticəsi dəyişmədi

`chunking 500/150` yenə **8/8 örtük, 1/2 sızma, 48 büdcə** — baseline ilə
eyni büdcədə. `CHUNK_SIZE=500` tövsiyəsi və onun üzərində qurulmuş dev və
holdout run-ları toxunulmamış qalır.

## Genişləndirmə: düzəldilmiş ölçmə nəticəni ZƏİFLƏTMƏDİ, GÜCLƏNDİRDİ

Bu, gözlənilən nəticə deyildi və açıq deyilməlidir. Retraksiya
genişləndirmənin 2.25 dəfə artıq büdcə ilə ölçüldüyünü qeyd etmişdi;
düzəlişdən sonra gözlənti nəticənin pisləşməsi idi.

Əksi oldu: büdcə-bərabər rejimdə genişləndirmə **46** chunk çəkir — yəni
baseline-dan (**48**) **az** — və buna baxmayaraq örtüyü tamamlayır, sızmanı
isə artırmır. Səbəb bölgünün indi düzgün işləməsidir: 12 dev sualından
yalnız 2-si bölünür, hər biri 3 sorğuya, və hər sorğu `k // 3 = 1` chunk
çəkir. Yəni əlavə sorğu ümumi büdcəni artırmır, onu daha yaxşı paylayır.

«↓ büdcə az» işarəsi problem deyil: eyni örtüyü daha ucuz almaq nəticəni
gücləndirir. Alət yalnız **artıq** büdcəni uyğunsuz sayır.

Büdcə-sərbəst sətir (64) məhz müqayisə üçün saxlanılır: əlavə büdcənin
**heç nə qazandırmadığını** göstərir — örtük və sızma eynidir.

## Tövsiyə dəyişmir, amma səbəbi dəqiqləşir

`CHUNK_SIZE=500` **birinci** qalır, çünki uçdan-uca dev və holdout run-ları
ilə təsdiqlənib — genişləndirmə isə yalnız örtük səviyyəsində ölçülüb.

Genişləndirmə artıq «eyni nəticəni verir, amma kod tələb edir» deyil, daha
dəqiq bir şeydir: **indeksə toxunmadan, baseline-dan az büdcə ilə eyni
örtüyü verir.** Bu, onu növbəti dövr üçün ciddi namizəd edir.

Məhdudiyyət qalır və o, mexanizmdən görünür: bölgü əl ilə seçilmiş
bağlayıcılara baxır, holdout-un multi-hop sualı isə hopları «ilə» ilə
birləşdirir — yəni orada bölgü işə düşməzdi. Bunu aradan qaldırmaq üçün
bağlayıcı siyahısını uzatmaq yox, sintaktik təhlil və ya model lazımdır; o
isə ayrıca dövrdür.
