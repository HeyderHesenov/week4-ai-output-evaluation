# Generasiya qatı ölçmə dövrü — dizayn (2026-08-14)

## Niyə bu dövr

Retrieval dövrü bağlandı: `CHUNK_SIZE=500 / CHUNK_OVERLAP=150` holdout-da
15/24 → 18/24 verdi və göndərilməsi tövsiyə olundu. Qalan **iki** holdout
uğursuzluğu retrieval qatında deyil — hər ikisi generasiya qatındadır və
sübutu saxlanmış artefaktdan, pul xərclənmədən çıxarılıb:

| case | simptom | təkrar | düzgün fakt kontekstdə idimi |
|---|---|---|---|
| `hold_false_premise_dord_gun` | `generation_wrong` (determinist ədəd yoxlaması) | 3/3 | **[1] — ən yüksək ballı chunk (0.615)**: «Əməkdaşlar həftədə maksimum **üç gün** uzaqdan işləyə bilər» |
| `hold_ambiguous_hesabat` | `over_refusal` | 3/3 | **[4]** (0.475): «İnsident hesabatı **48 saat** ərzində rəsmiləşdirilir» |

Birinci halda model yanlış müqəddiməni qəbul etdi və təsdiq prosedurunu izah
etdi — həddi heç anmadan. İkinci halda cavab kontekstdə ola-ola «Sənədlərdə bu
suala cavab tapılmadı» dedi. **Hər ikisi 3 təkrarın hamısında baş verdi**, yəni
səs-küy deyil.

Mexanizm ehtimalı SUT-un öz qaydalarındadır: 5-ci qayda «kontekstdə tam və ya
qismən cavab YOXDURSA imtina et», 6-cı qayda «qismən əminlik kifayət deyil».
`data/variants/v1_tam_cavab.yaml`-ın öz mənşə qeydi də bu ikisini «şübhə varsa
imtina et» davranışının mənbəyi kimi göstərmişdi.

**Ehtiyatlılıq üçün əsas:** `v1_tam_cavab` məhz bu qaydalara toxundu, dev-də
10/12 → 11/12 (+2/−1) verdi və **holdout-a transfer olunmadı** (+0/−0). Yəni bu
SUT-da «qayda əlavə et» yanaşmasının bir dəfə uğursuz olduğuna dair ölçülmüş
sübutumuz var. Dizayn buna görə ucuz qapılar üzərində qurulub.

## Ölçmə problemi — dizaynı bu müəyyən edir

Dev bu dövrü olduğu kimi ölçə bilmir:

- `dev_false_premise_free_cache` — **keçir**. Yanlış müqəddimə uğursuzluğunun
  dev-də heç bir siqnalı yoxdur.
- `dev_ambiguous_limit` — uğursuzdur, amma `gradable: judge`, və
  **kappa = 0.13**. Qeyri-müəyyənlik yarısının yeganə dev siqnalı zəif ölçülmüş
  hakimdən keçir.
- Holdout **3 icradadır**; ona qarşı tənzimləmək kor-yoxlama iddiasını məhv
  edər.

Ona görə dövr **əvvəlcə ölçmə qabiliyyətini** qurur, sonra müdaxiləni sınayır.

## Dövrün forması — beş addım, hər biri növbətinin qapısı

İstənilən mənfi nəticədə dövr **orada bağlanır və yazılır**. Mənfi nəticə
təslimatdır: retrieval sweep-i də «dörd oxun heç biri işləmir» dedi və həmin
sənəd dəyərli oldu.

| # | addım | qiymət | qapı |
|---|---|---|---|
| 0 | 2 yeni dev case + möhürün yenilənməsi | pulsuz | Jaccard oxşarlığı ölçülür və çap olunur |
| 1 | genişlənmiş dev-də baseline run | ~$0.05 | yeni case-lər uğursuzluğu təkrar istehsal edirmi |
| 2 | generation probe — saxlanmış kontekst təkrar oynadılır | **~$0.02** | hansısa prompt qolu düzəldirmi |
| 3 | qazanan qoldan `v2a`/`v2b`, ayrıca dev run-ları | ~$0.05 hərəsi | dev-də qazanc var, reqressiya yox |
| 4 | **bir** holdout icrası, öncədən qeydə alınmış proqnozla | ~$0.10 | ledger 3 → 4 |

Addım 2-nin qiyməti **ölçülmüş rəqəmdən** törəyir, təxmindən yox:
`runs/20260812T095145Z-holdout-baseline` artefaktında 24 sintez çağırışı
cəmi 32,328 giriş + 705 çıxış tokeni idi, yəni `gpt-4o-mini` qiymətləri ilə
**$0.005**. Həmin run-ın $0.0966 xərcinin qalanı **hakimdəndir** (`claude-opus-5`,
$5/$25 per Mtok) — probe isə hakim çağırmır. Probe 72 çağırış edir (aşağıya
bax) və qol suffiksləri giriş tokenini bir qədər artırır, ona görə $0.02 yuxarı
həddir.

Ən pis hal (addım 2 mənfi): dövr **~$0.07**-yə bağlanır, holdout heç
toxunulmur.

## Addım 0 — yeni dev case-ləri

**İki** case əlavə olunur. Üçüncüyə ehtiyac yoxdur: qeyri-müəyyənlik qaydasının
imtina qapısını həddən artıq boşaltmadığını dev-dəki **mövcud iki
`out_of_corpus` case-i** yoxlayacaq — onlar bu dövrün nəzarət qrupudur.

1. **ikinci `false_premise`** — mövcud dev case-i keçdiyi üçün davranışı
   həqiqətən sıxan biri lazımdır.
2. **ikinci `ambiguous`, `gradable: both`** — determinist lövbəri olsun ki,
   kappa = 0.13 olan hakim yeganə siqnal olmasın.

### Çirklənmə qadağaları (məcburi, spesifikasiyanın icra olunan hissəsi)

Case-ləri yazan tərəf (assistant) bu sessiyada holdout-un hər iki uğursuz
cavabını tam mətni ilə oxudu. Bu, real risk yaradır: fərqinə varmadan holdout
sualının pərdələnmiş nüsxəsi yazıla bilər, və onda «dev-də işlədi, holdout-da
da işlədi» nəticəsi özünü təsdiqləyən dövrəyə çevrilər. Qadağalar:

- case yalnız **korpus sənədlərinə** baxaraq yazılır; müəlliflik zamanı holdout
  sualının mətninə **yenidən baxılmır**;
- hər yeni case holdout qarşılığından **fərqli sənəd və fərqli fakt** işlədir
  (`hold_false_premise_dord_gun` → `sirket_qaydalari.pdf` / uzaqdan iş həddi;
  `hold_ambiguous_hesabat` → hesabat müddətləri — yeni case-lər bunların heç
  birini işlətməyəcək);
- hər yeni sualın **hər holdout sualı ilə token-səviyyəli Jaccard oxşarlığı**
  ölçülür. Karkasda bu mexanizm variant fraqmentləri üçün artıq var
  (`eval/split.py`, 3-cü mexanizm) — case mətnlərinə də tətbiq olunur.
  **İddia yox, rəqəm.** Ölçülən maksimum dəyər `logs/generation_cycle.md`-də
  case-lərin əlavə olunması bölməsində çap olunur (bu addımda hələ probe
  artefaktı yoxdur), və bir test onu hesablanan saxlayır: yeni dev sualı ilə
  istənilən holdout sualı arasındakı Jaccard mövcud dev/holdout cütlərində
  müşahidə olunan maksimumdan yuxarı olsa, test uğursuz olur.
- Heyder case-ləri təsdiqləyir; təsdiqdən əvvəl heç bir run işlədilmir.

### Möhürün yenilənməsi

`python -m eval.cli seal-split --force`.

- `holdout_ids` **dəyişmir** — bölgü YAML-dakı `split:` sahəsindən oxunur
  (`eval/split.py:89-90`), təsadüfi bölünmə yoxdur. İcradan sonra bu, çap
  olunur və testlə yoxlanılır.
- `dataset_sha256` dəyişir. Nəticə: **əvvəlki dev run-ları ilə müqayisə yalnız
  ortaq 12 case üzərində aparılır**; 2 yeni case ayrıca göstərilir və onların
  «əvvəli» yoxdur.
- Hesabatda açıq deklarasiya olunur — karkasın öz `ContaminationError` mesajı
  bunu tələb edir: «Dəyişiklik qəsdəndirsə: bölgünü yenidən möhürləyin və bunu
  hesabatda AÇIQ qeyd edin».

## Addım 2 — `tools/generation_probe.py`

Dövrün ürəyi. Sual: **kontekst düzgün olduqda, model qaydaların təzyiqi altında
düzgün cavabı ümumiyyətlə verə bilirmi?**

### Metod

Saxlanmış dev run-ının `observations.jsonl`-ından chunk-lar götürülür və
**retrieval heç işlədilmir** — nə embedding, nə indeks, nə vektor axtarışı.
Retrieval beləliklə konstruksiyaya görə sabit qalır, yəni ölçülən yeganə şey
generasiyadır.

Prompt SUT-un öz qurucuları ilə bərpa olunur, yenidən yazılmır:
`rag.pipeline.build_user_message(question, chunks, nonce)`,
`rag.pipeline.make_nonce()`, `rag.pipeline.SYSTEM_INSTRUCTION`. Saxlanmış
`ChunkView` → SUT-un `RetrievedChunk`-ı üçün kiçik adapter yazılır; adapterin
sahə dəsti testlə kilidlənir, çünki sahə uyğunsuzluğu bərpanı səssizcə
sadiqsiz edərdi.

### Qollar

| qol | əlavə olunan `system_suffix` |
|---|---|
| **0 — nəzarət** | heç nə; SUT-un prompt-u olduğu kimi |
| 1 | yanlış müqəddimə: sualın fərziyyəsi kontekstlə ziddiyyət təşkil edirsə, düzgün faktı de və fərziyyəni açıq düzəlt |
| 2 | qeyri-müəyyənlik: kontekst bir neçə fərqli oxunuşu dəstəkləyirsə, nə səssizcə birini seç, nə də imtina et — oxunuşları adlandır |
| 3 | 1 + 2 birlikdə — qarşılıqlı təsiri görmək üçün |

Case dəsti: yeni iki dev case + `dev_ambiguous_limit` +
`dev_false_premise_free_cache` (sonuncu keçir, ona görə **reqressiya
nəzarətidir**: qol 1 və 3 onu sındırmamalıdır) + iki `out_of_corpus` case
(qol 2-nin imtina qapısını boşaltmadığını yoxlayır).

**Hər qol 3 dəfə təkrarlanır** — run protokolu ilə eyni. Səbəb: model
determinist deyil, ona görə bir dəfə düzələn cavab təsadüf ola bilər və
«qol 1 işləyir» nəticəsi tək nümunədən çıxarıla bilməz. Uğursuzluqların özü
3/3 determinist idi, müalicənin də eyni ölçüdə yoxlanması simmetrikdir.

6 case × 4 qol × 3 təkrar = **72 çağırış**, hakimsiz, ~$0.015.

### Etibarlılıq qapısı — nəzarət qolu

**Əgər qol 0 orijinal uğursuzluğu təkrar istehsal etmirsə, təkrar oynatma sadiq
deyil və bütün cədvəl etibarsızdır.** Bu, alətin çıxış kodunda ifadə olunur, bir
qeyd kimi yox.

Əlavə olaraq **yalnız qol 0-da** `system_sha256` orijinal run-dakı ilə
tutuşdurulur: system prompt-da nonce yoxdur, ona görə bu, dəqiq yoxlanandır.
Qol 1-3 system prompt-u qəsdən dəyişir, yəni onlarda uyğunsuzluq gözləniləndir
və yoxlanmır — orada `system_sha256` sadəcə qeyd olunur ki, hansı qolun hansı
mətnlə işlədiyi artefaktdan bilinsin.

Cavab mətni təkrar oynatmada hərfi-hərfinə eyni olmaya bilər (model
determinist deyil), ona görə nəzarət **təsnifat səviyyəsində** aparılır: qol 0
`generation_wrong` / `over_refusal` təsnifatını təkrar istehsal etməlidir,
eyni cümlələri yox.

### Təsnifat

**Yalnız determinist.** `eval.graders`-dəki ədəd yoxlaması
(`numeric_claim_satisfied`) və imtina aşkarlanması. Hakim çağırılmır, yəni
kappa = 0.13 bu cədvələ girmir.

### Artefakt

Bu gün (`203df03`) sərtləşdirilmiş `ProbeWriter` müqaviləsi ilə:
`logs/probes/<probe_id>/` — manifest ilk anda `status: yarımçıq`, sonda
`tamam`/`uğursuz`; `rows.jsonl`; `summary.md`. Sənədə köçürülən cədvəl
`summary.md`-də generasiya olunur və `tests/test_logs_iddialari.py` onu
yoxlayır.

## Addım 3 — variantlar

Qol 2-nin nəticəsindən asılı olaraq `data/variants/v2a_*.yaml` (yanlış
müqəddimə) və `v2b_*.yaml` (qeyri-müəyyənlik) yazılır — mövcud
`system_suffix` + `few_shot` sxemi ilə. Few-shot nümunələri üçün karkasın
mövcud qaydası qüvvədədir: `source_case_ids ⊆ dev_ids`.

**Ayrı-ayrı ölçülür**, çünki `v1`-in dərsi budur ki, birləşmiş dəyişikliyin
hansı hissəsinin işlədiyi ayırd edilə bilmirdi. Dev run-ları ucuzdur; holdout
isə hər iki qayda üçün **bir** icra ilə kifayətlənir, çünki iki holdout
uğursuzluğu **fərqli case-lərdir** — atribusiya hansı case-in döndüyündən
gəlir, hansı variantın işlədiyindən yox.

## Addım 4 — holdout protokolu

- Proqnoz holdout icrasından **əvvəl** `logs/generation_cycle.md`-ə yazılır:
  hansı case dönəcək və **hansı mexanizmə görə**. Retrieval dövründə bu nümunə
  işlədi və proqnoz dəqiq tutdu.

Dövrün bütün mətn qeydi **tək bir sənəddə** — `logs/generation_cycle.md` —
saxlanılır: case-lərin əlavəsi və Jaccard rəqəmləri, probe cədvəli, variant
nəticələri, öncədən qeydiyyat, holdout nəticəsi. Səbəb: retrieval dövründə qeyd
üç fayla (`retrieval_sweep.md`, `retrieval_experiments.md`, `before_after.md`)
dağıldı və hansı iddianın hansı artefaktla dəstəkləndiyini izləmək çətinləşdi —
retraksiya markerləri məhz oradan doğuldu.
- **Bir** icra. Ledger 3 → 4.
- p-dəyər gözləntisi əvvəlcədən yazılır: 8 holdout case ilə dəqiq ikitərəfli
  McNemar p < 0.05 üçün **6 diskordant cüt** lazımdır, burada isə generasiya
  qatında cəmi 2 case uğursuzdur. Yəni p = 1.000 gözlənilir və bu, **güc**
  ifadəsidir, effekt ifadəsi deyil. Sübut öncədən qeydiyyat + hər üç təkrarda
  ardıcıllıq + determinist təsnifat + sıfır reqressiyadır.

## Testlər

Yeni kod üçün, repo-nun qaydası ilə (əvvəl uğursuz olan test, sonra düzəliş):

- `ChunkView` → `RetrievedChunk` adapteri: sahə dəsti kilidlənir;
- nəzarət qolu qapısı: qol 0 uğursuzluğu təkrar istehsal etmirsə alət sıfırdan
  fərqli kod qaytarır;
- Jaccard ölçməsi: bilinən cüt üçün gözlənilən dəyər;
- möhürün additivliyi: `seal-split --force`-dan sonra `holdout_ids` dəyişməyib;
- probe artefaktının `status` müqaviləsi (mövcud testlər genişləndirilir).

Bütün testlər **açarsız və şəbəkəsiz** qalır — CI-nin mövcud müqaviləsi budur.

## Əhatə dairəsindən KƏNAR

- Retrieval parametrləri və chunking — həmin dövr bağlanıb.
- Hakim kalibrasiyası (kappa raund 2) — ayrıca dövrdür; bu dizayn hakimi
  ölçmə yolundan çıxarmaqla ona ehtiyacı aradan qaldırır.
- SUT kodunun redaktəsi — pin edilmiş commit toxunulmaz qalır; bütün müdaxilə
  `system_suffix` vasitəsilədir.
- Köhnə run manifestlərindəki mütləq yol məsələsi — ayrıca, bu dövrlə əlaqəsiz.
