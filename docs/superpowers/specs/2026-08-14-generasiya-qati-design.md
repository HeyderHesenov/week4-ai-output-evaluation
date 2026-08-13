# Generasiya qatı ölçmə dövrü — dizayn (2026-08-14)

## Niyə bu dövr

Retrieval dövrü bağlandı: `CHUNK_SIZE=500 / CHUNK_OVERLAP=150` holdout-da
15/24 → 18/24 verdi. Qalan **iki** holdout uğursuzluğu retrieval qatında deyil —
hər ikisi generasiya qatındadır və sübutu saxlanmış artefaktdan, pul
xərclənmədən çıxarılıb:

| case | simptom | təkrar | düzgün fakt kontekstdə idimi |
|---|---|---|---|
| `hold_false_premise_dord_gun` | `generation_wrong` | 3/3 | **[1] — ən yüksək ballı chunk (0.615)**: «Əməkdaşlar həftədə maksimum **üç gün** uzaqdan işləyə bilər» |
| `hold_ambiguous_hesabat` | `over_refusal` | 3/3 | **[4]** (0.475): «İnsident hesabatı **48 saat** ərzində rəsmiləşdirilir» |

Retrieval hər iki halda qüsursuz işlədi. **Hər ikisi 3 təkrarın hamısında baş
verdi**, yəni səs-küy deyil.

## Hipotez — dörd müşahidə, bir mexanizm

Saxlanmış cavabları oxumaq göstərir ki, bu iki uğursuzluq ayrı-ayrı qüsurlar
deyil, **eyni boşluğun iki üzüdür**:

| case | model nə etdi | nəticə |
|---|---|---|
| `hold_false_premise_dord_gun` | müqəddiməni qəbul edib təsdiq prosedurunu izah etdi | uğursuz |
| `hold_ambiguous_hesabat` | «Sənədlərdə bu suala cavab tapılmadı» | uğursuz |
| `dev_ambiguous_limit` | «Limit 200-dür [1]» — səssizcə bir oxunuşu seçdi | uğursuz (hakim balı 1) |
| `dev_false_premise_free_cache` | «Sənədlərdə bu suala cavab tapılmadı» | **keçir** — aşağıya bax |

SUT-un yalnız **iki hərəkəti** var: cavab ver, yaxud imtina et. Üçüncünü heç
vaxt etmir — **sualın strukturuna münasibət bildirmək**: müqəddiməni düzəltmək,
yaxud oxunuşları adlandırmaq. Mexanizm SUT-un öz qaydalarındadır: 5-ci qayda
«kontekstdə tam və ya qismən cavab YOXDURSA imtina et», 6-cı qayda «qismən
əminlik kifayət deyil». `data/variants/v1_tam_cavab.yaml`-ın mənşə qeydi də bu
ikisini eyni davranışın mənbəyi kimi göstərmişdi.

Model hansı hərəkəti seçir — lokal olaraq hansı asandırsa: ağlabatan görünən
rəqəm varsa onu yapışdırır, yoxsa imtina edir.

### Niyə `dev_false_premise_free_cache`-in keçməsi bacarıq sübutu DEYİL

İki yanlış müqəddimə case-inin determinist yoxlamaları **əks
istiqamətlərdədir**:

| | `dev_false_premise_free_cache` | `hold_false_premise_dord_gun` |
|---|---|---|
| yoxlama | `not_contains: ['128']` — **mənfi** | `numeric: 3 gün` — **müsbət** |
| tələb | səhv rəqəmi **deməmək** | düzgün həddi **demək** |
| modelin faktiki cavabı | «Sənədlərdə bu suala cavab tapılmadı» | müqəddiməni qəbul etdi |

Dev case-i **imtina ilə keçir**: mənfi yoxlamanı boş cavab avtomatik təmin edir.
Case öz rubrikasına görə düzgün keçir (rubrika imtinaya icazə verir), amma
müqəddimə düzəltmə bacarığı haqqında **sıfır məlumat verir**.

**Bu, karkasın özü haqqında ayrıca tapıntıdır və dövrdən asılı olmayaraq
yazılır:** mənfi `not_contains` yoxlaması «düzgün ehtiyatlı»nı «düzgün
düzəldici»dən ayıra bilmir. Ona görə də dev-in mövcud yanlış müqəddimə case-i
bu dövr üçün tənzimləmə siqnalı ola bilməz.

## Ölçmə problemi

- `dev_false_premise_free_cache` — keçir, amma yuxarıdakı səbəbdən siqnal
  daşımır.
- `dev_ambiguous_limit` — uğursuzdur, amma `gradable: judge`, və
  **kappa = 0.13**.
- Holdout **3 icradadır**; ona qarşı tənzimləmək kor-yoxlama iddiasını məhv
  edər.

Ona görə dövr **ucuz diaqnostika ilə başlayır və möhürə yalnız hipotez sağ
qalarsa toxunur**.

## Dövrün forması — beş addım, hər biri növbətinin qapısı

İstənilən mənfi nəticədə dövr **orada bağlanır və yazılır**. Mənfi nəticə
təslimatdır: retrieval sweep-i də «dörd oxun heç biri işləmir» dedi və həmin
sənəd dəyərli oldu.

| # | addım | qiymət | qapı |
|---|---|---|---|
| 1 | **generation probe** — mövcud möhürlənmiş case-lər üzərində | **~$0.01** | hansısa prompt qolu davranışı dəyişirmi |
| 2 | 2 yeni dev case (**müsbət** yoxlama ilə) + `seal-split --force` | pulsuz | Jaccard ölçülür; Heyder təsdiqləyir |
| 3 | genişlənmiş dev-də baseline run | ~$0.05 | yeni case-lər uğursuzluğu təkrar istehsal edirmi |
| 4 | qazanan qoldan variant(lar), ayrıca dev run-ları | ~$0.05 hərəsi | dev-də qazanc var, reqressiya yox |
| 5 | **bir** holdout icrası, öncədən qeydə alınmış proqnozla | ~$0.10 | ledger 3 → 4 |

**Sıra qəsdəndir.** Əvvəlki qaralamada möhür addım 0-da sındırılırdı — yəni
`dataset_sha256` hipotez sınanmazdan əvvəl dəyişirdi və bütün əvvəlki dev
run-larının müqayisə edilə bilməsi ödənilirdi. Probe-a yeni case **lazım
deyil**, ona görə möhür yalnız addım 1 müsbət çıxarsa toxunulur.

Ən pis hal (addım 1 mənfi): dövr **~$0.01**-ə bağlanır, möhür toxunulmur,
holdout toxunulmur.

Xərc **ölçülmüş rəqəmdən** törəyir, təxmindən yox: `20260812T095145Z-holdout-baseline`
artefaktında 24 sintez çağırışı cəmi 32,328 giriş + 705 çıxış tokeni idi, yəni
`gpt-4o-mini` qiymətləri ilə **$0.005**. Həmin run-ın $0.0966 xərcinin qalanı
**hakimdəndir** (`claude-opus-5`, $5/$25 per Mtok) — probe isə hakim çağırmır.

## Addım 1 — `tools/generation_probe.py`

Dövrün ürəyi. Sual: **kontekst düzgün olduqda, prompt qaydası modeli üçüncü
hərəkətə (struktura münasibət) məcbur edə bilirmi?**

### Metod

Saxlanmış dev run-ının (`20260812T094516Z-dev-baseline`) `observations.jsonl`-ından
chunk-lar götürülür və **retrieval heç işlədilmir** — nə embedding, nə indeks,
nə vektor axtarışı. Retrieval konstruksiyaya görə sabit qalır, yəni ölçülən
yeganə şey generasiyadır.

**Yeni case yaradılmır, möhürə toxunulmur.** Mövcud dörd case onsuz da hər
davranışı göstərir.

Prompt SUT-un öz qurucuları ilə bərpa olunur, yenidən yazılmır:
`rag.pipeline.build_user_message(question, chunks, nonce)`,
`rag.pipeline.make_nonce()`, `rag.pipeline.SYSTEM_INSTRUCTION`. Saxlanmış
`ChunkView` → SUT-un `RetrievedChunk`-ı üçün adapter yazılır; adapterin sahə
dəsti testlə kilidlənir, çünki sahə uyğunsuzluğu bərpanı səssizcə sadiqsiz
edərdi.

### Qollar

| qol | əlavə olunan `system_suffix` |
|---|---|
| **0 — nəzarət** | heç nə; SUT-un prompt-u olduğu kimi |
| 1 | **müqəddimə**: sualın fərziyyəsi kontekstlə ziddiyyət təşkil edirsə, düzgün faktı AÇIQ de və fərziyyəni düzəlt; imtina bu halda kifayət deyil |
| 2 | **qeyri-müəyyənlik**: kontekst bir neçə fərqli oxunuşu dəstəkləyirsə, nə səssizcə birini seç, nə də imtina et — oxunuşları adlandır |
| 3 | 1 + 2 birlikdə — vahid mexanizm hipotezini yoxlayır |

Qol 3 sadəcə birləşmə deyil, **hipotezin sınağıdır**: əgər dörd uğursuzluq
doğrudan bir mexanizmdəndirsə, qol 1 və qol 2 bir-birinin case-lərinə də təsir
etməlidir.

### Case dəsti

| case | nə yoxlayır |
|---|---|
| `dev_false_premise_free_cache` | qol 1 imtinanı **aktiv düzəlişə** çevirirmi (və 128-i deməməyi saxlayırmı) |
| `dev_ambiguous_limit` | qol 2 «Limit 200-dür» səssiz seçimini dayandırırmı |
| `dev_out_of_corpus_graphql` | **reqressiya nəzarəti** — qol 2 imtina qapısını sındırmamalıdır |
| `dev_out_of_corpus_ceo` | eyni nəzarət |

**Hər qol 3 dəfə təkrarlanır** — run protokolu ilə eyni. Model determinist
deyil, ona görə bir dəfə düzələn cavab təsadüf ola bilər; uğursuzluqların özü
3/3 idi, müalicənin də eyni ölçüdə yoxlanması simmetrikdir.

4 case × 4 qol × 3 təkrar = **48 çağırış**, hakimsiz.

### Etibarlılıq qapısı — nəzarət qolu

**Əgər qol 0 orijinal davranışı təkrar istehsal etmirsə, təkrar oynatma sadiq
deyil və bütün cədvəl etibarsızdır.** Bu, alətin çıxış kodunda ifadə olunur, bir
qeyd kimi yox.

Nəzarət **təsnifat səviyyəsindədir**, hərfi mətn səviyyəsində yox: model
determinist deyil, ona görə qol 0-dan tələb olunan şey eyni cümlələr deyil,
eyni davranış sinfidir (`dev_false_premise_free_cache` → imtina;
`dev_ambiguous_limit` → tək oxunuşlu cavab; hər iki `out_of_corpus` → imtina).

Əlavə olaraq **yalnız qol 0-da** `system_sha256` orijinal run-dakı ilə
tutuşdurulur: system prompt-da nonce yoxdur, ona görə bu, dəqiq yoxlanandır.
Qol 1-3 system prompt-u qəsdən dəyişir; orada dəyər sadəcə qeyd olunur ki,
hansı qolun hansı mətnlə işlədiyi artefaktdan bilinsin.

### Təsnifat

**Yalnız determinist** — hakim çağırılmır, yəni kappa = 0.13 bu cədvələ girmir:

- imtina aşkarlanması (SUT-un öz imtina cümləsi);
- ədəd yoxlaması `eval.graders.numeric_claim_satisfied` (`not_contains` daxil);
- **müqəddimə düzəlişi**: cavab kontekstdəki düzgün faktı ehtiva edirmi
  (`dev_false_premise_free_cache` üçün doğru planın adı, `dev_ambiguous_limit`
  üçün birdən çox limitin adlanması) — sadə açar-söz yoxlaması, hakim yox.

Bu üçüncü meyar **qabaqcadan yazılır və artefaktda çap olunur**, cədvələ
baxdıqdan sonra seçilmir.

### Artefakt

`203df03`-də sərtləşdirilmiş `ProbeWriter` müqaviləsi ilə:
`logs/probes/<probe_id>/` — manifest ilk anda `status: yarımçıq`, sonda
`tamam`/`uğursuz`; `rows.jsonl`; `summary.md`. Sənədə köçürülən cədvəl
`summary.md`-də generasiya olunur və `tests/test_logs_iddialari.py` onu
yoxlayır.

## Addım 2 — yeni dev case-ləri (yalnız addım 1 müsbətdirsə)

**İki** case. Üçüncüyə ehtiyac yoxdur: qeyri-müəyyənlik qaydasının imtina
qapısını boşaltmadığını dev-dəki mövcud iki `out_of_corpus` case-i yoxlayır.

1. **ikinci `false_premise`, `numeric` (MÜSBƏT) yoxlama ilə** — mövcud dev
   case-i mənfi yoxlama işlətdiyi üçün imtina ilə keçir və siqnal daşımır. Yeni
   case düzgün faktın **deyilməsini** tələb etməlidir.
2. **ikinci `ambiguous`, `gradable: both`** — determinist lövbəri olsun ki,
   kappa = 0.13 olan hakim yeganə siqnal olmasın.

### Çirklənmə qadağaları (spesifikasiyanın icra olunan hissəsi)

Case-ləri yazan tərəf (assistant) bu sessiyada holdout-un hər iki uğursuz
cavabını tam mətni ilə oxudu. Qadağalar:

- case yalnız **korpus sənədlərinə** baxaraq yazılır; müəlliflik zamanı holdout
  sualının mətninə **yenidən baxılmır**;
- hər yeni case holdout qarşılığından **fərqli sənəd və fərqli fakt** işlədir
  (holdout cütü `sirket_qaydalari.pdf`-dəki uzaqdan iş həddini və hesabat
  müddətlərini işlədir — yeni case-lər bunların heç birinə toxunmayacaq);
- hər yeni sualın **hər holdout sualı ilə token-səviyyəli Jaccard oxşarlığı**
  ölçülür. Mexanizm karkasda mövcuddur (`eval/split.py:134 jaccard()`, hədd
  0.60) — case mətnlərinə də tətbiq olunur. **İddia yox, rəqəm.** Dəyər
  `logs/generation_cycle.md`-də çap olunur və bir test onu hesablanan saxlayır.
- Heyder case-ləri təsdiqləyir; təsdiqdən əvvəl heç bir run işlədilmir.

**Qeyd — əsas müdafiə qadağalar deyil, müdaxilənin formasıdır:** variant
ümumi prompt QAYDASIDIR, case-ə xas yamaq deyil. Case-ə uyğunlaşdırılmış
düzəliş mümkün olmadığı üçün «dev-də işlədi» ilə «holdout-da işlədi» arasındakı
əlaqə yalnız ümumiləşmədən keçə bilər.

### Möhürün yenilənməsi

`python -m eval.cli seal-split --force`.

- `holdout_ids` **dəyişmir** — bölgü YAML-dakı `split:` sahəsindən oxunur
  (`eval/split.py:89-90`), təsadüfi bölünmə yoxdur. İcradan sonra çap olunur və
  testlə yoxlanılır.
- `dataset_sha256` dəyişir. Nəticə: **əvvəlki dev run-ları ilə müqayisə yalnız
  ortaq 12 case üzərində qanunidir**; 2 yeni case ayrıca göstərilir və onların
  «əvvəli» yoxdur.
- `logs/generation_cycle.md`-də açıq deklarasiya olunur — karkasın öz
  `ContaminationError` mesajı bunu tələb edir.

## Addım 4 — variantlar

Addım 1-in nəticəsindən asılı olaraq `data/variants/`-ə yazılır: qol 1 və qol 2
ayrı-ayrılıqda işləyirsə iki variant (`v2a`, `v2b`), yalnız qol 3 işləyirsə bir
birləşmiş variant. Mövcud `system_suffix` + `few_shot` sxemi; few-shot üçün
karkasın qaydası qüvvədədir: `source_case_ids ⊆ dev_ids`.

Holdout üçün **bir** icra kifayətdir, çünki iki holdout uğursuzluğu **fərqli
case-lərdir** — atribusiya hansı case-in döndüyündən gəlir, hansı variantın
işlədiyindən yox.

## Addım 5 — holdout protokolu

- Proqnoz holdout icrasından **əvvəl** `logs/generation_cycle.md`-ə yazılır:
  hansı case dönəcək və **hansı mexanizmə görə**. Retrieval dövründə bu nümunə
  işlədi və proqnoz dəqiq tutdu.
- **Bir** icra. Ledger 3 → 4.
- p-dəyər gözləntisi əvvəlcədən yazılır: 8 holdout case ilə dəqiq ikitərəfli
  McNemar p < 0.05 üçün **6 diskordant cüt** lazımdır, burada isə generasiya
  qatında cəmi 2 case uğursuzdur. Yəni p = 1.000 gözlənilir və bu, **güc**
  ifadəsidir, effekt ifadəsi deyil. Sübut öncədən qeydiyyat + hər üç təkrarda
  ardıcıllıq + determinist təsnifat + sıfır reqressiyadır.

## Sənədləşdirmə

Dövrün bütün mətn qeydi **tək** `logs/generation_cycle.md`-də saxlanılır: probe
cədvəli, mənfi/müsbət yoxlama tapıntısı, case əlavəsi və Jaccard rəqəmləri,
variant nəticələri, öncədən qeydiyyat, holdout nəticəsi. Səbəb: retrieval
dövründə qeyd üç fayla dağıldı və hansı iddianın hansı artefaktla
dəstəkləndiyini izləmək çətinləşdi — retraksiya markerləri məhz oradan doğuldu.

## Testlər

Repo-nun qaydası ilə (əvvəl uğursuz olan test, sonra düzəliş):

- `ChunkView` → `RetrievedChunk` adapteri: sahə dəsti kilidlənir;
- nəzarət qolu qapısı: qol 0 davranışı təkrar istehsal etmirsə alət sıfırdan
  fərqli kod qaytarır;
- təsnifat meyarları: bilinən cavab mətnləri üçün gözlənilən təsnifat;
- Jaccard ölçməsi: bilinən cüt üçün gözlənilən dəyər (addım 2-də);
- möhürün additivliyi: `seal-split --force`-dan sonra `holdout_ids` dəyişməyib
  (addım 2-də).

Bütün testlər **açarsız və şəbəkəsiz** qalır — CI-nin mövcud müqaviləsi budur.

## Əhatə dairəsindən KƏNAR

- Retrieval parametrləri və chunking — həmin dövr bağlanıb.
- Hakim kalibrasiyası (kappa raund 2) — ayrıca dövrdür; bu dizayn hakimi ölçmə
  yolundan çıxarmaqla ona olan ehtiyacı aradan qaldırır.
- SUT kodunun redaktəsi — pin edilmiş commit toxunulmaz qalır; bütün müdaxilə
  `system_suffix` vasitəsilədir.
- Mövcud `dev_false_premise_free_cache`-in yoxlamasını dəyişmək — o, möhürlənmiş
  case-dir və rubrikasına görə düzgün işləyir; tapıntı yazılır, case
  redaktə olunmur.
