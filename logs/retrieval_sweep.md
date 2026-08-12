# Retrieval parametr dövrü — ÖLÇÜLDÜ, HİPOTEZ TƏSDİQLƏNMƏDİ (2026-08-12)

## Hipotez

`logs/before_after.md` prompt variantının holdout-a keçmədiyini göstərdi və
uğursuzluqların bir hissəsinin retrieval qatında olduğunu qeyd etdi. Buradan
hipotez: `TOP_K` / `RELEVANCE_THRESHOLD` / `SOFT_FLOOR_MARGIN` dəyişməklə
çatışmayan sənəd kontekstə salına bilər — prompt modelin heç almadığı mətni
bərpa edə bilmir, ona görə növbəti dövr SUT konfiqurasiyasında olmalıdır.

## Metod — `tools/retrieval_sweep.py`

Tam run $0.042-dir və cavab keyfiyyətini ölçür. Lakin hipotezin ilk şərti daha
dardır: **lazımi sənəd ümumiyyətlə qəbul edilirmi?** Bu sual üçün nə sintez,
nə hakim lazımdır. Sweep yalnız sorğu embedding-i işlədir (~$0.001 bütün grid
üçün) və iki rəqəm çıxarır:

| ölçü | mənası |
|---|---|
| **örtük** | datasetdəki `gold_sources` sənədlərinin hamısı qəbul edilmiş chunk-lar arasındadırmı (8 dev case-də `gold_sources` var) |
| **sızma** | korpusda cavabı OLMAYAN suala chunk qəbul edilirmi (2 `out_of_corpus` case) |

İkinci ölçü qəsdən var: təkcə örtüyə baxsaq, astananı sıfıra endirmək
«mükəmməl» görünərdi və sistem hallüsinasiya maşınına çevrilərdi.

Qapı məntiqi **yenidən yazılmayıb** — həm `RagPipeline._retrieve`, həm
`RagPipeline._accepts` SUT-un özündən çağırılır, yoxsa sweep istehsalatdan
fərqli bir şeyi ölçərdi. Yalnız dev case-ləri verilir: parametri holdout
sübutuna baxaraq seçmək holdout-u dev-ə çevirərdi.

Baseline parametrləri ilə sweep müşahidə olunmuş uğursuzluğu **eynilə** təkrar
istehsal etdi (`dev_multihop_mtls_cache`, çatışan `atlas_api_senedi.md`) —
alətin istehsalatla eyni şeyi ölçdüyünün yoxlanışı budur.

## Nəticə: dörd oxun heç biri işləmir

| ox | sınanan | nəticə |
|---|---|---|
| `TOP_K` | 4, 6, 8 | **heç bir fərq yoxdur** — üç dəyər eyni nəticə verir |
| `RELEVANCE_THRESHOLD` | 0.42 → 0.30 | 0.30-da örtük 8/8, amma sızma 1/2 → **2/2** |
| `SOFT_FLOOR_MARGIN` | 0.10 → 0.26 | **heç bir fərq yoxdur** |
| `LEXICAL_THRESHOLD` | 0.35 → 0.12 | 0.17-də örtük 8/8, amma sızma **2/2** |

Örtüyü 8/8 edən HƏR parametr korpusdan kənar qapını da uçurur.

## Səbəb — bir chunk-ın rəqəmləri hər şeyi izah edir

`dev_multihop_mtls_cache` üçün çəkilən chunk-lar (astana 0.42, yumşaq hədd
0.32, leksik astana 0.35):

| # | dense | leksik | mənbə |
|---|---|---|---|
| 1 | 0.457 | 0.339 | atlas_infra_qeydleri.md |
| 2 | 0.411 | 0.275 | atlas_infra_qeydleri.md |
| **3** | **0.335** | **0.173** | **atlas_api_senedi.md** ← lazım olan |
| 4 | 0.320 | 0.104 | atlas_api_senedi.md |

Lazımi chunk **çəkilir** — problem `TOP_K`-da deyil, ona görə k-nı artırmaq heç
nə dəyişmir. O, qapıda ölür:

- dense 0.335 < 0.42 → normal yoldan keçmir;
- yumşaq hədd xilası leksik ≥ 0.35 tələb edir, onunku 0.173 → **marja nə qədər
  böyüsə də xilas işə düşmür.** Marja oxunun ölü olmasının səbəbi budur.

Leksik astananı 0.17-yə endirsək bu chunk keçir — amma eyni anda
`dev_out_of_corpus_ceo` da chunk qəbul etməyə başlayır (0 → 1, 0.12-də 3).
Yəni «zəif embedding-li DOĞRU sənəd» ilə «korpusda olmayan sual» eyni bal
zolağındadır. Onları ayıran qapı dəyəri yoxdur.

## Qərar: pullu dövr İŞƏ SALINMADI

Sweep ~$0.001-ə hipotezin ilk şərtini rədd etdi. Namizədləri tam dev run-larına
(~$0.17) vermək mövcud sübut qarşısında əsassız olardı: hər namizəd bir
multi-hop case qazanıb anti-hallüsinasiya qapısını itirir.

**Dürüst məhdudiyyət:** sweep retrieval ÖRTÜYÜNÜ ölçür, uçdan-uca pass-rate-i
yox. Qəbul edilmiş chunk hələ hallüsinasiya demək deyil — SUT-da ikinci qat
(grounding) da var. Yəni sweep zəruri şərtin pozulduğunu göstərir, kifayət
şərti barədə danışmır. Uçdan-uca təsdiq istənilsə, 0.30 namizədi ilə bir dev
run ($0.042) sızmanın həqiqətən hallüsinasiyaya çevrilib-çevrilmədiyini
göstərərdi.

## Bundan sonra hara baxmaq lazımdır

Rəqəmlər problemi retrieval parametrindən **kənara** yönəldir:

1. **Chunking.** Lazımi fakt 0.335 bal alan chunk-ın içindədir. `CHUNK_SIZE` /
   `CHUNK_OVERLAP` dəyişikliyi onu daha fokuslu chunk-a sala bilər — bu, dense
   balı qapı dəyişmədən qaldırar. Qiyməti: indeksin yenidən qurulması
   (embedding xərci), ona görə bu ayrıca dövrdür.
2. **Sorğu genişləndirmə (multi-hop üçün).** Sual iki fakt istəyir; tək
   embedding hər ikisini eyni dərəcədə təmsil etmir. Alt-sorğulara bölmək
   retrieval qatının dəyişikliyidir, parametrin yox.
3. **Embedding modeli.** 0.335 vs 0.42 fərqi modelin təmsil gücündədir.

Üçü də parametr deyil, dəyişiklikdir — hər biri öz ölçmə dövrünü tələb edir.

## Artefaktlar

- `logs/retrieval_sweep.json` — hər namizədin hər case üzrə xam nəticəsi
- `tools/retrieval_sweep.py` — alət; təkrar istehsal:

```bash
python tools/retrieval_sweep.py --top-k 4 6 8 --threshold 0.42 0.38 0.34 0.30
python tools/retrieval_sweep.py --lexical 0.35 0.25 0.17 0.12 --margin 0.14
```

---

# ⟶ RETRAKSİYA (2026-08-12): yuxarıdakı cədvəlin sübutu natamamdır

Yuxarıdakı mətn **silinmir** — README qaydası açıqdır: *«Ölçmə qeydi geriyə
dönük redaktə olunmur»*. Geri götürülən hissə burada, sərhədi ilə birlikdə
göstərilir.

## Nə pozulub

Sweep üç dəfə ardıcıl işlədilib və hər üçü **eyni** default `--out` faylına
(`logs/retrieval_sweep.json`) yazıb. Sonuncu icra əvvəlkiləri əvəzləyib.
Diskdə qalan artefaktda cəmi **4 namizəd** var — hamısı
`top_k=4, astana=0.42, marja=0.14`, yalnız leksik astana dəyişir.

Nəticədə «Nəticə: dörd oxun heç biri işləmir» cədvəlinin dörd sətrindən
**yalnız `LEXICAL_THRESHOLD` sətri** artefaktla dəstəklənir:

| ox | artefaktda varmı |
|---|---|
| `TOP_K` (4, 6, 8) | **YOX** — JSON-da `top_k ≠ 4` namizəd yoxdur |
| `RELEVANCE_THRESHOLD` (0.42 → 0.30) | **YOX** — `astana ≠ 0.42` namizəd yoxdur |
| `SOFT_FLOOR_MARGIN` (0.10 → 0.26) | **YOX** — sənəddəki iki əmrin heç biri marjanı sweep etmir |
| `LEXICAL_THRESHOLD` | bəli |

«Təkrar istehsal» blokundakı iki əmr də geri götürülür: onlar sənəddəki
cədvəli **istehsal etmir**. Faktiki olaraq üç əmr işlədilib, sənəddə isə iki
yazılıb, üstəlik ikincisi `--margin 0.14` pinləyir.

Bundan başqa, `dev_multihop_mtls_cache` chunk cədvəlinin üstündəki «yumşaq
hədd 0.32» **səhvdir**: artefakt marja 0.14 ilə, yəni yumşaq hədd **0.28**
ilə istehsal olunub. Chunk balları (0.457 / 0.411 / 0.335 / 0.320) artefaktla
üst-üstə düşür, yalnız qapı dəyəri səhv yazılıb.

## Nə pozulmayıb

Səhv qapı dəyəri tapıntını **dəyişmir** və bunu açıq demək lazımdır: yumşaq
hədd xilası `leksik ≥ 0.35` tələb edir, həmin chunk-ın leksik balı isə
**0.173**-dür. Yəni xilas nə 0.28-də, nə 0.32-də işə düşür — marja oxunun
ölü olmasının səbəbi elə budur.

## Düzəliş

`tools/retrieval_sweep.py` artıq sabit fayla yazmır: hər icra öz
`logs/probes/<probe_id>/` qovluğunu alır, mövcud qovluğun üstünə yazmır,
argv-ni və vaxtı manifestə qeyd edir, cədvəli isə `summary.md`-də özü
generasiya edir. `tests/test_logs_iddialari.py` sənəddəki hər cədvəlin
artefaktda mövcudluğunu yoxlayır.

Köhnə `logs/retrieval_sweep.json` **silinmir və dəyişdirilmir** — o, nəyin
həqiqətən işlədildiyinin qeydidir. Əhatəsi: yalnız leksik ox,
`top_k=4, astana=0.42, marja=0.14`.

Yenidən ölçmənin nəticəsi aşağıdakı bölmədədir.

---

# YENİDƏN ÖLÇÜLDÜ (2026-08-12) — nəticə TƏSDİQLƏNDİ, indi sübutu var

Dörd ox **bir çağırışda** ölçüldü ki, sətirlər bir-biri ilə müqayisə oluna
bilsin: 3 × 3 × 3 × 2 = **54 namizəd**, hamısı tək artefaktda.

<!-- artefakt: 20260812T111026Z-sweep-top_k+threshold+soft_floor_margin+lexical_threshold -->

| TOP_K | astana | marja | leksik | örtük | sızma |
|---|---|---|---|---|---|
| 4 | 0.42 | 0.10 | 0.35 | 7/8 | 1/2 |
| 6 | 0.42 | 0.10 | 0.35 | 7/8 | 1/2 |
| 8 | 0.42 | 0.10 | 0.35 | 7/8 | 1/2 |
| 4 | 0.42 | 0.18 | 0.35 | 7/8 | 1/2 |
| 4 | 0.42 | 0.26 | 0.35 | 7/8 | 1/2 |
| 4 | 0.34 | 0.10 | 0.35 | 7/8 | 2/2 |
| 4 | 0.30 | 0.10 | 0.35 | 8/8 | 2/2 |
| 4 | 0.42 | 0.10 | 0.17 | 8/8 | 2/2 |

<!-- /artefakt -->

Yuxarıdakı sətirlər baseline qapısını saxlayıb hər dəfə **bir** oxu dəyişir.
Tam 54 sətirlik cədvəl `logs/probes/20260812T111026Z-sweep-top_k+threshold+soft_floor_margin+lexical_threshold/summary.md`-dədir.

## Nəticə əvvəlkindən daha güclüdür

**54 namizəddən 36-sı örtüyü tamamlayır (8/8). Onların SIFIRI sızmanı
baseline səviyyəsində (1/2) saxlayır.** Əvvəlki dörd sətirlik cədvəl bunu
iddia edirdi; indi 54 namizəd üzərində ölçülüb və artefaktda yoxlana bilər.

Ox üzrə:

- **TOP_K** — 4, 6, 8 üçün nəticə hərfi-hərfinə eynidir. Lazımi chunk onsuz
  da çəkilir; problem sıralamada deyil, qapıdadır.
- **SOFT_FLOOR_MARGIN** — 0.10 / 0.18 / 0.26 üçün nəticə eynidir. Yumşaq hədd
  xilası leksik ≥ 0.35 tələb edir, həmin chunk-ın leksik balı isə 0.173-dür;
  marja nə qədər böyüsə də xilas yolu açılmır.
- **RELEVANCE_THRESHOLD** — 0.34-də örtük hələ 7/8-dir, amma sızma artıq 2/2
  olur; 0.30-da örtük tamamlanır, sızma yenə 2/2.
- **LEXICAL_THRESHOLD** — 0.17-də örtük tamamlanır, sızma 2/2.

Yəni hipotez yenə rədd edilir, amma bu dəfə sənəddəki hər sətir artefaktda
var. Çıxarılan nəticə dəyişmir: chunking istiqaməti doğru seçilib.
