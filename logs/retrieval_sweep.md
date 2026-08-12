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
