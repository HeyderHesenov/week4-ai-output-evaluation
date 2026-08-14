# Generasiya qatı — prompt qaydası ÖLÇÜLDÜ və rədd edildi (2026-08-14)

Retrieval dövrü bağlandıqdan sonra holdout-da qalan uğursuzluqlar
`generation_wrong` və `over_refusal` idi — hər ikisi generasiya qatında. Bu
sənəd həmin qatı ölçür. Alət: `tools/generation_probe.py`, xam nəticə:
`logs/probes/20260814T200909Z-generasiya-1-müqəddimə+2-qeyri-müəyyənlik+3-hər ikisi/`.

**Nəticə: hipotez rədd edildi.** Dörd qolun heç biri heç bir case-də davranışı
dəyişmədi (12 xanadan 12-si nəzarətlə eyni). Dövr **bağlanır**: möhür
toxunulmur, holdout toxunulmur (ledger **3**-də qalır).

## Niyə bu dövr

Qalan iki holdout uğursuzluğunun hər ikisində **düzgün fakt modelin
kontekstində idi** — problem çəkmə deyil, çəkilənlə nə edilməsi idi.
Dev-də eyni naxışın iki daşıyıcısı var və onlar bu probe-un mövzusudur.

## Hipotez: iki hərəkət, çatışmayan üçüncü

SUT-un yalnız iki hərəkəti var: **cavab ver** və ya **imtina et**. Üçüncüsünü
etmir — sualın strukturuna münasibət bildirmək (yanlış müqəddiməni düzəltmək,
qeyri-müəyyən sualın oxunuşlarını adlandırmaq). Mənbə system prompt-un 5 və
6-cı qaydalarıdır: onlar «kontekstdə yoxdursa imtina et» deyir, amma «sual
kontekstlə ziddiyyət təşkil edirsə nə et» sualını cavabsız qoyur.

Yoxlanan şey: bu boşluğu **əlavə qayda ilə** doldurmaq olurmu.

## Probe metodu

- **Retrieval işlədilmir.** Chunk-lar `20260812T094516Z-dev-baseline`
  run-ından oxunur, prompt SUT-un öz qurucuları ilə bərpa olunur (yenidən
  yazılmır). Beləliklə qapı davranışı nəticəyə qarışa bilmir və embedding
  çağırılmır.
- **Hakim çağırılmır.** Kappa 0.13 ölçülüb (`logs/judge_bias.md`), ona görə
  təsnifat tam determinist açar sözlərlə aparılır və meyarlar ölçmədən ƏVVƏL
  koda yazılıb.
- **Dörd qol × üç case × üç təkrar = 36 çağırış**, $0.015, `gpt-4o-mini`
  (saxlanmış run ilə eyni model — artefakt bunu qeyd edir).
- **Üç sadiqlik qapısı**, hər biri ölçmədən əvvəl: `RetrievedChunk` sahə
  dəsti, system prompt hash-ının run-dakı ilə eyniliyi, modelin eyniliyi.
  Nəzarət qolu orijinal davranışı təkrar istehsal etməsə alət `5` qaytarır
  və cədvəl etibarsız elan olunur. **Hər üçü keçdi.**

`dev_out_of_corpus_ceo` qəsdən **kənardadır**: saxlanmış artefaktda
`reason=low_relevance` və **sıfır LLM çağırışı** var — qapı onu modelə
çatmamış kəsir, ona görə heç bir prompt dəyişikliyi ona toxuna bilməz. Cədvələ
salmaq ölçülməmiş şeyi ölçülmüş kimi göstərərdi.

`dev_out_of_corpus_graphql` isə **reqressiya nəzarətidir**: qeyri-müəyyənlik
qaydası imtinanı sındırsa, qazanc reqressiya ilə gələrdi.

## Ölçmədən əvvəl aşkarlanan test qüsuru

`dev_false_premise_free_cache` karkasda **imtina ilə keçir**. Yoxlaması
mənfidir (`not_contains: 128`), boş və ya imtina cavabı isə onu avtomatik
təmin edir. Yəni mənfi yoxlama «düzgün ehtiyatlı»nı «düzgün düzəldici»dən
ayıra bilmir — case keçir, halbuki model doğru cavabı verməyib.

Case **redaktə OLUNMUR**: möhürlənib və öz rubrikasına görə düzgündür. Bu,
karkas haqqında tapıntıdır və probe-un meyarı buna görə daha dardır —
`muqeddime_duzeldildi` üçün doğru **planın adlandırılması** tələb olunur.

## Nəticə

<!-- artefakt: 20260814T200909Z-generasiya-1-müqəddimə+2-qeyri-müəyyənlik+3-hər ikisi -->
| case | 0-nəzarət | 1-müqəddimə | 2-qeyri-müəyyənlik | 3-hər ikisi |
|---|---|---|---|---|
| dev_false_premise_free_cache | imtina (3/3) | imtina (3/3) | imtina (3/3) | imtina (3/3) |
| dev_ambiguous_limit | tek_oxunus (3/3) | tek_oxunus (3/3) | tek_oxunus (3/3) | tek_oxunus (3/3) |
| dev_out_of_corpus_graphql | imtina (3/3) | imtina (3/3) | imtina (3/3) | imtina (3/3) |
<!-- /artefakt -->

## Oxunuş

**Cədvəl düz xətdir və bu, təsnifat artefaktı deyil** — xam cavablar
yoxlanıldı:

- `dev_false_premise_free_cache`: 12 çağırışın hamısı **bayt-bayt eyni**
  cavabı verdi — `Sənədlərdə bu suala cavab tapılmadı.` Qol 1 açıq şəkildə
  «bu halda imtina kifayət deyil» yazdığı halda da.
- `dev_ambiguous_limit`: qol 2 (məhz bu case üçün yazılmış qayda) **heç bir
  variasiya belə yaratmadı** — 3/3 `Limit 200-dür [1].` Qol 1 və 3 yalnız
  ifadəni dəyişdi (``Limit `limit` parametri üçün maksimum dəyər 200-dür
  [1].``), oxunuşları yenə adlandırmadı. Söz seçimi dəyişir, davranış
  dəyişmir.
- Reqressiya yoxdur: `graphql` imtinası hər dörd qolda qaldı.

**Ən güclü sətir budur: fakt kontekstdə İDİ.** `dev_false_premise_free_cache`
üçün ən yüksək ballı chunk (`atlas_infra_qeydleri.md`, dense 0.569) hərfən
belə yazır: *«İzolyasiya olunmuş kənar keş qovşağı `atlas-cache-edge` adlanır,
**yalnız Enterprise planında ayrılır**»*. Model həmin cümləni oxuyub imtina
etdi — üstəlik ona düzəltməyi əmr edən qayda ilə birlikdə. Eyni şəkildə
`dev_ambiguous_limit`-in kontekstində ən azı üç fərqli «limit» var
(səhifələmə 200 [1], xərc limiti 95 AZN [2], sürət limiti [3][4]), yəni
qeyri-müəyyənlik həqiqətən oradadır.

Yəni bu, «modelin məlumatı çatmadı» halı deyil. **Sonradan əlavə edilən qayda
əvvəlki qaydaları üstələmir.**

## Bunun karkas üçün mənası

`PromptVariant` yalnız iki şey edə bilir: system prompt-un **SONUNA** mətn
əlavə etmək (`system_suffix`) və few-shot növbələri qoşmaq. Qaydanı əvəz etmək
mexanizmi yoxdur. Probe məhz birinci mexanizmi ölçdü və onun bu davranış üçün
**tavanını** tapdı: 5/6-cı qaydalar prompt-un içindədir və sonda gələn 15/16-cı
qaydalar onları geri ala bilmir.

Ona görə nəticə «prompt işləmir» deyil, daha dardır və daha faydalıdır:
**əlavə edilən qayda mövcud imtina qaydasını üstələmir.**

## Növbəti addımın qapısı

Dövr **bağlanır**. Yeni dev case-i yazılmır, `seal-split --force` çağırılmır,
`dataset_sha256` dəyişmir, holdout işlədilmir.

Sınanmamış qalan və növbəti dövrə namizəd olan iki yol — hər ikisi bu
nəticənin göstərdiyi istiqamətdədir:

1. **Few-shot** (`PromptVariant.few_shot`) — qayda yazmaq əvəzinə düzəldici
   cavabın nümunəsini göstərmək. Karkasda artıq var, bu probe-da ölçülməyib.
2. **5/6-cı qaydaların özünün dəyişdirilməsi** — mexanizm tələb edir
   (`system_replace`), yəni əvvəlcə karkas işi, sonra ölçmə. SUT redaktə
   olunmadan yalnız variant qatında edilə bilər.

Ölçülmüş və rədd edilmiş yol təkrarlanmır: eyni formada daha uzun qayda mətni
yazmaq bu cədvələ görə gözləntisiz bir xərcdir.
