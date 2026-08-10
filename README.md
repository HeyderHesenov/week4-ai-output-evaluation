# Week 4 — AI çıxışının qiymətləndirilməsi

Week 2-də qurulmuş RAG sual-cavab sistemini **dəyişdirmədən** qiymətləndirən
framework: deterministik qraderlər, LLM-as-judge, dev/holdout intizamı,
kök-səbəb təsnifatı və xərc/gecikmə uçotu.

Qiymətləndirilən sistem `vendor/week2-rag-document-qa` submodule-udur və
`.env`-dəki `SUT_COMMIT` ilə pin edilib. Pin uyğun gəlmirsə run **imtina
edir** — «sistemi dəyişmədən qiymətləndirdik» iddiası beləliklə söz deyil,
yoxlanan invariantdır.

Metodologiyanın tam izahı: [`docs/methodology.md`](docs/methodology.md).

---

## Qısa başlanğıc

```bash
# 1. Submodule və asılılıqlar
git submodule update --init --recursive
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# 2. Açarlar
cp .env.example .env      # OPENAI_API_KEY və ANTHROPIC_API_KEY doldurun

# 3. Testlər (açar və şəbəkə TƏLƏB ETMİR)
pytest -q

# 4. Qiymətləndirilən sistemin asılılıqları + indeks
pip install "langchain>=0.3,<0.4" "langchain-community>=0.3,<0.4" \
            "langchain-openai>=0.2,<0.4" "langchain-chroma>=0.2,<0.3" \
            "langchain-text-splitters>=0.3,<0.4" "pypdf>=5.0"
(cd vendor/week2-rag-document-qa && python -m rag.cli ingest --path data/)

# 5. Bölgünü möhürlə və işlət
python -m eval.cli seal-split
python -m eval.cli run --split dev --variant baseline
python -m eval.cli report <run_id>
```

> Streamlit qəsdən quraşdırılmır — bu framework SUT-un veb interfeysini
> deyil, kitabxana səviyyəsindəki `RagPipeline`-ı işlədir.

---

## Əmrlər

| Əmr | Nə edir |
|---|---|
| `seal-split [--force]` | dev/holdout bölgüsünü möhürləyir (`data/split_manifest.json`) |
| `run --split dev\|holdout\|all --variant <id> [--repeats N]` | Qiymətləndirməni işlədir |
| `optimize --split dev --variant <id>` | Eyni kod, **yalnız dev** — guard holdout-u rədd edir |
| `reclassify <run_id>` | Taksonomiyanı yenidən hesablayır (şəbəkəsiz, pulsuz) |
| `report <run_id> [--compare <run_id>]` | Markdown hesabat yaradır |
| `judge-bias [run_id]` | Hakim ↔ insan razılığı (Cohen kappa). Arqumentsiz: mənbələr `human_labels.yaml`-dan |
| `list-runs` | Run-ları və holdout icralarının sayını göstərir |

**Exit kodları:** `0` uğur · `1` konfiqurasiya/istifadəçi xətası ·
`2` dev/holdout intizamının pozulması.

`2`-nin ayrıca kod olması CI üçündür: çirklənmiş qiymətləndirmə adi
xətadan fərqli hadisədir — adi xəta run-ı dayandırır, çirklənmə isə artıq
yazılmış nəticələrin hamısını şübhə altına alır.

---

## İnsan etiketləri və kappa (əl ilə görülən yeganə iş)

Hakimin qərarının insan qərarı ilə nə dərəcədə uzlaşdığını ölçmək üçün
`data/human_labels.yaml`-da **9 case** var. Onları **layihə müəllifi** yazır;
modelə yazdırmaq ölçünü mənasız edər — model öz-özü ilə razılaşar və çıxan
rəqəm «hakim insanla uzlaşır» kimi oxunar, halbuki heç bir insan iştirak
etməyib.

```bash
# 1. Faylı aç: hər case blokunun sonundakı sətirdən `# ` silib bal yaz
#      # dev_open_incident_s1: 0     →       dev_open_incident_s1: 2
#    Girinti (iki boşluq) qalmalıdır — açar `labels:` altında olmalıdır.
$EDITOR data/human_labels.yaml

# 2. Ölç (arqumentsiz — mənbə run-ları faylın `sources` blokundadır)
python -m eval.cli judge-bias

# 3. Çıxan kappa və n dəyərlərini logs/judge_bias.md-nin 3-cü bölməsinə köçür
```

**Şkala hakimlə eynidir** (0–3) — fərqli şkala kappanı mənasız edər; aralıqdan
kənar bal `EvalError` verir. **Əvvəlcə hakimin balına baxmayın:** görsəniz ona
uyğunlaşarsınız (anchoring) və kappa razılığı deyil, təsirlənməni ölçər.

Hər etiket öz `(run_id, repeat)` ünvanına bağlıdır: insan **bir case-i** deyil,
**bir cavab mətnini** qiymətləndirir. Bu ünvan olmasa təkrarlar n-i şişirdər və
etiket başqa run-ın fərqli cavabı ilə müqayisə oluna bilərdi.

---

## Memarlıq

```
                 ┌──────────────────────────────────────────┐
                 │  eval/cli.py — əmrlər, exit kodları      │
                 └───────────────────┬──────────────────────┘
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         │  eval/runner.py — mühafizələr → icra → artefaktlar     │
         └───┬───────────────┬───────────────┬───────────────┬───┘
             │               │               │               │
      ┌──────┴─────┐  ┌──────┴─────┐  ┌──────┴─────┐  ┌──────┴──────┐
      │  split.py  │  │   sut.py   │  │  judge.py  │  │ artifacts.py│
      │ çirklənmə  │  │ SUT adapt. │  │  Anthropic │  │   JSONL     │
      └────────────┘  └──────┬─────┘  └────────────┘  └─────────────┘
                             │
                      ┌──────┴────────┐
                      │ instrument.py │  ← SUT-un öz inyeksiya nöqtələri
                      └──────┬────────┘
                             │
              vendor/week2-rag-document-qa (PIN EDİLMİŞ, toxunulmaz)

      graders.py → rootcause.py → metrics.py → report.py
      (deterministik)  (qat)      (Wilson/McNemar/kappa)  (Markdown)
```

`observation.py` bütün qatların ortaq lüğətidir və **yalnız stdlib**
işlədir: qraderi sınamaq üçün nə LangChain, nə Chroma, nə şəbəkə lazımdır.

---

## Dizayn qərarları

Hər qərarın səbəbi müvafiq modulun docstring-indədir. Ən vacib beşi:

**1. Xam sübut və törəmə ayrılıb.** `SutObservation` sistemin nə
qaytardığıdır; qiymət, uğursuzluq sinfi və metriklər ondan **törəyir** və
heç vaxt onun içinə yazılmır. Bu ayrım `reclassify` əmrini mümkün edir:
taksonomiya dəyişəndə saxlanmış müşahidələr üzərində yenidən hesablamaq
kifayətdir — sistemi yenidən işlətmək lazım deyil.

**2. `contains: ["21"]` kifayət deyil.** «21» alt-sətri «2100 manat»
cavabında da var. Ədədi iddialar dəyər + vahid + dözümlülüklə ifadə olunur;
qrader rəqəmi parse edir və vahidin yaxınlıqda olmasını tələb edir.

**3. Normalizasiya SUT-dan import edilmir.** Week 2-də hazır folding
funksiyası var, amma onu işlətsək, SUT-un folding baqı həm cavabı üredərkən,
həm qiymətləndirərkən işləyər və baq **ölçülə bilməz**. Qiymətləndirici
ölçdüyü sistemdən müstəqil olmalıdır.

**4. İmtina heç vaxt 0 bal deyil.** Hakimin imtinası cavabın pis olduğunu
deyil, **ölçülə bilmədiyini** bildirir. `judge_error` ayrıca sayılır və heç
bir faizin məxrəcinə düşmür.

**5. Holdout registri kilid deyil.** Sui-istifadəni qeyri-mümkün etmək
əvəzinə **görünən** edirik: hesabat holdout icralarını tam çap edir və
yoxlayıcı öz hökmünü verir.

---

## Nə ölçülmür (açıq boşluqlar)

Dürüstlük tam olsun deyə bunlar hesabatda hər dəfə çap olunur:

- **Embedding çağırışları** — `VectorStore`-un içində baş verir, token sayı
  sarğıya görünmür. Sorğu uzunluğundan təxmin etmək olardı, amma uydurulmuş
  rəqəm hesabatı «tam» göstərərdi.
- **OpenAI qiymətləri** `verified: false` işarəlidir — canlı qiymət
  səhifəsinə qarşı yoxlanmayıb; hesabatda banner çıxır.
- **Qapının atdığı chunk** cavab verilmiş halda görünmür (SUT yalnız qəbul
  ediləni qaytarır) — belə hallar `retrieval_miss` kimi görünür.
- **`runs/` altındakı 4 mövcud manifest `sut_path`-i MÜTLƏQ yol kimi
  saxlayır.** Sonrakı run-lar onu repo-ya nisbi yazır (`config_hash` beləliklə
  maşından asılı olmur). Köhnə manifestlər QƏSDƏN redaktə edilməyib: `config_hash`
  məhz həmin config bloku üzərindən hesablanıb, yolu sonradan dəyişmək
  artefaktı öz-özü ilə uyğunsuz edərdi. Ölçmə qeydi geriyə dönük redaktə
  olunmur — bu, karkasın bütün mövqeyinin şərtidir.

---

## Testlər

```bash
pytest -q            # bütün suite; açar və şəbəkə TƏLƏB ETMİR
pytest tests/test_split.py -v    # dev/holdout intizamı
```

Bütün xarici sərhədlər inyeksiya olunur (`InstrumentedLLM(inner=...)`,
`AnthropicJudge(client=...)`, `RagSut(pipeline_factory=...)`), ona görə CI
lokal mühitlə eyni şəraitdə işləyir və «lokalda keçir, CI-də keçmir» sinfi
aradan qalxır.
