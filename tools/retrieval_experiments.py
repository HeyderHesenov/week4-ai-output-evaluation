"""Chunking / embedding modeli / sorğu genişləndirmə — ölçmə dövrü.

`tools/retrieval_sweep.py` PARAMETR oxlarını sınadı və hamısını rədd etdi
(`logs/retrieval_sweep.md`). Bu alət növbəti üç DƏYİŞİKLİYİ eyni metrika
ilə ölçür — hər üçü indeksin və ya sorğunun özünü dəyişir, qapı parametrini
yox.

İNDEKS TƏCRİDİ — BU FAYLIN ƏN VACİB HİSSƏSİ
--------------------------------------------
Hər eksperiment ÖZ `persist_dir`-inə yazır. Mövcud `storage/chroma`
2026-08-10 baseline run-larının ölçüldüyü indeksdir; onun üstünə yazmaq
həmin run-ları təkrar istehsal olunmaz edərdi. `--workdir` paylaşılan
qovluğun içində olsa alət imtina edir.

METRİKA — `retrieval_sweep.py` ilə EYNİ
----------------------------------------
örtük : `gold_sources` sənədlərinin hamısı qəbul edilmiş chunk-lar arasındadırmı
sızma : korpusda cavabı olmayan suala chunk qəbul edilirmi

Qapı parametrləri baseline-da SABİT saxlanılır (0.42 / 0.10 / 0.35), çünki
burada ölçülən indeksin (və ya sorğunun) keyfiyyətidir.

BÜDCƏ BƏRABƏRLİYİ (2026-08-12 düzəlişi)
----------------------------------------
Qapını sabit saxlamaq kifayət etmirmiş. İlk versiyada genişləndirmə sətirləri
baseline-dan 2.25 dəfə ÇOX chunk çəkirdi (108 vs 48), çünki `sub_queries`
tam sualı həmişə əlavə edirdi. Yəni «genişləndirmə 8/8» nəticəsi əlavə
büdcə ilə alınmışdı və müqayisə ədalətli deyildi. İndi büdcə hər sətirdə
RƏQƏMLƏ yazılır və uyğunsuzluq alətin çıxış kodunu dəyişir — sənəddəki
intizam yazılı idi, amma heç nə onu hesablamırdı.

DİQQƏT — YALNIZ DEV.

ÇIXIŞ KODLARI
  0  uğurlu
  1  konfiqurasiya xətası
  2  --workdir paylaşılan indeksin içindədir
  3  ən azı bir sətrin retrieval büdcəsi baseline ilə uyğun deyil
  4  mövcud indeks gözlənilən parametrlərlə qurulmayıb
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# `python tools/...py` çağırışında Python `tools/`-u sys.path-a özü qoyur,
# `import tools.retrieval_experiments` (testlər, CI) isə qoymur.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_common import (  # noqa: E402
    PROBES_DIRNAME,
    PROJECT_ROOT,
    add_sut_to_path,
    cedvel,
    probe_yarat,
    summary_metni,
)

from eval.config import Settings, yoxla_retrieval  # noqa: E402
from eval.dataset import load_dataset  # noqa: E402
from eval.errors import ConfigError  # noqa: E402
from eval.sut import index_fingerprint  # noqa: E402

# Qapı — bütün eksperimentlərdə eyni qalır (SUT defaultları).
GATE = {"relevance_threshold": 0.42, "soft_floor_margin": 0.10, "lexical_threshold": 0.35}

# GATE sabiti də yoxlanır: gələcək redaktə pis dəyəri səssizcə gətirə bilməsin.
# Açar adları SUT sahələridir (`relevance_threshold`), validator isə env
# adlarını işlədir (`threshold`) — ona görə açıq şəkildə uyğunlaşdırılır.
yoxla_retrieval(
    threshold=GATE["relevance_threshold"],
    soft_floor_margin=GATE["soft_floor_margin"],
    lexical_threshold=GATE["lexical_threshold"],
)

BAGLAYICILAR = (" və ", " həmçinin ", " eləcə də ")
MIN_SOZ = 3
SUAL_MARKERLERI = frozenset(
    {
        "nə", "nədir", "nədən", "nəyə", "nələr", "neçə", "necə", "hansı",
        "hansıları", "harada", "haradan", "hara", "kim", "kimin", "kimə",
        "niyə", "qədər", "qədərdir", "vaxt", "vaxtdır",
    }
)
SUAL_SONLUQLARI = ("mi", "mı", "mu", "mü", "midir", "mıdır", "mudur", "müdür")


@dataclass(frozen=True)
class Experiment:
    name: str
    chunk_size: int = 800
    chunk_overlap: int = 200
    embedding_model: str = ""       # boş = framework defaultu
    expand_queries: bool = False
    # "matched": k sorğular arasında bölünür, büdcə baseline-dan böyük ola
    # bilmir. "free": hər sorğu tam k çəkir — ayrı və AÇIQ etiketlənmiş
    # eksperiment, baseline ilə müqayisə edilə bilməz.
    budget_mode: str = "matched"

    @property
    def slug(self) -> str:
        return self.name.replace(" ", "_").replace("/", "-").replace("+", "ve")


# ---------------------------------------------------------------------------
# Sorğu genişləndirmə
# ---------------------------------------------------------------------------


def _sualdir(parca: str) -> bool:
    """Parça MÜSTƏQİL sual kimi oxunurmu.

    Bu şərt olmasa « və » İKİ İSMİ birləşdirəndə bölgü birinci ismi SİLİR:
    «Backup və restore prosedurları nədir?» → «restore prosedurları nədir».
    Nəticə səs-küy deyil — retrieval bu qırıntıya İNAMLA səhv sənəd
    qaytarır, yəni ölçmə pisləşməni gizlədir.
    """
    sozler = parca.split()
    if len(sozler) < MIN_SOZ:
        return False
    kicik = [s.strip("?,.:;»«").lower() for s in sozler]
    if any(s in SUAL_MARKERLERI for s in kicik):
        return True
    return any(s.endswith(SUAL_SONLUQLARI) for s in kicik)


def _bol(sual: str) -> list[str]:
    """Bağlayıcılardan bölür; hər iki tərəf sual deyilsə BOŞ siyahı."""
    for sep in BAGLAYICILAR:
        if sep not in sual:
            continue
        parcalar = [p.strip(" ?,.") for p in sual.split(sep)]
        if len(parcalar) >= 2 and all(_sualdir(p) for p in parcalar):
            return parcalar
    return []


def sub_queries(question: str) -> list[str]:
    """Alt-sorğular — DETERMİNİSTİK, model çağırmır.

    Niyə model yox: sual bölgüsünü LLM-ə versək, ölçdüyümüz şey retrieval
    deyil, həmin LLM-in o gün necə böldüyü olar; nəticə təkrarlanmaz və
    xərc gətirər.

    ÜÇ QAYDA, hər biri ölçülmüş bir səhvin cavabıdır:

    1. Bölgü BAŞ TUTMASA nəticə TAM SUALDIR — bir sorğu, artıq yox.
       Əvvəlki versiya `?`-siz nüsxəni ayrı sorğu kimi saxlayırdı (parçalar
       strip edilir, müqayisə isə strip edilməmiş sualla aparılırdı), ona
       görə 12 dev sualının 9-unda bağlayıcı olmadığı halda hər biri 2
       sorğu çəkirdi.
    2. Bölgü YALNIZ hər iki tərəf müstəqil sual kimi oxunanda qəbul olunur.
    3. Tam sual HƏMİŞƏ birinci sorğudur — genişləndirmə əlavədir, əvəzləmə
       deyil; bölgü səhv olsa da tam sualın nəticəsi itmir.
    """
    parcalar = _bol(question)
    if not parcalar:
        return [question]
    return [question, *parcalar]


# ---------------------------------------------------------------------------
# İndeks
# ---------------------------------------------------------------------------


# BARMAQ İZİ `eval.sut`-DAN GƏLİR, KOPYALANMIR.
#
# Əvvəllər eyni düstur burada ikinci dəfə yazılmışdı və «probe artefaktı ilə
# run manifesti birləşdirilə bilir» iddiasını yalnız bir ŞƏRH qoruyurdu: iki
# nüsxədən biri dəyişsə, birləşdirmə səssizcə yalan olardı. `eval.sut` heç bir
# rag/langchain import-u gətirmir, ona görə bu import şəbəkəsizliyi pozmur.


@dataclass(frozen=True)
class IndexInfo:
    chunk_count: int
    sha256: str
    source: str          # "quruldu" | "mövcud"


def sut_settings_for(settings: Settings, exp: Experiment, persist_dir: Path):
    add_sut_to_path(settings)
    from rag.config import Settings as SutSettings

    # `persist_dir` AÇIQ override kimi gedir. Əvvəlki versiya
    # `os.environ["PERSIST_DIR"]`-i qlobal dəyişirdi və heç vaxt bərpa
    # etmirdi — bir dəfəlik skript üçün zərərsiz, amma test üçün düşmən.
    return SutSettings.load(
        openai_api_key=settings.require_openai_key(),
        embedding_model=exp.embedding_model or settings.embedding_model,
        chunk_size=exp.chunk_size,
        chunk_overlap=exp.chunk_overlap,
        persist_dir=persist_dir,
        top_k=4,
        **GATE,
    )


def ensure_index(settings: Settings, sut_settings, *, store=None) -> IndexInfo:
    """İndeksi qurur — və hazır sayılanın DOĞRU indeks olduğunu yoxlayır.

    Əvvəlki versiya `if existing > 0: return existing` deyirdi: yarımçıq
    kəsilmiş ingest-dən qalan natamam kolleksiya növbəti işlətmədə «hazır»
    sayılır və hesabat ONUN üzərində yazılırdı — hansı parametrlərlə
    qurulduğunu göstərən heç bir qeyd olmadan.

    İndi gözlənilən chunk-lar YERLİ hesablanır (`chunk_documents` şəbəkəyə
    çıxmır, pulsuzdur) və ID dəstləri müqayisə edilir.
    """
    from rag.ingest import chunk_documents, load_documents

    if store is None:
        # İMPORT MƏHZ BURADA: `store` inyeksiya olunanda `rag.store` (və onunla
        # birlikdə chroma) heç yüklənmir, yəni funksiya şəbəkəsiz test edilə bilir.
        from rag.store import VectorStore

        store = VectorStore(sut_settings)
    documents = load_documents(settings.sut_path / "data")
    chunks = chunk_documents(documents, sut_settings)
    gozlenilen = {c.metadata["chunk_id"] for c in chunks}

    # YALNIZ İCTİMAİ API. `count()` və `existing_ids()` `VectorStore`-un öz
    # metodlarıdır. Kolleksiyanın daxilinə (`_store`) uzanmaq aləti SUT-un
    # daxili quruluşuna bağlayardı — və pin edilmiş commit dəyişəndə bu,
    # səssiz sıfır kimi görünərdi, xəta kimi yox.
    umumi = int(store.count())
    var_olan = set(store.existing_ids(sorted(gozlenilen))) if gozlenilen else set()

    if umumi:
        if var_olan == gozlenilen and umumi == len(gozlenilen):
            return IndexInfo(umumi, index_fingerprint(gozlenilen), "mövcud")
        if umumi > len(var_olan):
            # Kolleksiyada gözlənilməyən chunk var: ya başqa chunking, ya
            # başqa korpus. Onun üzərində ölçmək hansı indeksin ölçüldüyünü
            # bilinməz edərdi.
            raise ConfigError(
                f"{sut_settings.persist_dir} başqa parametrlərlə qurulub: "
                f"kolleksiyada {umumi} chunk var, gözlənilən {len(gozlenilen)}, "
                f"uyğun gələn {len(var_olan)}.\n"
                "Ya qovluğu silin, ya başqa --workdir verin."
            )
        # Yalnız əskik var → kəsilmiş ingest; aşağıda tamamlanır.
    store.add(chunks)
    return IndexInfo(int(store.count()), index_fingerprint(gozlenilen), "quruldu")


# ---------------------------------------------------------------------------
# Ölçmə
# ---------------------------------------------------------------------------


def matched_sorgular(sorgular: Sequence[str], *, top_k: int) -> tuple[list[str], int]:
    """`matched` rejim üçün sorğu siyahısını büdcəyə SIĞDIRAR.

    NİYƏ KƏSMƏ LAZIMDIR: `k = max(1, top_k // len(sorgular))` tək başına
    invariantı SAXLAMIR. `k` bir-dən aşağı düşə bilmir, ona görə sorğu sayı
    `top_k`-nı keçəndə cəm da keçir: `top_k=4`, 5 alt-sorğu → `k=1`, büdcə 5.
    Nəticədə «matched» adlanan sətir `✗ BÜDCƏ ARTIQ` damğası alırdı — yəni ad
    ilə davranış ziddiyyətdə idi. (Cari dev korpusunda baş vermir: yalnız 2
    sual bölünür, hərəsi 3-ə. Çap olunmuş heç bir rəqəm bundan asılı deyil.)

    TAM SUAL HƏMİŞƏ QALIR — `sub_queries`-in 3-cü qaydası. Kəsmə sondan
    aparılır, çünki bölgü nə qədər dərinə getsə, parça bir o qədər dar olur.

    Atılanların sayı QAYTARILIR: səssiz kəsmə «5 alt-sorğu ölçüldü» təəssüratı
    yaradardı, halbuki yuxarıdakı nümunədə 4-ü ölçülüb.
    """
    if len(sorgular) <= top_k:
        return list(sorgular), 0
    return list(sorgular[:top_k]), len(sorgular) - top_k


def evaluate(pipeline, question: str, gold: set[str], *, expand: bool, budget_mode: str) -> dict:
    sorgular = sub_queries(question) if expand else [question]
    tam_k = pipeline.settings.top_k
    atilan = 0
    if budget_mode == "matched":
        # Büdcə sorğular arasında BÖLÜNÜR və siyahı büdcəyə sığdırılır:
        # cəm baseline-dan böyük ola bilmir.
        sorgular, atilan = matched_sorgular(sorgular, top_k=tam_k)
        k = max(1, tam_k // len(sorgular))
    else:
        k = tam_k

    accepted, retrieved_total, seen = [], 0, set()
    for q in sorgular:
        for chunk in pipeline._retrieve(q, k=k):
            retrieved_total += 1
            if not pipeline._accepts(chunk)[0]:
                continue
            key = str(getattr(chunk, "chunk_id", id(chunk)))
            if key not in seen:
                seen.add(key)
                accepted.append(chunk)
    got = {Path(str(getattr(c, "source", ""))).name for c in accepted}
    return {
        "queries": len(sorgular),
        "dropped_subqueries": atilan,
        "k_per_query": k,
        "retrieval_budget": len(sorgular) * k,
        "retrieved": retrieved_total,
        "accepted": len(accepted),
        "covered": not (gold - got),
        "missing": sorted(gold - got),
    }


def run(settings: Settings, exp: Experiment, cases, workdir: Path) -> dict:
    add_sut_to_path(settings)
    from rag.pipeline import RagPipeline
    from rag.store import VectorStore

    sut_settings = sut_settings_for(settings, exp, workdir / exp.slug)
    # BİR STORE, İKİ İSTİFADƏ. Əvvəllər `ensure_index` özü bir `VectorStore`
    # qurur, sonra `RagPipeline` üçün EYNİ `persist_dir` üzərində ikincisi
    # yaradılırdı — 9 eksperimentə 18 chroma klienti. Bəzi langchain-chroma
    # versiyaları bir yol üçün ziddiyyətli klient konfiqurasiyasında istisna
    # qaldırır, və iki klient eyni kolleksiyanı fərqli anlarda görə bilər.
    store = VectorStore(sut_settings)
    index = ensure_index(settings, sut_settings, store=store)
    pipeline = RagPipeline(sut_settings, store=store, llm=object())

    per_case, covered, total, leaked, leak_total, budce = {}, 0, 0, 0, 0, 0
    for case in cases:
        gold = {Path(s).name for s in (case.gold_sources or [])}
        row = evaluate(
            pipeline, case.question, gold,
            expand=exp.expand_queries, budget_mode=exp.budget_mode,
        )
        per_case[case.id] = row
        budce += row["retrieval_budget"]
        if gold:
            total += 1
            covered += bool(row["covered"])
        elif case.category == "out_of_corpus":
            leak_total += 1
            leaked += bool(row["accepted"])

    return {
        "name": exp.name,
        "chunk_size": exp.chunk_size,
        "chunk_overlap": exp.chunk_overlap,
        "embedding_model": exp.embedding_model or settings.embedding_model,
        "expand_queries": exp.expand_queries,
        "budget_mode": exp.budget_mode,
        "index_chunks": index.chunk_count,
        "index_sha256": index.sha256,
        "index_source": index.source,
        "retrieval_budget": budce,
        "covered": covered,
        "total": total,
        "leaked": leaked,
        "leak_total": leak_total,
        "cases": per_case,
    }


EXPERIMENTS = [
    Experiment("baseline (800/200)"),
    Experiment("chunking 500/150", chunk_size=500, chunk_overlap=150),
    Experiment("chunking 400/120", chunk_size=400, chunk_overlap=120),
    Experiment("chunking 300/90", chunk_size=300, chunk_overlap=90),
    Experiment("chunking 250/60", chunk_size=250, chunk_overlap=60),
    Experiment("embedding 3-large", embedding_model="text-embedding-3-large"),
    Experiment(
        "3-large + chunking 300/90",
        chunk_size=300, chunk_overlap=90,
        embedding_model="text-embedding-3-large",
    ),
    Experiment("genişləndirmə (büdcə bərabər)", expand_queries=True),
    Experiment("genişləndirmə (büdcə sərbəst)", expand_queries=True, budget_mode="free"),
]


def swept_axes(experiments: Sequence[Experiment] = EXPERIMENTS) -> list[str]:
    """Dəstin FAKTİKİ olaraq tərpətdiyi oxlar — `probe_id`-yə düşür.

    Əvvəllər siyahı `probe_id` çağırışında sabit yazılmışdı, ona görə
    `EXPERIMENTS`-dən bir ox çıxarılsa belə qovluq adı onu iddia etməyə davam
    edərdi — yəni artefaktın adı run-un etmədiyi ölçməni vəd edərdi. Sweep
    aləti bunu düzgün edir (`tools/retrieval_sweep.py:swept_axes`); burada da
    eyni qayda.

    Baseline (heç nə dəyişməyən sətir) ox sayılmır: o, müqayisə nöqtəsidir.
    """
    esas = Experiment("baseline")
    oxlar = []
    if any(
        e.chunk_size != esas.chunk_size or e.chunk_overlap != esas.chunk_overlap
        for e in experiments
    ):
        oxlar.append("chunking")
    if any(e.embedding_model for e in experiments):
        oxlar.append("embedding")
    if any(e.expand_queries for e in experiments):
        oxlar.append("genişləndirmə")
    return oxlar


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--workdir", type=Path, required=True,
        help="İndekslərin qurulacağı TƏCRİD qovluğu (paylaşılan storage/ OLMAMALIDIR)",
    )
    ap.add_argument("--probes-dir", type=Path, default=None)
    ap.add_argument(
        "--allow-unmatched-budget", action="store_true",
        help="Büdcəsi uyğun olmayan sətir üçün sıfırdan fərqli kod qaytarma.",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)

    settings = Settings.load()
    shared = (settings.sut_path / "storage").resolve()
    workdir = args.workdir.resolve()
    if shared == workdir or shared in workdir.parents:
        print(
            f"İMTİNA: --workdir paylaşılan indeksin içindədir ({shared}).\n"
            "Baseline indeksinin üstünə yazmaq 2026-08-10 run-larını təkrar "
            "istehsal olunmaz edərdi.",
            file=sys.stderr,
        )
        return 2

    dataset = load_dataset(settings.dataset_path)
    cases = [c for c in dataset.cases if c.split == "dev"]
    args.workdir.mkdir(parents=True, exist_ok=True)

    probes_dir = args.probes_dir or (settings.logs_dir / PROBES_DIRNAME)
    writer, kimlik = probe_yarat(
        alet="eksperiment",
        oxlar=swept_axes(),
        argv=["python", "tools/retrieval_experiments.py", *argv],
        settings=settings,
        dataset=dataset,
        probes_dir=probes_dir,
    )

    # ÖLÇMƏ ARTEFAKT MÜQAVİLƏSİ İÇİNDƏ APARILIR.
    #
    # Qovluq artıq açılıb və `status: yarımçıq` manifesti diskdədir. Hansı
    # yolla çıxsaq da (uğur, `ConfigError`, chroma xətası, Ctrl-C) manifest
    # vəziyyəti DEYİR — səssiz yarımçıq qovluq qalmır.
    try:
        return _olc(writer, kimlik, settings, cases, args)
    except ConfigError as exc:
        writer.mark_failed(f"ConfigError: {exc}")
        print(f"İndeks xətası: {exc}", file=sys.stderr)
        return 4
    except BaseException as exc:  # noqa: BLE001 — KeyboardInterrupt da daxil
        writer.mark_failed(type(exc).__name__)
        raise


def _olc(writer, kimlik: dict, settings: Settings, cases, args) -> int:
    print(f"{len(cases)} dev case × {len(EXPERIMENTS)} eksperiment → {writer.paths.root}")
    print(f"qapı sabitdir: {GATE}\n")

    results = [run(settings, exp, cases, args.workdir) for exp in EXPERIMENTS]

    baseline_budce = results[0]["retrieval_budget"]
    setirler, uygunsuz = [], []
    for r in results:
        # BÜDCƏ FƏRQİNİN İSTİQAMƏTİ ƏHƏMİYYƏTLİDİR.
        #
        # Baseline-dan ÇOX büdcə nəticəni şübhəli edir: sətir üstünlüyü
        # dəyişiklikdən yox, əlavə chunk-dan almış ola bilər — D2 məhz bu idi.
        # Baseline-dan AZ büdcə isə əks istiqamətdədir: eyni örtüyü daha ucuz
        # almaq nəticəni ZƏİFLƏTMİR, gücləndirir. Onu «uyğunsuz» saymaq
        # düzgün ölçülmüş qazancı səhvən şübhə altına salardı.
        artiq = r["retrieval_budget"] > baseline_budce
        az = r["retrieval_budget"] < baseline_budce
        if artiq:
            uygunsuz.append(r["name"])
        isare = " ✗ BÜDCƏ ARTIQ" if artiq else (" ↓ büdcə az" if az else "")
        print(
            f"{r['name']:34s} chunk={r['index_chunks']:3d}  "
            f"örtük: {r['covered']}/{r['total']}   sızma: {r['leaked']}/{r['leak_total']}   "
            f"büdcə: {r['retrieval_budget']}{isare}"
        )
        for cid, row in r["cases"].items():
            if row["missing"]:
                print(f"    ✗ {cid:32s} çatışmır: {', '.join(row['missing'])}")
        writer.append_row(r)
        setirler.append(
            [
                r["name"],
                r["index_chunks"],
                f"{r['covered']}/{r['total']}",
                f"{r['leaked']}/{r['leak_total']}",
                f"{r['retrieval_budget']}{isare}",
            ]
        )

    govde = cedvel(
        ["eksperiment", "indeks chunk", "örtük", "sızma", "retrieval büdcəsi"], setirler
    )
    if uygunsuz:
        govde += (
            f"\n\n> ⚠️ **Büdcə ARTIQ** ({', '.join(uygunsuz)}): bu sətirlər "
            f"baseline-dan ({baseline_budce}) ÇOX retrieval büdcəsi ilə ölçülüb, "
            "ona görə üstünlükləri dəyişiklikdən deyil, əlavə chunk-dan gələ bilər "
            "və eyni cədvəldə müqayisə edilə bilməz."
        )
    govde += (
        f"\n\n> Baseline retrieval büdcəsi: **{baseline_budce}** çəkilən chunk. "
        "«↓ büdcə az» işarəsi problem deyil: eyni örtüyü daha ucuz almaq nəticəni "
        "gücləndirir."
    )

    writer.write_manifest(
        {**kimlik, "gate": GATE, "baseline_retrieval_budget": baseline_budce,
         "unmatched_budget": uygunsuz}
    )
    writer.write_summary(
        summary_metni(
            basliq="Chunking / embedding / sorğu genişləndirmə (dev)",
            probe_id_=writer.paths.probe_id,
            argv=kimlik["argv"],
            govde=govde,
        )
    )
    print(f"\nArtefakt: {writer.paths.root.relative_to(PROJECT_ROOT)}")
    if uygunsuz and not args.allow_unmatched_budget:
        print(f"\n⚠️ Büdcəsi uyğun olmayan sətir: {', '.join(uygunsuz)}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
