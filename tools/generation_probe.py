"""Generation probe — kontekst düzgün olduqda prompt qaydası nə dəyişir.

NİYƏ AYRICA ALƏT
----------------
Sual generasiya qatı haqqındadır, ona görə retrieval SABİT saxlanmalıdır. Tam
run işlətmək onu hər dəfə yenidən icra edər və qapı davranışı nəticəyə
qarışar. Burada chunk-lar SAXLANMIŞ artefaktdan oxunur: retrieval
konstruksiyaya görə sabitdir, embedding çağırılmır, indeks açılmır.

Hakim də çağırılmır — 2026-08-12-də kappa 0.13 ölçüldü, yəni hakimdən keçən
hər rəqəm ehtiyat bayrağı daşıyır. Bu cədvəl tam determinist qalır.

NİYƏ SUT QURUCULARI İNYEKSİYA OLUNUR
------------------------------------
`rag.pipeline` modul səviyyəsində `rag.store`-u, o isə `langchain_chroma`-nı
import edir. CI bu paketləri quraşdırmır (yalnız anthropic/dotenv/PyYAML), ona
görə testlərdə həqiqi import «lokalda keçir, CI-də keçmir» sinfini yaradardı —
repo-nun CI şərhi məhz bu sinfin mövcud olmadığını iddia edir. Prompt
qurucularını burada TƏKRAR YAZMAQ isə daha pis olardı: onda ölçdüyümüz prompt
SUT-un prompt-u olmazdı.

Sadiqliyi qoruyan şey testlər deyil, İCRA VAXTI QAPISIDIR: `sut_qurucular()`
SUT-un `RetrievedChunk` sahə dəstini yoxlayır və uyğunsuzluqda ölçməni
dayandırır.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_common import (  # noqa: E402
    PROBES_DIRNAME,
    add_sut_to_path,
    artefakt_yolu,
    cedvel,
    probe_yarat,
    summary_metni,
)

from eval.artifacts import RunPaths, load_run  # noqa: E402
from eval.config import Settings  # noqa: E402
from eval.errors import EvalError  # noqa: E402

# Hash KARKASIN öz funksiyası ilə hesablanır. İkinci tərif yazmaq ölçməni
# səssizcə sındırırdı: probe `hexdigest()[:16]`, karkas isə tam 64 simvol
# yazırdı, ona görə nəzarət qapısı prompt heç dəyişməsə belə bağlanırdı.
from eval.instrument import sha256_text  # noqa: E402
from eval.observation import ChunkView  # noqa: E402


class ProbeError(EvalError):
    """Probe ölçməni davam etdirə bilmir."""


# SUT-un `RetrievedChunk` dataclass-ının GÖZLƏNİLƏN sahə dəsti.
SUT_CHUNK_SAHELERI = frozenset(
    {"label", "text", "source", "page", "chunk_index", "score", "lexical_score", "chunk_id"}
)


@dataclass(frozen=True)
class SutQurucular:
    """SUT-dan götürülən hər şey — bir yerdə, inyeksiya oluna bilən formada."""

    system_instruction: str
    build_user_message: Callable[[str, list, str], str]
    chunk_cls: type
    is_refusal: Callable[[str], bool]
    make_nonce: Callable[[], str]


def yoxla_saheler(chunk_cls: type) -> None:
    """Sahə dəsti gözlənildiyi kimidirmi — ÖLÇMƏ QAPISI.

    Test deyil, icra vaxtı yoxlaması: CI `rag.*`-ı import edə bilmədiyi üçün
    həqiqi sinif orada heç vaxt yoxlanmazdı. Uyğunsuzluqda ölçmə dayanır,
    çünki səssiz davam etmək «eyni kontekstdə ölçüldü» iddiasını yalan edərdi.
    """
    var_olan = {f.name for f in fields(chunk_cls)}
    if var_olan != SUT_CHUNK_SAHELERI:
        artiq = sorted(var_olan - SUT_CHUNK_SAHELERI)
        eskik = sorted(SUT_CHUNK_SAHELERI - var_olan)
        raise ProbeError(
            f"{chunk_cls.__name__} sahə dəsti gözlənildiyi kimi deyil "
            f"(artıq: {artiq or '—'}, əskik: {eskik or '—'}).\n"
            "SUT-un pin edilmiş commit-i dəyişib. Adapter köhnə dəstlə obyekt "
            "qursa, prompt səssizcə fərqlənər və cədvəl «eyni kontekstdə "
            "ölçüldü» iddiasını yalandan daşıyar."
        )


def sut_chunk(cv: ChunkView, *, q: SutQurucular) -> Any:
    """`ChunkView` → SUT-un `RetrievedChunk`-ı."""
    return q.chunk_cls(
        label=cv.label,
        text=cv.text,
        source=cv.source,
        page=cv.page,
        chunk_index=cv.chunk_index,
        score=cv.score,
        lexical_score=cv.lexical_score,
        chunk_id=cv.chunk_id,
    )


def mesajlar(
    question: str,
    chunks: Sequence[ChunkView],
    *,
    suffix: str,
    nonce: str,
    q: SutQurucular,
) -> list[dict]:
    """SUT-un `_answer`-inin qurduğu mesajların EYNİSİ (+ suffiks)."""
    system = q.system_instruction + (f"\n\n{suffix}" if suffix.strip() else "")
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": q.build_user_message(
                question, [sut_chunk(c, q=q) for c in chunks], nonce
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Təsnifat
# ---------------------------------------------------------------------------
#
# MEYARLAR QABAQCADAN SABİTDİR. Cədvələ baxdıqdan sonra meyar seçmək nəticəni
# istənilən istiqamətə çevirə bilər, ona görə onlar burada yazılır və
# artefaktda çap olunur.

# Korpus faktı (`atlas_infra_qeydleri.md`): `atlas-cache-edge` YALNIZ Enterprise
# planında ayrılır, limiti 128 GB. Müqəddiməni düzəltmək = doğru planı
# adlandırmaq; müqəddiməni qəbul etmək = Free üçün rəqəm vermək.
_DUZGUN_PLAN = "enterprise"
_YANLIS_REQEM = "128"

# Korpusdakı fərqli «limit» oxunuşları (`atlas_api_senedi.md` + infra qeydləri).
_LIMIT_NOVLERI = ("sürət", "səhifələmə", "yaddaş", "gündəlik", "eyni vaxtda")
_AYDINLASDIRMA = ("hansı limit", "hansı limiti", "dəqiqləşdir")


def _fold(text: str) -> str:
    """Azərbaycan «I/ı» problemi üçün SUT ilə eyni qaydada kiçildir."""
    return text.replace("I", "ı").replace("İ", "i").casefold()


@dataclass(frozen=True)
class ProbeCase:
    case_id: str
    nezaret_davranisi: str   # qol 0-dan GÖZLƏNİLƏN sinif (etibarlılıq qapısı)
    izah: str


PROBE_CASES = (
    ProbeCase(
        "dev_false_premise_free_cache", "imtina",
        "mənfi yoxlama ilə keçir, amma imtina ilə — düzəliş bacarığı ölçülməyib",
    ),
    ProbeCase(
        "dev_ambiguous_limit", "tek_oxunus",
        "«Limit 200-dür [1]» — səssizcə bir oxunuş seçilib",
    ),
    ProbeCase(
        "dev_out_of_corpus_graphql", "imtina",
        "REQRESSİYA NƏZARƏTİ — qol 2 imtinanı sındırmamalıdır",
    ),
)


def tesnif(case_id: str, text: str, *, q: SutQurucular) -> str:
    """Cavab mətni → davranış sinfi. Tam determinist, hakimsiz."""
    if q.is_refusal(text):
        return "imtina"
    alt = _fold(text)

    if case_id == "dev_false_premise_free_cache":
        if _DUZGUN_PLAN in alt:
            return "muqeddime_duzeldildi"
        if _YANLIS_REQEM in alt:
            return "muqeddime_qebul"
        return "diger"

    if case_id == "dev_ambiguous_limit":
        if any(a in alt for a in _AYDINLASDIRMA):
            return "oxunuslar_adlandi"
        if sum(1 for n in _LIMIT_NOVLERI if n in alt) >= 2:
            return "oxunuslar_adlandi"
        return "tek_oxunus"

    return "diger"


# ---------------------------------------------------------------------------
# Qollar və ölçmə
# ---------------------------------------------------------------------------
#
# Hər qol SUT-un 5/6-cı qaydalarının yaratdığı boşluğu bir istiqamətdən
# doldurur. Mətnlər variant `system_suffix`-i ilə EYNİ formadadır, ona görə
# qazanan qol birbaşa `data/variants/`-ə köçürülə bilir.

_MUQEDDIME = """ƏLAVƏ QAYDA — SUALIN MÜQƏDDİMƏSİ:
15. Sualın içindəki fərziyyə kontekstlə ZİDDİYYƏT təşkil edirsə (sual bir
    dəyəri, planı və ya şərti mövcud kimi göstərir, kontekst isə başqasını
    yazır), əvvəlcə düzgün faktı istinadla AÇIQ yaz və fərziyyəni düzəlt.
    Bu halda imtina kifayət deyil: kontekstdə cavab VAR, sadəcə sual onu
    səhv adlandırıb. 1 və 2-ci qaydalar qüvvədə qalır — düzəliş yalnız
    kontekstdə yazılana əsaslana bilər."""

_QEYRI_MUEYYEN = """ƏLAVƏ QAYDA — QEYRİ-MÜƏYYƏN SUAL:
16. Kontekst sualın BİRDƏN ÇOX fərqli oxunuşunu dəstəkləyirsə (eyni söz
    müxtəlif şeylərə aiddir və hər birinin öz dəyəri var), nə səssizcə
    birini seç, nə də imtina et: oxunuşları adlandır və mümkünsə hər biri
    üçün dəyəri istinadla yaz. Hansının nəzərdə tutulduğunu soruşmaq da
    qəbul olunandır."""


@dataclass(frozen=True)
class Qol:
    ad: str
    system_suffix: str


QOLLAR = (
    Qol("0-nəzarət", ""),
    Qol("1-müqəddimə", _MUQEDDIME),
    Qol("2-qeyri-müəyyənlik", _QEYRI_MUEYYEN),
    Qol("3-hər ikisi", f"{_MUQEDDIME}\n\n{_QEYRI_MUEYYEN}"),
)

NEZARET_QOLU = QOLLAR[0].ad


def model_adi(llm: Any) -> str:
    """LLM obyektinin model adı — `InstrumentedLLM` ilə EYNİ qayda ilə."""
    return str(
        getattr(llm, "model_name", None) or getattr(llm, "model", None) or "naməlum"
    )


def olc(
    observations: dict[str, dict], *, llm, q: SutQurucular, tekrar: int = 3
) -> list[dict]:
    """Hər case × hər qol × `tekrar` — bir sətir hər çağırışa."""
    model = model_adi(llm)
    setirler: list[dict] = []
    for case in PROBE_CASES:
        obs = observations.get(case.case_id)
        if obs is None:
            continue
        for qol in QOLLAR:
            for n in range(1, tekrar + 1):
                msgs = mesajlar(
                    obs["question"], obs["chunks"],
                    suffix=qol.system_suffix, nonce=q.make_nonce(), q=q,
                )
                cavab = str(llm.invoke(msgs).content).strip()
                setirler.append({
                    "case_id": case.case_id,
                    "qol": qol.ad,
                    "tekrar": n,
                    "sinif": tesnif(case.case_id, cavab, q=q),
                    "cavab": cavab,
                    "model": model,
                    "system_sha256": sha256_text(msgs[0]["content"]),
                })
    return setirler


def nezaret_sadiqdir(setirler: Sequence[dict]) -> tuple[bool, list[str]]:
    """Qol 0 orijinal davranışı təkrar istehsal edirmi.

    ÇOXLUQ üzrə qiymətləndirilir, hər təkrar üzrə yox: model determinist
    deyil, ona görə tək sapma sadiqsizlik deyil. Amma çoxluq sapırsa, təkrar
    oynatma orijinal sistemi əks etdirmir və CƏDVƏL ETİBARSIZDIR.
    """
    gozlenilen = {c.case_id: c.nezaret_davranisi for c in PROBE_CASES}
    sebebler: list[str] = []
    for case_id, gozlenen in gozlenilen.items():
        siniflər = [
            s["sinif"] for s in setirler
            if s["case_id"] == case_id and s["qol"] == NEZARET_QOLU
        ]
        if not siniflər:
            continue
        uygun = sum(1 for s in siniflər if s == gozlenen)
        if uygun * 2 <= len(siniflər):
            sebebler.append(
                f"{case_id}: nəzarət qolu «{gozlenen}» gözlənilirdi, "
                f"{len(siniflər)} təkrarın yalnız {uygun}-i uyğun gəldi"
            )
    return (not sebebler), sebebler


# ---------------------------------------------------------------------------
# Həqiqi SUT, saxlanmış kontekst və CLI
# ---------------------------------------------------------------------------


def sut_qurucular(settings: Settings) -> SutQurucular:
    """HƏQİQİ SUT — `rag.*` import edilən YEGANƏ yer.

    Bura yalnız CLI yolundan çağırılır, ona görə testlər chroma tələb etmir.
    """
    add_sut_to_path(settings)
    from rag.pipeline import (  # noqa: PLC0415
        SYSTEM_INSTRUCTION,
        _is_refusal,
        build_user_message,
        make_nonce,
    )
    from rag.store import RetrievedChunk  # noqa: PLC0415

    yoxla_saheler(RetrievedChunk)
    return SutQurucular(
        system_instruction=SYSTEM_INSTRUCTION,
        build_user_message=build_user_message,
        chunk_cls=RetrievedChunk,
        # `has_citation=False`: probe sitat yoxlamasını təkrar etmir, yalnız
        # imtina cümləsini tanıyır. SUT-un öz funksiyası olduğu üçün tərif
        # bir dənədir.
        is_refusal=lambda text: _is_refusal(text, has_citation=False),
        make_nonce=make_nonce,
    )


def llm_qur(settings: Settings):
    """SUT-un öz LLM qurucusu — probe başqa model işlətməməlidir."""
    add_sut_to_path(settings)
    from rag.config import Settings as SutSettings  # noqa: PLC0415
    from rag.config import build_llm  # noqa: PLC0415

    return build_llm(SutSettings.load(openai_api_key=settings.require_openai_key()))


def musahideleri_oxu(run_id: str) -> dict[str, dict]:
    """Saxlanmış run-dan hər case üçün BİRİNCİ təkrarın kontekstini götürür.

    Birinci təkrar seçilir, çünki ölçülən şey qollar arasındakı fərqdir —
    başlanğıc kontekst hamısı üçün eyni olmalıdır.
    """
    settings = Settings.load()
    paths = RunPaths.for_run(settings.runs_dir, run_id)
    if not paths.exists():
        raise FileNotFoundError(paths.root)
    run = load_run(paths)
    out: dict[str, dict] = {}
    for obs in run.observations:
        if obs.repeat != 1 or obs.case_id in out:
            continue
        # Sintez çağırışının system hash-i və modeli: qol 0 onlarla
        # tutuşdurulur. Çağırış olmayan case (qapı kəsib) bura düşsə, dəyərlər
        # boş qalır və yoxlamalar onu atlayır — uydurulmuş dəyər yazmaqdansa
        # boşluğu etiraf etmək doğrudur.
        sintez = [c for c in obs.llm_calls if c.role == "synthesis"]
        out[obs.case_id] = {
            "case_id": obs.case_id,
            "question": obs.question,
            "chunks": list(obs.chunks),
            "system_sha256": sintez[0].system_sha256 if sintez else "",
            "model": sintez[0].model if sintez else "",
        }
    return out


def _nezaret_qolu_uzre(
    setirler: Sequence[dict],
    observations: dict[str, dict],
    *,
    sahe: str,
    sebeb_metni: Callable[[str, str, str], str],
) -> tuple[bool, list[str]]:
    """Qol 0-ın bir sahəsini saxlanmış run ilə tutuşdurur.

    YALNIZ NƏZARƏT QOLU: qol 1-3 prompt-u qəsdən dəyişir. Saxlanmış dəyər
    boşdursa yoxlama ATLANIR — bilinməyəni uyğunsuzluq kimi göstərmək
    ölçməni əsassız dayandırardı.
    """
    sebebler: list[str] = []
    for setir in setirler:
        if setir["qol"] != NEZARET_QOLU:
            continue
        gozlenilen = (observations.get(setir["case_id"]) or {}).get(sahe, "")
        if not gozlenilen or setir[sahe] == gozlenilen:
            continue
        sebeb = sebeb_metni(setir["case_id"], gozlenilen, setir[sahe])
        if sebeb not in sebebler:
            sebebler.append(sebeb)
    return (not sebebler), sebebler


def system_prompt_deyismeyib(
    setirler: Sequence[dict], observations: dict[str, dict]
) -> tuple[bool, list[str]]:
    """Qol 0-ın system prompt-u saxlanmış run-dakı ilə eynidirmi.

    Davranış yoxlamasından AYRIDIR və ondan güclüdür: davranış təsadüfən
    üst-üstə düşə bilər, hash düşməz.
    """
    return _nezaret_qolu_uzre(
        setirler, observations, sahe="system_sha256",
        sebeb_metni=lambda case_id, gozlenilen, indi: (
            f"{case_id}: system prompt dəyişib — "
            f"run-da `{gozlenilen}`, indi `{indi}`"
        ),
    )


def model_deyismeyib(
    setirler: Sequence[dict], observations: dict[str, dict]
) -> tuple[bool, list[str]]:
    """Probe saxlanmış run ilə EYNİ modeli ölçürmü.

    Model `LLM_MODEL` env dəyişəni ilə gəlir, yəni `config_hash`-a düşmür və
    səssizcə sürüşə bilər. Başqa modellə ölçmək qolları müqayisə etməyi
    dayandırır: fərq qoldan da, modeldən də gələ bilər və ayırd edilə bilməz.
    Davranış qapısı bunu tutmaya bilər — başqa model də gözlənilən sinfi
    qaytara bilər.
    """
    return _nezaret_qolu_uzre(
        setirler, observations, sahe="model",
        sebeb_metni=lambda case_id, gozlenilen, indi: (
            f"{case_id}: model dəyişib — run `{gozlenilen}` ilə ölçülüb, "
            f"probe isə `{indi}` işlədir"
        ),
    )


def _summary_govdesi(
    setirler: Sequence[dict], sadiq: bool, sebebler: Sequence[str]
) -> str:
    """Sənədə köçürüləcək cədvəl — hər case × qol üçün üstün sinif."""
    qol_adlari = [q.ad for q in QOLLAR]
    matris: dict[tuple[str, str], list[str]] = {}
    for s in setirler:
        matris.setdefault((s["case_id"], s["qol"]), []).append(s["sinif"])

    cedvel_setirleri = []
    for case in PROBE_CASES:
        setir = [case.case_id]
        for qol in qol_adlari:
            siniflər = matris.get((case.case_id, qol), [])
            if not siniflər:
                setir.append("—")
                continue
            usta = max(sorted(set(siniflər)), key=siniflər.count)
            setir.append(f"{usta} ({siniflər.count(usta)}/{len(siniflər)})")
        cedvel_setirleri.append(setir)

    out = cedvel(["case", *qol_adlari], cedvel_setirleri)
    if not sadiq:
        out += "\n\n> ⚠️ **NƏZARƏT QOLU SADİQ DEYİL** — " + "; ".join(sebebler) + (
            ". Yuxarıdakı cədvəl orijinal sistemi əks etdirmir və nəticə "
            "çıxarmaq üçün işlədilə bilməz."
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="kontekstin götürüləcəyi dev run_id")
    ap.add_argument("--probes-dir", type=Path, default=None)
    ap.add_argument("--tekrar", type=int, default=3)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)

    settings = Settings.load()
    try:
        observations = musahideleri_oxu(args.run)
    except FileNotFoundError as exc:
        print(f"Run tapılmadı: {exc}", file=sys.stderr)
        return 2

    from eval.dataset import load_dataset  # noqa: PLC0415

    dataset = load_dataset(settings.dataset_path)
    probes_dir = args.probes_dir or (settings.logs_dir / PROBES_DIRNAME)
    writer, kimlik = probe_yarat(
        alet="generasiya",
        oxlar=[qol.ad for qol in QOLLAR[1:]],
        argv=["python", "tools/generation_probe.py", *argv],
        settings=settings,
        dataset=dataset,
        probes_dir=probes_dir,
    )

    try:
        llm = llm_qur(settings)
        setirler = olc(
            observations, llm=llm, q=sut_qurucular(settings), tekrar=args.tekrar,
        )
        for setir in setirler:
            writer.append_row(setir)

        davranis_ok, davranis_sebebleri = nezaret_sadiqdir(setirler)
        hash_ok, hash_sebebleri = system_prompt_deyismeyib(setirler, observations)
        model_ok, model_sebebleri = model_deyismeyib(setirler, observations)
        sadiq = davranis_ok and hash_ok and model_ok
        sebebler = [*model_sebebleri, *hash_sebebleri, *davranis_sebebleri]

        writer.write_manifest({
            **kimlik, "nezaret_sadiqdir": sadiq,
            "nezaret_sebebleri": sebebler, "tekrar": args.tekrar,
            "model": model_adi(llm),
        })
        writer.write_summary(summary_metni(
            basliq="Generasiya qolları (dev, saxlanmış kontekst)",
            probe_id_=writer.paths.probe_id, argv=kimlik["argv"],
            govde=_summary_govdesi(setirler, sadiq, sebebler),
        ))
    except BaseException as exc:  # noqa: BLE001 — KeyboardInterrupt da daxil
        writer.mark_failed(type(exc).__name__)
        raise

    print(f"\nArtefakt: {artefakt_yolu(writer.paths.root)}")
    if not sadiq:
        print("\n⚠️ NƏZARƏT QOLU SADİQ DEYİL — cədvəl etibarsızdır:", file=sys.stderr)
        for sebeb in sebebler:
            print(f"   {sebeb}", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
