"""İki alətin paylaşdığı probe köməkçiləri.

Sweep və eksperiment alətləri eyni müqaviləyə tabedir: hər icra öz
`logs/probes/<probe_id>/` qovluğunu alır, kimlik bloku argv-ni saxlayır və
sənədə köçürülən cədvəl `summary.md`-də generasiya olunur. Həmin ortaq
hissə burada, ona görə iki alət bir-birindən sürüşə bilmir.
"""

from __future__ import annotations

import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.artifacts import ProbePaths, ProbeWriter, probe_id, probe_identity  # noqa: E402
from eval.config import Settings  # noqa: E402
from eval.dataset import Dataset  # noqa: E402
from eval.runner import read_harness_commit  # noqa: E402

PROBES_DIRNAME = "probes"


def indi_utc() -> str:
    """`20260812T131500Z` — `run_id` ilə eyni format."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def add_sut_to_path(settings: Settings) -> None:
    """SUT yolu BİR DƏFƏ əlavə olunur.

    Əvvəlki versiya hər eksperiment üçün `sys.path.insert` edirdi — 9
    eksperimentdə 9 eyni giriş. Zərərsiz görünür, amma `sys.path` uzunluğu
    import vaxtına təsir edir və təkrar girişlər hansı nüsxənin işlədiyini
    oxumağı çətinləşdirir.
    """
    yol = str(settings.sut_path)
    if yol not in sys.path:
        sys.path.insert(0, yol)


def probe_yarat(
    *,
    alet: str,
    oxlar: Sequence[str],
    argv: Sequence[str],
    settings: Settings,
    dataset: Dataset,
    probes_dir: Path,
    sut_commit: str = "",
) -> tuple[ProbeWriter, dict[str, Any]]:
    """Probe qovluğunu açır və kimlik blokunu qaytarır.

    `sut_commit` boş qala bilər: sweep SUT-un git vəziyyətini yoxlamır
    (bu, `RagSut.preflight`-in işidir və o, pullu yolda çağırılır). Boş
    dəyəri uydurulmuş commit-lə doldurmaq artefaktı yalan edərdi.
    """
    # VAXT BİR DƏFƏ OXUNUR. İki ayrı `indi_utc()` çağırışı saniyə sərhədini
    # kəsə bilirdi: qovluq adı `…T131459Z`, manifestin `started_at`-i isə
    # `…T131500Z` olurdu. Qovluq adı hər şeyin istinad etdiyi KİMLİKDİR
    # (sənəd ona görə sitat gətirir), ona görə iki dəyər arasındakı fərq
    # sonradan «hansı icra?» sualını çətinləşdirir.
    indi = indi_utc()
    kimlik = probe_identity(
        alet=alet,
        argv=argv,
        started_at=indi,
        harness_commit=read_harness_commit(PROJECT_ROOT),
        sut_commit=sut_commit or settings.sut_commit,
        config_hash=settings.config_hash,
        dataset_sha256=dataset.sha256,
        koke=PROJECT_ROOT,
    )
    # Kimlik writer-ə ÖTÜRÜLÜR: qovluq yaranan anda manifest də düşür, ona
    # görə kəsilmiş icra da kim olduğunu deyə bilir.
    writer = ProbeWriter(
        ProbePaths.for_probe(probes_dir, probe_id(alet=alet, oxlar=oxlar, indi=indi)),
        secrets=settings.live_secrets,
        kimlik=kimlik,
    )
    return writer, kimlik


def artefakt_yolu(kok: Path) -> str:
    """Artefakt qovluğunun ÇAP olunan yolu.

    `relative_to` repo-dan kənar yolda `ValueError` atır, yəni `--probes-dir`
    müvəqqəti qovluğa yönəldiləndə ölçmə BİTDİKDƏN sonra, artefakt artıq
    diskdə olduğu halda alət çökürdü — və `except BaseException` bunu
    «uğursuz» kimi möhürləyirdi. Kənar yol olduğu kimi çap olunur: oxucunun
    qovluğu tapa bilməsi ilə repo daxilində nisbi yol saxlamaq arasında
    ziddiyyət yoxdur.
    """
    try:
        return str(kok.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(kok)


def cedvel(basliqlar: Sequence[str], setirler: Sequence[Sequence[Any]]) -> str:
    """Markdown cədvəli — sənədə OLDUĞU KİMİ köçürülür."""
    out = ["| " + " | ".join(basliqlar) + " |", "|" + "---|" * len(basliqlar)]
    for setir in setirler:
        out.append("| " + " | ".join(str(x) for x in setir) + " |")
    return "\n".join(out)


def summary_metni(*, basliq: str, probe_id_: str, argv: Sequence[str], govde: str) -> str:
    """`summary.md` — `logs/*.md`-ə köçürüləcək blok.

    `tests/test_logs_iddialari.py` sənəddəki `<!-- artefakt: <id> -->`
    blokunun məhz bu mətndə olduğunu yoxlayır, ona görə format sabit
    saxlanılmalıdır.

    ƏMR YAPIŞDIRILA BİLƏN OLMALIDIR. `argv_temizle` maşın yolunu
    `<müvəqqəti-qovluq>` ilə əvəz edir, `<` və `>` isə shell-də
    yönləndirmədir: sitatsız sətir `--workdir`-i dəyərsiz qoyur və shell
    `syntax error` verir. Bloka «bu cədvəl hansı əmrlə alınıb?» sualının
    cavabı kimi baxılır — işləməyən əmr bu vəzifəni yerinə yetirmir.
    `shlex.quote` yalnız ehtiyacı olan tokenə toxunur, ona görə `--top-k 4`
    kimi ölçmə parametrləri oxunaqlı qalır.
    """
    emr = " ".join(shlex.quote(t) for t in argv)
    return (
        f"# {basliq}\n\n"
        f"`probe_id`: `{probe_id_}`\n\n"
        f"```bash\n{emr}\n```\n\n"
        f"{govde}\n"
    )
