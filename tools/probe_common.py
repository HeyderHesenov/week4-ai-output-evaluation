"""İki alətin paylaşdığı probe köməkçiləri.

Sweep və eksperiment alətləri eyni müqaviləyə tabedir: hər icra öz
`logs/probes/<probe_id>/` qovluğunu alır, kimlik bloku argv-ni saxlayır və
sənədə köçürülən cədvəl `summary.md`-də generasiya olunur. Həmin ortaq
hissə burada, ona görə iki alət bir-birindən sürüşə bilmir.
"""

from __future__ import annotations

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
    pid = probe_id(alet=alet, oxlar=oxlar, indi=indi_utc())
    writer = ProbeWriter(
        ProbePaths.for_probe(probes_dir, pid), secrets=settings.live_secrets
    )
    kimlik = probe_identity(
        alet=alet,
        argv=argv,
        started_at=indi_utc(),
        harness_commit=read_harness_commit(PROJECT_ROOT),
        sut_commit=sut_commit or settings.sut_commit,
        config_hash=settings.config_hash,
        dataset_sha256=dataset.sha256,
        koke=PROJECT_ROOT,
    )
    return writer, kimlik


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
    """
    emr = " ".join(argv)
    return (
        f"# {basliq}\n\n"
        f"`probe_id`: `{probe_id_}`\n\n"
        f"```bash\n{emr}\n```\n\n"
        f"{govde}\n"
    )
