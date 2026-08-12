"""Sənəddəki hər cədvəl artefaktda MÖVCUD olmalıdır.

NİYƏ BU TEST VAR
----------------
2026-08-12-də aşkarlandı ki, `logs/retrieval_sweep.md` dörd ox üzrə nəticə
yazır, diskdəki artefakt isə yalnız birini saxlayır: sweep üç dəfə eyni fayla
yazmış, sonuncu qalanları əvəzləmişdi. Sənəd rəqəm iddia edirdi, sübut isə
yox idi — və heç nə bunu tutmurdu.

İntizam artıq yazılı idi («hər rəqəm təkrar istehsal oluna bilən artefaktla
dəstəklənir»), amma yazılı intizam sürüşür. Bu test onu hesablanan hala
gətirir.

MÜQAVİLƏ
--------
`logs/*.md` faylında belə bir blok:

    <!-- artefakt: <probe_id> -->
    | cədvəl | sətirləri |
    <!-- /artefakt -->

blokun içindəki hər boş olmayan sətir
`logs/probes/<probe_id>/summary.md`-də HƏRFİ-HƏRFİNƏ mövcud olmalıdır.

Sətir səviyyəsində yoxlanır, blok səviyyəsində yox: sənəd 54 sətirlik
cədvəlin yalnız bir hissəsini sitat gətirə bilər — bu, dürüstdür, çünki
tam cədvələ yol da göstərilir. Qadağan olunan şey uydurulmuş və ya
transkripsiyada təhrif olunmuş sətirdir.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS = PROJECT_ROOT / "logs"
PROBES = LOGS / "probes"

_BLOK = re.compile(
    r"<!--\s*artefakt:\s*(?P<probe_id>[^\s>]+)\s*-->(?P<govde>.*?)<!--\s*/artefakt\s*-->",
    re.DOTALL,
)


def _bloklar() -> list[tuple[Path, str, str]]:
    out = []
    for md in sorted(LOGS.glob("*.md")):
        for m in _BLOK.finditer(md.read_text(encoding="utf-8")):
            out.append((md, m.group("probe_id"), m.group("govde")))
    return out


def test_en_azi_bir_iddia_artefakta_baglanib() -> None:
    """Test özü sükutla keçməməlidir.

    Marker konvensiyası unudulsa, aşağıdakı parametrli test SIFIR halda
    işləyər və dəst yaşıl qalar — yəni qoruma yox olar, amma bunu heç kim
    görməz.
    """
    assert _bloklar(), (
        "logs/*.md-də heç bir <!-- artefakt: ... --> bloku tapılmadı. "
        "Nəticə cədvəlləri artefakta bağlanmalıdır."
    )


@pytest.mark.parametrize("md,probe_id,govde", _bloklar())
def test_iddia_bloku_ARTEFAKTDA_movcuddur(md: Path, probe_id: str, govde: str) -> None:
    summary = PROBES / probe_id / "summary.md"
    assert summary.exists(), (
        f"{md.name}: `{probe_id}` artefaktı yoxdur ({summary}). "
        "Sənəd mövcud olmayan sübuta istinad edir."
    )
    metn = summary.read_text(encoding="utf-8")

    for setir in (s.strip() for s in govde.splitlines()):
        if not setir:
            continue
        assert setir in metn, (
            f"{md.name} → `{probe_id}`: bu sətir artefaktda yoxdur:\n"
            f"  {setir}\n"
            "Ya sənəd əl ilə redaktə olunub, ya ölçmə yenidən aparılmayıb."
        )


def test_probe_artefaktlarinin_hamisinda_manifest_var() -> None:
    if not PROBES.exists():
        pytest.skip("probe artefaktı yoxdur")
    for qovluq in sorted(p for p in PROBES.iterdir() if p.is_dir()):
        assert (qovluq / "manifest.json").exists(), f"{qovluq.name}: manifest yoxdur"
        kimlik = json.loads((qovluq / "manifest.json").read_text(encoding="utf-8"))
        assert kimlik.get("argv"), f"{qovluq.name}: argv qeyd olunmayıb"
        assert kimlik.get("started_at"), f"{qovluq.name}: vaxt qeyd olunmayıb"


# Maşına aid mütləq yol prefiksləri. Repo daxilindəki yollar nisbi yazılır
# (`eval.artifacts.argv_temizle`), ona görə bunlardan biri artefaktda görünürsə
# — ya təmizləmə işləməyib, ya artefakt əl ilə yaradılıb.
_MASIN_YOLU = re.compile(r"(?<![\w./])/(?:Users|home|private|tmp|var|opt)/[^\s\"'\\]*")


def test_probe_artefaktlarinda_MASINA_aid_yol_yoxdur() -> None:
    """Artefakt ictimai repo-ya düşür — mütləq yol maşını tanıdır.

    2026-08-12-də eksperiment manifesti `--workdir`-i olduğu kimi saxlamışdı:
    müvəqqəti sessiya qovluğunun tam yolu, istifadəçi adı ilə birlikdə.
    Yol nəticəyə təsir etmir (istənilən boş qovluq eyni cədvəli verir), ona
    görə onu saxlamağın faydası yox, yalnız zərəri var.
    """
    if not PROBES.exists():
        pytest.skip("probe artefaktı yoxdur")
    tapinti: list[str] = []
    for fayl in sorted(PROBES.rglob("*")):
        if not fayl.is_file():
            continue
        for tapilan in _MASIN_YOLU.findall(fayl.read_text(encoding="utf-8")):
            tapinti.append(f"{fayl.relative_to(PROJECT_ROOT)}: {tapilan}")
    assert not tapinti, "artefaktda maşına aid mütləq yol:\n  " + "\n  ".join(tapinti)


def test_ARTEFAKTLARDA_NaN_ve_Infinity_yoxdur() -> None:
    """Çılpaq `NaN` JSON deyil (RFC 8259) — belə artefakt sübut kimi oxunmur."""

    def _partla(deyer: str):
        raise AssertionError(f"artefaktda JSON olmayan sabit: {deyer}")

    fayllar = [
        *PROJECT_ROOT.glob("runs/*/manifest.json"),
        *PROBES.glob("*/rows.jsonl"),
        *PROBES.glob("*/manifest.json"),
    ]
    assert fayllar, "yoxlanacaq artefakt tapılmadı"
    for fayl in fayllar:
        for sətir in fayl.read_text(encoding="utf-8").splitlines():
            if not sətir.strip():
                continue
            if fayl.suffix == ".jsonl":
                json.loads(sətir, parse_constant=_partla)
        if fayl.suffix == ".json":
            data = json.loads(fayl.read_text(encoding="utf-8"), parse_constant=_partla)
            _sonlu_yoxla(data, fayl)


def _sonlu_yoxla(data, fayl: Path) -> None:
    if isinstance(data, dict):
        for v in data.values():
            _sonlu_yoxla(v, fayl)
    elif isinstance(data, list):
        for v in data:
            _sonlu_yoxla(v, fayl)
    elif isinstance(data, float):
        assert math.isfinite(data), f"{fayl}: sonlu olmayan ədəd"
