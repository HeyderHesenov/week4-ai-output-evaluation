"""Domen xətaları — SDK istisnaları buradan yuxarı qalxmır.

NİYƏ AYRICA İYERARXİYA
----------------------
`runner`, `report` və `cli` nə `openai`, nə `anthropic` paketini import
etməməlidir. Hər iki SDK-nın istisnaları yalnız BİR yerdə — `judge.py`-dakı
`_to_domain_error()` və `sut.py`-dakı adapterdə — bu siniflərə çevrilir.
Bunun iki praktik faydası var:

1. Test dəsti açarsız və SDK-sız işləyə bilir (fake-lər eyni domen
   xətalarını atır).
2. SDK versiya yeniləməsi zamanı istisna adı dəyişsə, sınıq yer bir
   fayldadır, bütün kod bazası deyil.

XƏTA SEÇİMİ
-----------
`EvalError` qəsdən `RuntimeError`-dan törəyir, `Exception`-dan yox: CLI
`except EvalError` ilə tutur, amma `KeyboardInterrupt` və `SystemExit`
(hər ikisi `BaseException`) ötüb keçir.
"""

from __future__ import annotations


class EvalError(RuntimeError):
    """Bütün framework xətalarının kökü."""


class ConfigError(EvalError):
    """İstifadəçi `.env` və ya mühit dəyişənini düzəltməlidir."""


class DatasetError(EvalError):
    """`data/testset.yaml` sxem və ya məntiq baxımından yanlışdır."""


class ContaminationError(EvalError):
    """dev/holdout intizamı pozulub.

    Bu xəta qəsdən DAYANDIRICIDIR (exit 2). Çirklənmiş qiymətləndirmə
    yanlış nəticədən daha pisdir: yanlış nəticə görünür, çirklənmiş
    nəticə isə düzgün görünür.
    """


class SutError(EvalError):
    """Qiymətləndirilən sistem istifadəyə yararsızdır.

    Nümunə: boş Chroma indeksi, gözlənilməyən commit, tapılmayan modul.
    Bu xətanın PULLU çağırışlardan ƏVVƏL atılması vacibdir.
    """


class ArtifactError(EvalError):
    """JSONL artefaktı oxunmur, natamamdır və ya replay-də çatışmır."""


class JudgeTransientError(EvalError):
    """Keçici hakim xətası — təkrar cəhd məntiqlidir (429, 5xx, timeout)."""


class JudgePermanentError(EvalError):
    """Daimi hakim xətası — təkrar cəhd mənasızdır (401, 403, 404, 400)."""
