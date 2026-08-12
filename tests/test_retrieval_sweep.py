"""`tools/retrieval_sweep.py` — arqument yoxlaması və artefakt müqaviləsi.

Açarsız və şəbəkəsiz: alətin `rag.*` import-ları funksiya daxilindədir, ona
görə modul langchain/chroma olmadan import olunur və `main()` pis
arqumentlərdə SUT-a çatmadan qayıdır.
"""

from __future__ import annotations

from tools.retrieval_sweep import Candidate, build_parser, main, swept_axes


def test_parser_qurulur_ve_defoltlar_baseline_dir() -> None:
    args = build_parser().parse_args([])
    assert args.top_k == [4]
    assert args.threshold == [0.42]
    assert args.margin == [0.10]
    assert args.lexical == [0.35]


def test_ARQUMENTLER_SUT_a_getmezden_EVVEL_yoxlanilir() -> None:
    """D5: `--threshold 1.8` işləyib 0/8 cədvəli çap edirdi.

    O cədvəl tapıntı kimi oxunurdu, halbuki sadəcə şkala səhvi idi. İndi
    `main()` 1 qaytarır və heç bir embedding çağırışı olmur.
    """
    assert main(["--threshold", "1.8"]) == 1


def test_NaN_marja_SUT_a_getmir() -> None:
    assert main(["--margin", "nan"]) == 1


def test_SONSUZ_marja_SUT_a_getmir() -> None:
    assert main(["--margin", "inf"]) == 1


def test_hedden_boyuk_top_k_SUT_a_getmir() -> None:
    """k > 50 retrieval deyil, korpusun tamamını prompta tökməkdir."""
    assert main(["--top-k", "99999"]) == 1


def test_gridin_BIR_pis_deyeri_de_butun_icrani_dayandirir() -> None:
    """Yarısı ölçülüb yarısı dayanan sweep yarımçıq artefakt yaradardı."""
    assert main(["--threshold", "0.42", "1.8"]) == 1


# --- namizəd etiketi -------------------------------------------------------


def test_yumsaq_hedd_SIFIRLANANDA_etiket_bunu_deyir() -> None:
    """`marja >= astana` dense qapını söndürür — qanuni, amma görünməlidir."""
    cand = Candidate(top_k=4, threshold=0.42, soft_floor_margin=0.50, lexical_threshold=0.35)
    assert cand.qapi_sonur
    assert "dense qapı sönür" in cand.label


def test_normal_namizedde_xeberdarliq_yoxdur() -> None:
    cand = Candidate(top_k=4, threshold=0.42, soft_floor_margin=0.10, lexical_threshold=0.35)
    assert not cand.qapi_sonur
    assert "sönür" not in cand.label
    assert abs(cand.soft_floor - 0.32) < 1e-9


# --- probe_id oxları -------------------------------------------------------


def test_swept_axes_YALNIZ_birden_cox_deyeri_olan_oxu_sayir() -> None:
    """`probe_id` qovluq siyahısında hansı sweep-in nəyi əhatə etdiyini deyir."""
    args = build_parser().parse_args(["--top-k", "4", "6", "--lexical", "0.35"])
    assert swept_axes(args) == ["top_k"]


def test_swept_axes_bos_olanda_probe_id_sabit_deyir() -> None:
    from eval.artifacts import probe_id

    args = build_parser().parse_args([])
    assert swept_axes(args) == []
    assert probe_id(alet="sweep", oxlar=[], indi="20260812T000000Z").endswith("-sabit")
