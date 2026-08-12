"""Markdown hesabat — rəqəm və onun ETİBARLILIĞI yan-yana.

HESABATIN QAYDASI
-----------------
Hər iddia ya rəqəmlə, ya da «ölçülmədi» ilə müşayiət olunur. Üç yerdə bu
qayda xüsusilə görünür:

  1. Pass-rate həmişə Wilson intervalı ilə çap olunur — 8 case-də «100%»
     tək başına yanıltıcıdır.
  2. Müqayisə cütlənmiş McNemar testi ilə verilir; p-dəyər əhəmiyyətsizdirsə,
     bu, gizlədilmir — «yaxşılaşma ölçülə bilmədi» yazılır.
  3. Xərc cədvəlinin altında təsdiqlənməmiş qiymətlərin banneri və embedding
     boşluğu qeyd olunur.

Hesabat holdout registrini TAM çap edir. Bu, dizayn qərarıdır: yoxlayıcı
holdout-un bir yoxsa doqquz dəfə işlədildiyini özü görsün.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .artifacts import RunArtifacts
from .config import PriceTable
from .dataset import Dataset
from .metrics import (
    EMBEDDING_GAP_NOTE,
    UNMEASURABLE,
    LabelSource,
    cost_report,
    judge_bias,
    latency_report,
    mcnemar,
    pass_rate,
    pass_rate_by,
    percentile,
    stable_pass,
    unstable_cases,
)
from .rootcause import group_by_category, group_by_layer
from .split import LedgerEntry


def build_report(
    run: RunArtifacts,
    dataset: Dataset,
    prices: PriceTable,
    *,
    baseline: RunArtifacts | None = None,
    human_labels: Mapping[str, int] | None = None,
    label_sources: Mapping[str, LabelSource] | None = None,
    ledger: Sequence[LedgerEntry] = (),
) -> str:
    blocks = [
        _header(run),
        _summary(run),
        _by_category(run, dataset),
        _root_causes(run),
        _stability(run),
        _failures(run),
    ]
    if baseline is not None:
        blocks.append(_comparison(baseline, run))
    blocks.extend(
        [
            _cost(run, prices),
            _latency(run),
            _judge_section(run, human_labels or {}, label_sources),
            _integrity(run, ledger),
        ]
    )
    return "\n\n".join(b for b in blocks if b).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Bölmələr
# ---------------------------------------------------------------------------


def _header(run: RunArtifacts) -> str:
    m = run.manifest
    rows = [
        ("Run", m.get("run_id", run.run_id)),
        ("Bölgü", m.get("split", "?")),
        ("Variant", f"{m.get('variant_label', '?')} (`{m.get('variant_id', '?')}`)"),
        ("Başladı / bitdi", f"{m.get('started_at', '?')} → {m.get('finished_at', '?')}"),
        ("Case sayı × təkrar", f"{m.get('case_count', '?')} × {m.get('repeats', '?')}"),
        ("Test dəsti sha256", f"`{str(m.get('dataset_sha256', ''))[:16]}`"),
        ("Bölgü manifesti sha256", f"`{str(m.get('split_manifest_sha256', ''))[:16]}`"),
        ("Variant sha256", f"`{str(m.get('variant_sha256', ''))[:16]}`"),
        ("Hakim prompt sha256", f"`{str(m.get('judge_prompt_sha256', ''))[:16]}`"),
        ("Konfiq hash", f"`{m.get('config_hash', '')}`"),
        ("SUT commit", f"`{str(m.get('sut_commit', ''))[:12]}`"),
        ("Harness commit", f"`{str(m.get('harness_commit', ''))[:12]}`"),
    ]
    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return (
        f"# Qiymətləndirmə hesabatı — {m.get('run_id', run.run_id)}\n\n"
        "İki run yalnız BÜTÜN aşağıdakı hash-lər üst-üstə düşəndə müqayisə\n"
        "edilə bilər. Fərqli hash = fərqli ölçmə.\n\n"
        "| | |\n|---|---|\n" + body
    )


def _summary(run: RunArtifacts) -> str:
    rate = pass_rate(run.causes)
    lines = [
        "## Xülasə",
        "",
        f"- **Keçid nisbəti:** {rate.describe()}",
        f"- **Sərt nisbət** (ölçülə bilməyənlər uğursuz sayılır): {rate.strict_rate:.0%}",
    ]
    if rate.unmeasurable:
        lines.append(
            f"- ⚠️ {rate.unmeasurable} nəticə ölçülə bilmədi (SUT/hakim xətası). "
            "Bu rəqəm sistemin keyfiyyəti haqqında məlumat DAŞIMIR — ölçmə qatına baxın."
        )
    return "\n".join(lines)


def _by_category(run: RunArtifacts, dataset: Dataset) -> str:
    key = {c.id: c.category for c in dataset}
    rates = pass_rate_by(run.causes, key)
    if not rates:
        return ""
    rows = "\n".join(
        f"| {name} | {r.passed}/{r.measured} | {r.rate:.0%} | "
        f"{r.interval[0]:.0%}-{r.interval[1]:.0%} | {r.unmeasurable} |"
        for name, r in rates.items()
    )
    return (
        "## Kateqoriya üzrə\n\n"
        "| Kateqoriya | Keçdi | Nisbət | 95% CI | Ölçülməyən |\n"
        "|---|---|---|---|---|\n" + rows
    )


def _root_causes(run: RunArtifacts) -> str:
    layers = group_by_layer(list(run.causes))
    categories = group_by_category(list(run.causes))
    if not layers:
        return "## Kök səbəb\n\nUğursuzluq yoxdur."

    layer_rows = "\n".join(f"| {name} | {count} |" for name, count in layers.items())
    cat_rows = "\n".join(
        f"| {name} | {count} |" for name, count in categories.items() if name != "ok"
    )
    return (
        "## Kök səbəb\n\n"
        "Prompt-u düzəltmək retrieval qatındakı uğursuzluğa bir bal da qazandırmır — "
        "ona görə əsas cədvəl QAT üzrədir.\n\n"
        "| Qat | Uğursuzluq |\n|---|---|\n" + layer_rows + "\n\n"
        "<details><summary>Kateqoriya üzrə detal</summary>\n\n"
        "| Kateqoriya | Say |\n|---|---|\n" + cat_rows + "\n\n</details>"
    )


def _stability(run: RunArtifacts) -> str:
    outcomes = _outcomes(run)
    repeats = int(run.manifest.get("repeats", 1) or 1)
    if repeats < 2:
        return (
            "## Sabitlik\n\n"
            "REPEATS=1 — sürüşkənlik ölçülməyib. SUT temperature=0-da belə "
            "deterministik deyil; holdout üçün REPEATS=3 tövsiyə olunur."
        )
    unstable = unstable_cases(outcomes)
    stable = stable_pass(outcomes)
    passed = sum(1 for v in stable.values() if v)
    lines = [
        "## Sabitlik",
        "",
        f"- Təkrar sayı: {repeats}",
        f"- BÜTÜN təkrarlarda keçən case: {passed}/{len(stable)}",
    ]
    if unstable:
        lines.append(f"- ⚠️ Sürüşən case-lər (təmiz keçid sayılmır): {', '.join(unstable)}")
    else:
        lines.append("- Sürüşən case yoxdur.")
    return "\n".join(lines)


def _failures(run: RunArtifacts) -> str:
    failures = [c for c in run.causes if c.is_failure]
    if not failures:
        return ""
    rows = "\n".join(
        f"| `{c.case_id}` | {c.repeat} | {c.category} | {c.layer} | {_escape(c.detail)} |"
        for c in failures
    )
    return (
        "## Uğursuz case-lər\n\n"
        "| Case | Təkrar | Kateqoriya | Qat | Detal |\n|---|---|---|---|---|\n" + rows
    )


def _comparison(baseline: RunArtifacts, variant: RunArtifacts) -> str:
    base_pass = stable_pass(_outcomes(baseline))
    var_pass = stable_pass(_outcomes(variant))
    result = mcnemar(base_pass, var_pass)

    base_rate = pass_rate(baseline.causes)
    var_rate = pass_rate(variant.causes)

    lines = [
        "## Before / after",
        "",
        f"- Baseline (`{baseline.manifest.get('variant_id', '?')}`): {base_rate.describe()}",
        f"- Variant (`{variant.manifest.get('variant_id', '?')}`): {var_rate.describe()}",
    ]
    # KONFİQURASİYA FƏRQİ — sənəd bu qaydanı yazırdı, amma heç nə yoxlamırdı.
    #
    # «İki run yalnız bütün hash-lər üst-üstə düşəndə müqayisə edilə bilər»
    # cümləsi hesabatın başındadır, lakin fərqli `config_hash`-li iki run
    # SƏSSİZCƏ müqayisə olunurdu. Retrieval override-ları gələndən sonra bu
    # daha da vacibdir: TOP_K dəyişən run baseline-dan qanuni olaraq fərqlidir
    # və oxucu bunu görməlidir. Hansı AÇARIN fərqləndiyi də sadalanır, çünki
    # «hash fərqlidir» tək başına oxucuya nə etməli olduğunu demir.
    lines += _config_diff_lines(baseline.manifest, variant.manifest)

    # Variant run-ı promptu HEÇ dəyişməyibsə, aşağıdakı McNemar iki eyni
    # sistemi müqayisə edir və «əhəmiyyətli deyil» nəticəsi yanıldıcıdır.
    applications = variant.manifest.get("variant_applications")
    if variant.manifest.get("variant_id") != "baseline" and applications == 0:
        lines.append(
            "- 🛑 **Variant bir dəfə də tətbiq olunmayıb** "
            "(`variant_applications: 0`): rol xəritəsi SUT-un promptunu "
            "tanımayıb, yəni bu run baseline ilə eynidir. Aşağıdakı müqayisə "
            "heç nə ölçmür."
        )
    lines += [
        "",
        "Cütlənmiş McNemar testi (eyni case-lər, dəqiq binom):",
        "",
        f"- {result.describe()}",
    ]
    if not result.significant:
        lines.append(
            "- ⚠️ Fərq statistik əhəmiyyətli DEYİL: bu nümunə ölçüsündə "
            "yaxşılaşmanı təsadüfdən ayırmaq mümkün olmadı. «Prompt işlədi» "
            "nəticəsi çıxarmaq üçün dəlil kifayət etmir."
        )
    if result.regressed:
        lines.append(
            f"- {result.regressed} case variantda GERİLƏYİB — orta rəqəm bunu gizlədir."
        )
    return "\n".join(lines)


# Yalnız TƏMSİLİ fərqi olan açarlar: dəyər başqa yazılır, ölçmə eynidir.
# `sut_path` mütləq yoldan repo-ya nisbi yola keçdi (2026-08-10 düzəlişi),
# ona görə ondan ƏVVƏL yazılmış artefaktlar yeni run-larla fərqli hash verir,
# halbuki sistem eynidir. Bunu «ölçmə fərqlidir» kimi göstərmək yalan olardı;
# gizlətmək də yalan olardı — ona görə ayrıca sətirdə deyilir.
_COSMETIC_CONFIG_KEYS = frozenset({"sut_path"})


def _diff_keys(base: Mapping[str, Any], var: Mapping[str, Any]) -> list[str]:
    return sorted(k for k in set(base) | set(var) if base.get(k) != var.get(k))


def _detail(base: Mapping[str, Any], var: Mapping[str, Any], keys: list[str]) -> str:
    return ", ".join(f"`{k}`: {base.get(k, '(yox)')} → {var.get(k, '(yox)')}" for k in keys)


def _config_diff_lines(base_manifest: Mapping[str, Any], var_manifest: Mapping[str, Any]) -> list[str]:
    """Müqayisə edilən iki run-ın konfiqurasiya fərqləri.

    `config_hash`-A GÜVƏNMİR — VƏ BU QƏSDƏNDİR.
    ---------------------------------------------
    İlk versiya yalnız hash fərqləndikdə işləyirdi və `config` blokuna
    baxırdı. Bu, real bir run-da YANLIŞ nəticə verdi: chunking dəyişikliyi
    (`CHUNK_SIZE=500`) SUT-a env vasitəsilə gedir, framework-un `config`
    blokuna DÜŞMÜR, ona görə `config_hash` eyni qalır. Hesabat isə
    «yalnız təmsili fərq, ölçülən sistem eynidir» yazdı — halbuki indeks
    tamamilə başqa idi.

    Ona görə indi əsl mənbə `sut_retrieval`-dır: o, pipeline-dan GERİ
    oxunur, yəni SUT-un həqiqətən nə işlətdiyini göstərir və env-dən gələn
    dəyişikliyi də tutur.
    """
    out: list[str] = []

    base_cfg = base_manifest.get("config") or {}
    var_cfg = var_manifest.get("config") or {}
    if base_manifest.get("config_hash") != var_manifest.get("config_hash"):
        differing = _diff_keys(base_cfg, var_cfg)
        substantive = [k for k in differing if k not in _COSMETIC_CONFIG_KEYS]
        cosmetic = [k for k in differing if k in _COSMETIC_CONFIG_KEYS]
        if substantive:
            out.append(
                f"- ⚠️ **Konfiqurasiya fərqlidir** ({len(substantive)} açar): "
                f"{_detail(base_cfg, var_cfg, substantive)}. Aşağıdakı fərq YALNIZ "
                "variantın deyil, bu dəyişikliklərin də nəticəsi ola bilər."
            )
        if cosmetic and not substantive:
            out.append(
                f"- ℹ️ Konfiqurasiyada yalnız təmsili fərq "
                f"({', '.join(f'`{k}`' for k in cosmetic)}): dəyər başqa formatda "
                "yazılıb. `config_hash` buna görə fərqlidir."
            )
        if not differing:
            out.append(
                "- ⚠️ `config_hash` fərqlidir, amma `config` blokunda fərq tapılmadı — "
                "artefakt köhnə sxemlə yazılıb. Müqayisə ehtiyatla oxunmalıdır."
            )

    # --- SUT-un FAKTİKİ indeksi və retrieval kimliyi ---
    #
    # SIRA ƏN GÜCLÜ ŞAHİDDƏN BAŞLAYIR. Hər biri əvvəlkindən zəifdir, ona görə
    # yalnız biri çap olunur — dörd sətir eyni şeyi dörd cür deyərdi.
    out.extend(_index_diff_lines(base_manifest, var_manifest))
    return out


def _index_diff_lines(
    base_manifest: Mapping[str, Any], var_manifest: Mapping[str, Any]
) -> list[str]:
    base_index = base_manifest.get("sut_index") or {}
    var_index = var_manifest.get("sut_index") or {}
    base_fp, var_fp = base_index.get("sha256"), var_index.get("sha256")

    # 1. Barmaq izi — indeksin MƏZMUNUNDAN törəyir, ən güclü sübutdur.
    if base_fp and var_fp:
        if base_fp != var_fp:
            return [
                f"- ⚠️ **İndeks FƏRQLİDİR** — barmaq izi `{base_fp}` → `{var_fp}`. "
                "İki run eyni sənəd mətnləri üzərində ölçülməyib. Bu, "
                "`config_hash`-də GÖRÜNMÜR."
            ]
        return []

    # 2. Chunk sayı — HƏR manifestdə var, o cümlədən `sut_retrieval`-dan
    #    əvvəlki 2026-08-10 artefaktlarında. Köhnə run-larla müqayisədə
    #    yeganə şahid budur.
    base_count = base_manifest.get("sut_chunk_count")
    var_count = var_manifest.get("sut_chunk_count")
    if base_count is not None and var_count is not None and base_count != var_count:
        return [
            f"- ⚠️ **İndeksdəki chunk sayı fərqlidir** ({base_count} → {var_count}): "
            "iki run EYNİ indeksdə ölçülməyib. Bu rəqəm barmaq izindən əvvəlki "
            "artefaktlarda da mövcuddur, ona görə köhnə run-larla müqayisədə "
            "yeganə şahiddir."
        ]

    # 3. Niyyət səviyyəsi — `chunk_size` və qapı parametrləri.
    base_sut = base_manifest.get("sut_retrieval")
    var_sut = var_manifest.get("sut_retrieval")
    if base_sut is not None and var_sut is not None:
        differing = _diff_keys(base_sut, var_sut)
        if differing:
            return [
                f"- ⚠️ **SUT retrieval konfiqurasiyası fərqlidir** ({len(differing)} "
                f"açar): {_detail(base_sut, var_sut, differing)}. Bu, `config_hash`-də "
                "GÖRÜNMÜR — SUT parametrləri env-dən gəlir."
            ]
        return []

    # 4. Heç bir şahid yoxdur.
    return [
        "- ⚠️ Run-lardan birində `sut_index`/`sut_retrieval` qeydi yoxdur (artefakt "
        "bu sahələr əlavə olunmazdan əvvəl yazılıb). **İki run-ın eyni indeksdə "
        "ölçüldüyü artefaktdan TƏSDİQLƏNƏ BİLMİR** — `chunk_size` niyyətdir, "
        "indeksin özünün sübutu deyil."
    ]


def _cost(run: RunArtifacts, prices: PriceTable) -> str:
    calls = [c for o in run.observations for c in o.llm_calls]
    report = cost_report(calls, list(run.verdicts), prices)
    cases = len({o.case_id for o in run.observations})

    rows = "\n".join(
        f"| {m.model} | {m.provider} | {m.calls} | {m.input_tokens:,} | {m.output_tokens:,} "
        f"| {m.cached_read_tokens:,} | ${m.usd:.4f} |"
        for m in report.models
    ) or "| (çağırış yoxdur) | | | | | | |"

    lines = [
        "## Xərc",
        "",
        "| Model | Provayder | Çağırış | Giriş token | Çıxış token | Keş oxunuşu | USD |",
        "|---|---|---|---|---|---|---|",
        rows,
        "",
        f"**Cəmi: ${report.total_usd:.4f}** (case başına ${report.per_case_usd(cases):.4f})",
        "",
        f"> {EMBEDDING_GAP_NOTE}",
    ]
    if report.unverified:
        lines.append(
            f"> ⚠️ Təsdiqlənməmiş qiymətlər: {', '.join(report.unverified)}. "
            "Bu modellərin xərci TƏXMİNİDİR."
        )
    if report.missing_usage:
        lines.append(
            f"> ⚠️ {report.missing_usage} çağırışda token sayı alınmadı "
            "(`usage_source=missing`) — həmin çağırışlar xərcə DAXİL DEYİL."
        )
    return "\n".join(lines)


def _latency(run: RunArtifacts) -> str:
    report = latency_report(run.observations)
    if not report.observations:
        return ""
    totals = [o.total_ms for o in run.observations]
    lines = [
        "## Gecikmə",
        "",
        "Üç sütun yan-yanadır ki, cəmin bağlandığı görünsün — bağlanmayan hesab "
        "gizli ölçmə boşluğu deməkdir.",
        "",
        "| Hissə | Cəmi (ms) | Pay |",
        "|---|---|---|",
        f"| retrieval | {report.retrieval_ms:,.0f} | {report.share(report.retrieval_ms):.0%} |",
        f"| LLM | {report.llm_ms:,.0f} | {report.share(report.llm_ms):.0%} |",
        f"| overhead | {report.overhead_ms:,.0f} | {report.share(report.overhead_ms):.0%} |",
        f"| **cəmi** | **{report.total_ms:,.0f}** | {1.0:.0%} |",
        "",
        f"- Bağlanmayan qalıq: {report.unreconciled_ms:,.1f} ms",
        f"- Sorğu başına orta: {report.mean_ms:,.0f} ms; "
        f"p50 {percentile(totals, 0.5):,.0f} ms; p95 {percentile(totals, 0.95):,.0f} ms",
    ]
    return "\n".join(lines)


def _judge_section(
    run: RunArtifacts,
    human_labels: Mapping[str, int],
    label_sources: Mapping[str, LabelSource] | None = None,
) -> str:
    if not run.verdicts:
        return ""
    # Mənbə verilməyibsə, etiketlərin BU run-ın 1-ci təkrarına aid olduğu
    # qəbul edilir. Bu, «bütün verdiktlərə qarşı qoş» davranışından fərqlidir:
    # ünvan birmənalıdır, ona görə təkrarlar n-i şişirtmir.
    sources = dict(label_sources or {})
    assumed = sorted(cid for cid in human_labels if cid not in sources)
    for case_id in assumed:
        sources[case_id] = LabelSource(run.run_id, 1)

    bias = judge_bias(
        human_labels, sources, run.keyed_verdicts(), run.keyed_answer_lengths()
    )

    errors = [v for v in run.verdicts if not v.ok]
    lines = [
        "## Hakim",
        "",
        f"- Verdikt sayı: {len(run.verdicts)}; xətalı: {len(errors)}",
        f"- Model: `{run.manifest.get('config', {}).get('judge_model', '?')}`, "
        f"prompt sha256 `{str(run.manifest.get('judge_prompt_sha256', ''))[:16]}`",
    ]
    if human_labels:
        # Əhatə açıq yazılır: etiketlərin bir hissəsi başqa run-a aiddir və
        # onlar bu hesabatda ölçülmür. Bunu gizlətmək kappanı «tam» göstərərdi.
        lines.append(
            f"- İnsan etiketi əhatəsi: {bias.sample_size}/{len(human_labels)} "
            f"etiket bu run-da ölçüldü"
        )
    if assumed:
        # Bəyan edilmiş mənbə ilə fərz edilmiş mənbəni ayırmasaq, hesabat
        # hər iki halda eyni «tam əhatə» görüntüsü verər və insanın heç
        # görmədiyi cavaba qarşı hesablanmış kappa sağlam görünərdi.
        lines.append(
            f"- ⚠️ Mənbəsi BƏYAN EDİLMƏMİŞ etiket ({len(assumed)}): "
            f"{', '.join(assumed)} — bu run-ın 1-ci təkrarına aid FƏRZ edildi. "
            "Etiketin hansı cavaba aid olduğu təsdiqlənməyib."
        )
    if bias.sample_size:
        lines.append(f"- İnsan etiketi ilə razılıq: {bias.describe()}")
        if bias.kappa < 0.6:
            lines.append(
                "- ⚠️ kappa < 0.60: hakimin qərarı insan qərarı ilə zəif uzlaşır. "
                "Hakim-törəmə rəqəmlər ehtiyatla oxunmalıdır."
            )
    elif human_labels:
        lines.append(
            "- Bu run-a aid etiket yoxdur — kappa hesablanmadı. "
            "Etiketlər başqa run-ın cavablarına bağlıdır."
        )
    else:
        lines.append(
            "- İnsan etiketi verilməyib — kappa hesablanmadı "
            "(`data/human_labels.yaml` doldurun)."
        )
    if bias.unmatched:
        lines.append(
            f"- Bu run-da qarşılığı olmayan etiket: {', '.join(bias.unmatched)} "
            "(başqa run-a aiddir və ya cavab tapılmadı)"
        )
    if bias.unmeasurable:
        lines.append(
            f"- Ölçülə bilməyən etiket (hakim xətası/imtinası): "
            f"{', '.join(bias.unmeasurable)} — 0 bala çevrilməyib"
        )
    lines.append(
        f"- Verbosity qərəzi (bal ↔ cavab uzunluğu, Spearman ρ): "
        f"{bias.verbosity_rho:+.2f} (n={bias.verbosity_sample})"
    )
    if bias.verbosity_rho > 0.5 and bias.verbosity_sample >= 5:
        lines.append(
            "- ⚠️ Güclü müsbət korrelyasiya: hakim uzun cavaba yüksək bal verməyə "
            "meyllidir, yəni bal qismən sözçülüyü ölçür."
        )
    if errors:
        lines.append("")
        lines.append("Hakim xətaları (heç biri 0 bala çevrilməyib):")
        lines.extend(f"  - `{v.case_id}` — {_escape(v.error)}" for v in errors)
    return "\n".join(lines)


def _integrity(run: RunArtifacts, ledger: Sequence[LedgerEntry]) -> str:
    similarity = run.manifest.get("max_holdout_similarity", 0.0)
    lines = [
        "## Bütövlük",
        "",
        f"- Variant mətninin holdout sualları ilə ÖLÇÜLMÜŞ maksimum Jaccard "
        f"oxşarlığı: **{similarity:.2f}** (hədd: 0.60). Bu, iddia deyil, rəqəmdir.",
        f"- Test dəsti möhürləndikdən sonra dəyişməyib "
        f"(sha256 `{str(run.manifest.get('dataset_sha256', ''))[:16]}` uyğun gəldi).",
    ]
    lines.append("")
    lines.append("### Holdout registri (yalnız-əlavə)")
    lines.append("")
    if not ledger:
        lines.append("Holdout hələ işlədilməyib.")
    else:
        lines.append(f"Holdout **{len(ledger)}** dəfə işlədilib:")
        lines.append("")
        lines.append("| # | Vaxt | Variant | Case | Qeyd |")
        lines.append("|---|---|---|---|---|")
        lines.extend(
            f"| {i} | {e.at} | `{e.variant_id}` | {e.case_count} | {_escape(e.note)} |"
            for i, e in enumerate(ledger, start=1)
        )
        if len(ledger) > 2:
            lines.append("")
            lines.append(
                "> ⚠️ Holdout ikidən çox dəfə işlədilib. Hər əlavə icra onun "
                "«kor yoxlama» statusunu bir qədər zəiflədir — yoxlayıcı bunu "
                "nəzərə almalıdır."
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Köməkçilər
# ---------------------------------------------------------------------------


def _outcomes(run: RunArtifacts) -> dict[str, list[bool]]:
    """Case → təkrarlar üzrə keçdi/keçmədi.

    ÖLÇÜLƏ BİLMƏYƏNLƏR MƏXRƏCDƏN ÇIXIR. `judge_error` və `sut_error` sistemin
    cavabı haqqında heç nə demir; onları `False` saymaq bir 429-u və ya
    timeout-u McNemar cədvəlində GERİLƏMƏ kimi göstərirdi — hesabatın baş
    sətri «−1 case variantda GERİLƏYİB» yazırdı, halbuki şəbəkə problemi idi.
    `pass_rate` bu qaydanı `UNMEASURABLE` ilə artıq tətbiq edir; müqayisə də
    eyni qaydanı tətbiq etməlidir, yoxsa iki rəqəm bir-birinə zidd olur.
    """
    out: dict[str, list[bool]] = {}
    for cause in run.causes:
        if cause.category in UNMEASURABLE:
            continue
        out.setdefault(cause.case_id, []).append(cause.category == "ok")
    return {case_id: results for case_id, results in out.items() if results}


def _escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()
