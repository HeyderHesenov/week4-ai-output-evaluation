"""Metriklər — nöqtə qiyməti deyil, ölçünün ETİBARLILIĞI ilə birlikdə.

NİYƏ WILSON İNTERVALI
---------------------
«Holdout-da 8/8, yəni 100%» cümləsi 20 case-lik dəstdə yanıltıcıdır: 8
müşahidə ilə həqiqi uğur nisbətinin 63%-dən aşağı olmadığını demək olar,
daha çoxunu yox. Wilson intervalı kiçik n-də normal approksimasiyadan
üstündür və p=0 və p=1 hallarında da mənalı sərhəd verir (Wald intervalı
orada eni SIFIR göstərir — sistematik yalan). Hər pass-rate rəqəmi
intervalı ilə birlikdə çap olunur.

NİYƏ MCNEMAR, İKİ FAİZİN FƏRQİ DEYİL
-------------------------------------
Baseline və variant EYNİ case-lər üzərində işləyir, yəni müşahidələr
cütlənmişdir. İki müstəqil nisbət testi bu korrelyasiyanı nəzərə almır və
gücü boş yerə itirir. McNemar yalnız FİKRİNİ DƏYİŞƏN case-lərə baxır:
baseline-da keçib variantda keçməyənlər (b) və əksi (c). 20 case-lik dəstdə
kiçik nümunə üçün dəqiq binom testi işlədilir, χ² approksimasiyası yox.

NİYƏ İKİ PASS-RATE
------------------
`judge_error` və `sut_error` sistemin cavabı haqqında heç nə demir. Onları
uğursuzluq saymaq şəbəkə problemini sistemin qüsuruna çevirir; məxrəcdən
çıxarmaq isə uğursuzluğu gizlətməyə imkan verir. Hər ikisi çap olunur:
`rate` (ölçülə bilənlər üzrə) və `strict_rate` (ölçülə bilməyənlər
uğursuzluq sayılır). Fərq böyükdürsə, rəqəmə deyil, ölçmə qatına baxmaq
lazımdır.

XƏRC HESABATINDAKI AÇIQ BOŞLUQ
-------------------------------
Embedding çağırışları SUT-un `VectorStore`-unun İÇİNDƏ baş verir və
`InstrumentedStore` onların token sayını görmür. Sorğu uzunluğundan token
təxmin etmək mümkün idi, amma uydurulmuş rəqəm hesabatı «tam» göstərərdi.
Əvəzinə boşluq `EMBEDDING_GAP_NOTE` kimi açıq çap olunur.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .config import PriceTable
from .judge import Verdict
from .observation import LlmCall, SutObservation
from .rootcause import RootCause

# Ölçülə bilməyən nəticələr — sistemin cavabı haqqında məlumat daşımırlar.
UNMEASURABLE = frozenset({"sut_error", "judge_error", "empty_index"})

EMBEDDING_GAP_NOTE = (
    "Embedding çağırışları ölçülmür: onlar SUT-un VectorStore-unun içində baş verir "
    "və token sayı sarğıya görünmür. Aşağıdakı xərc yalnız sintez/korreksiya/hakim "
    "çağırışlarını əhatə edir."
)

Z_95 = 1.959963984540054


# ---------------------------------------------------------------------------
# Nisbət və interval
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score intervalı. `total == 0` → (0.0, 1.0) — tam məlumatsızlıq."""
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    margin = (z / denominator) * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass(frozen=True)
class PassRate:
    passed: int
    measured: int
    unmeasurable: int

    @property
    def total(self) -> int:
        return self.measured + self.unmeasurable

    @property
    def rate(self) -> float:
        return self.passed / self.measured if self.measured else 0.0

    @property
    def strict_rate(self) -> float:
        """Ölçülə bilməyənlər uğursuzluq sayılır — aşağı hədd."""
        return self.passed / self.total if self.total else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.passed, self.measured)

    def describe(self) -> str:
        low, high = self.interval
        base = f"{self.passed}/{self.measured} = {self.rate:.0%} (95% CI: {low:.0%}-{high:.0%})"
        if self.unmeasurable:
            base += f"; ölçülə bilməyən: {self.unmeasurable}"
        return base


def pass_rate(causes: Sequence[RootCause]) -> PassRate:
    passed = sum(1 for c in causes if c.category == "ok")
    unmeasurable = sum(1 for c in causes if c.category in UNMEASURABLE)
    return PassRate(
        passed=passed,
        measured=len(causes) - unmeasurable,
        unmeasurable=unmeasurable,
    )


def pass_rate_by(
    causes: Sequence[RootCause],
    key: Mapping[str, str],
) -> dict[str, PassRate]:
    """`key` case_id → qrup (kateqoriya, bölgü...)."""
    groups: dict[str, list[RootCause]] = {}
    for cause in causes:
        groups.setdefault(key.get(cause.case_id, "naməlum"), []).append(cause)
    return {name: pass_rate(items) for name, items in sorted(groups.items())}


# ---------------------------------------------------------------------------
# Sabitlik (təkrarlar arası)
# ---------------------------------------------------------------------------


def unstable_cases(outcomes: Mapping[str, Sequence[bool]]) -> tuple[str, ...]:
    """Təkrarlar arasında sürüşən case-lər.

    SUT temperature=0-da belə deterministik deyil. Sürüşən case təmiz keçid
    SAYILMIR: «keçdi» dediyimiz nəticə növbəti işlətmədə dəyişəcəksə, o,
    ölçü deyil, təsadüfdür.
    """
    return tuple(
        sorted(cid for cid, results in outcomes.items() if len(set(results)) > 1)
    )


def stable_pass(outcomes: Mapping[str, Sequence[bool]]) -> dict[str, bool]:
    """Case yalnız BÜTÜN təkrarlarda keçibsə keçmiş sayılır."""
    return {cid: all(results) for cid, results in outcomes.items()}


# ---------------------------------------------------------------------------
# Cütlənmiş müqayisə
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class McNemarResult:
    improved: int  # baseline uğursuz → variant uğurlu
    regressed: int  # baseline uğurlu → variant uğursuz
    unchanged: int
    p_value: float

    @property
    def discordant(self) -> int:
        return self.improved + self.regressed

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def describe(self) -> str:
        verdict = "statistik əhəmiyyətli" if self.significant else "əhəmiyyətli DEYİL"
        return (
            f"+{self.improved} / -{self.regressed} "
            f"(dəyişməyən: {self.unchanged}), p = {self.p_value:.3f} — {verdict}"
        )


def mcnemar(
    baseline: Mapping[str, bool],
    variant: Mapping[str, bool],
) -> McNemarResult:
    """Cütlənmiş, dəqiq (binom) McNemar testi.

    χ² approksimasiyası deyil: 20 case-lik dəstdə diskordant sayı çox vaxt
    5-dən azdır və approksimasiya orada etibarsızdır.
    """
    shared = sorted(set(baseline) & set(variant))
    improved = sum(1 for cid in shared if not baseline[cid] and variant[cid])
    regressed = sum(1 for cid in shared if baseline[cid] and not variant[cid])
    unchanged = len(shared) - improved - regressed

    n = improved + regressed
    if n == 0:
        p = 1.0
    else:
        k = min(improved, regressed)
        tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
        p = min(1.0, 2 * tail)

    return McNemarResult(
        improved=improved, regressed=regressed, unchanged=unchanged, p_value=p
    )


# ---------------------------------------------------------------------------
# Hakim qərəzliliyi
# ---------------------------------------------------------------------------


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Təsadüfə düzəliş edilmiş razılıq.

    Xam razılıq faizi yanıldıcıdır: iki qiymətləndirici həmişə «3» versə,
    razılıq 100% olar və heç nə ölçülməz. Kappa təsadüfi razılığı çıxır.
    """
    if len(a) != len(b):
        raise ValueError("Etiket siyahılarının uzunluğu eyni olmalıdır.")
    n = len(a)
    if n == 0:
        return 0.0

    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    labels = set(a) | set(b)
    expected = sum(
        (sum(1 for x in a if x == k) / n) * (sum(1 for y in b if y == k) / n)
        for k in labels
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def agreement(a: Sequence[int], b: Sequence[int]) -> float:
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Sıra korrelyasiyası — bağlarda orta sıra verilir.

    Pearson yerinə sıra korrelyasiyası: bal ordinaldır (0-3), uzunluq isə
    uzun quyruqludur; Pearson-u bir uzun cavab təkbaşına sürükləyər.
    """
    if len(xs) != len(ys):
        raise ValueError("Uzunluqlar eyni olmalıdır.")
    n = len(xs)
    if n < 2:
        return 0.0
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


@dataclass(frozen=True)
class LabelSource:
    """İnsan etiketinin aid olduğu MƏHZ bir cavab.

    Etiket case_id-yə deyil, `(run_id, repeat)` ünvanına bağlanır. Səbəbi
    `judge_bias`-in docstring-indədir.
    """

    run_id: str
    repeat: int

    def key(self, case_id: str) -> tuple[str, str, int]:
        return (self.run_id, case_id, self.repeat)


@dataclass(frozen=True)
class JudgeBias:
    kappa: float
    agreement: float
    sample_size: int
    verbosity_rho: float
    verbosity_sample: int
    # Etiketi var, amma qoşalaşdırıla bilmədi — səssizcə atılmır, sadalanır.
    unmatched: tuple[str, ...] = ()
    unmeasurable: tuple[str, ...] = ()

    def describe(self) -> str:
        return (
            f"kappa = {self.kappa:.2f}, xam razılıq = {self.agreement:.0%} "
            f"(n={self.sample_size}); verbosity ρ = {self.verbosity_rho:+.2f} "
            f"(n={self.verbosity_sample})"
        )


def judge_bias(
    human_labels: Mapping[str, int],
    label_sources: Mapping[str, LabelSource],
    verdicts: Mapping[tuple[str, str, int], Verdict],
    answer_lengths: Mapping[tuple[str, str, int], int],
) -> JudgeBias:
    """Hakim ↔ insan razılığı və uzunluq qərəzi.

    NİYƏ ETİKET `(run_id, repeat)`-Ə BAĞLANIR
    -----------------------------------------
    İnsan bir case-i deyil, MƏHZ BİR CAVAB MƏTNİNİ qiymətləndirir. Etiketi
    yalnız `case_id` ilə saxlasaq, iki səhv baş verir:

    1. Holdout-da hər case bir neçə dəfə təkrarlanır. `case_id` üzrə
       qoşalaşdırma bir insan qərarını hər təkrara qarşı ayrıca sayar —
       4 qərar 12 müşahidə kimi görünər. Kappa-nın n-i şişər, güvən
       intervalı olduğundan dar çıxar.
    2. Eyni case fərqli run-larda (baseline / variant) fərqli cavab verə
       bilər. `case_id` üzrə qoşalaşdırma insanı heç görmədiyi mətnə görə
       hakimlə müqayisə edər.

    Ona görə hər etiket öz mənbə ünvanı ilə saxlanılır və məhz həmin
    verdiktlə qoşalaşır. Ünvanı tapılmayan etiket `unmatched`-ə düşür və
    ÇAĞIRAN TƏRƏF onu görünən etməlidir — səssiz atma ölçünü yalan göstərir.

    `verbosity_rho` verilmiş bütün verdiktlər üzrə hesablanır (təkrarlar
    daxil): orada sual «hakim uzun cavaba yüksək bal verirmi», yəni hər
    verdikt müstəqil müşahidədir və öz cavabının uzunluğu ilə qoşalaşır.
    """
    paired: list[tuple[int, int]] = []
    unmatched: list[str] = []
    unmeasurable: list[str] = []

    for case_id, human in human_labels.items():
        source = label_sources.get(case_id)
        if source is None:
            unmatched.append(case_id)
            continue
        verdict = verdicts.get(source.key(case_id))
        if verdict is None:
            unmatched.append(case_id)
        elif not verdict.ok or verdict.score is None:
            # İmtina/xəta 0 bal DEYİL — məxrəcdən çıxır, ayrıca sadalanır.
            unmeasurable.append(case_id)
        else:
            paired.append((human, verdict.score))

    scored = [(v.score, answer_lengths[key])
              for key, v in verdicts.items()
              if v.ok and v.score is not None and key in answer_lengths]

    human_scores = [h for h, _ in paired]
    machine = [m for _, m in paired]
    return JudgeBias(
        kappa=cohen_kappa(human_scores, machine) if paired else 0.0,
        agreement=agreement(human_scores, machine) if paired else 0.0,
        sample_size=len(paired),
        verbosity_rho=spearman([s for s, _ in scored], [l for _, l in scored]) if scored else 0.0,
        verbosity_sample=len(scored),
        unmatched=tuple(sorted(unmatched)),
        unmeasurable=tuple(sorted(unmeasurable)),
    )


# ---------------------------------------------------------------------------
# Xərc
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelCost:
    model: str
    provider: str
    calls: int
    input_tokens: int
    output_tokens: int
    cached_read_tokens: int
    cached_write_tokens: int
    usd: float
    missing_usage: int


@dataclass(frozen=True)
class CostReport:
    models: tuple[ModelCost, ...]
    unverified: tuple[str, ...]
    note: str = EMBEDDING_GAP_NOTE

    @property
    def total_usd(self) -> float:
        return sum(m.usd for m in self.models)

    @property
    def missing_usage(self) -> int:
        return sum(m.missing_usage for m in self.models)

    def per_case_usd(self, cases: int) -> float:
        return self.total_usd / cases if cases else 0.0


def cost_report(
    llm_calls: Sequence[LlmCall],
    verdicts: Sequence[Verdict],
    prices: PriceTable,
) -> CostReport:
    """Model üzrə xərc. Naməlum model SƏRT xətadır (`PriceTable.price_for`)."""
    buckets: dict[str, dict[str, int]] = {}

    def add(model: str, *, inp: int, out: int, cr: int, cw: int, missing: bool) -> None:
        b = buckets.setdefault(
            model,
            {"calls": 0, "input": 0, "output": 0, "cr": 0, "cw": 0, "missing": 0},
        )
        b["calls"] += 1
        b["input"] += inp
        b["output"] += out
        b["cr"] += cr
        b["cw"] += cw
        b["missing"] += int(missing)

    for call in llm_calls:
        add(
            call.model,
            inp=call.input_tokens,
            out=call.output_tokens,
            cr=call.cached_read_tokens,
            cw=0,
            missing=call.usage_source == "missing",
        )

    for verdict in verdicts:
        add(
            verdict.served_model,
            inp=verdict.input_tokens,
            out=verdict.output_tokens,
            cr=verdict.cached_read_tokens,
            cw=verdict.cached_write_tokens,
            missing=False,
        )

    models = []
    for name, b in sorted(buckets.items()):
        price = prices.price_for(name)
        models.append(
            ModelCost(
                model=name,
                provider=price.provider,
                calls=b["calls"],
                input_tokens=b["input"],
                output_tokens=b["output"],
                cached_read_tokens=b["cr"],
                cached_write_tokens=b["cw"],
                usd=price.cost_usd(b["input"], b["output"], b["cr"], b["cw"]),
                missing_usage=b["missing"],
            )
        )

    return CostReport(models=tuple(models), unverified=prices.unverified)


# ---------------------------------------------------------------------------
# Gecikmə
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyReport:
    total_ms: float
    llm_ms: float
    retrieval_ms: float
    overhead_ms: float
    observations: int

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.observations if self.observations else 0.0

    @property
    def unreconciled_ms(self) -> float:
        """Cəmin bağlanmadığı hissə. 0-dan fərqlidirsə, ölçmə boşluğu var."""
        return self.total_ms - (self.llm_ms + self.retrieval_ms + self.overhead_ms)

    def share(self, part: float) -> float:
        return part / self.total_ms if self.total_ms else 0.0


def latency_report(observations: Sequence[SutObservation]) -> LatencyReport:
    """Üç sütun yan-yana — cəmin bağlandığı GÖRÜNSÜN."""
    return LatencyReport(
        total_ms=sum(o.total_ms for o in observations),
        llm_ms=sum(o.llm_ms for o in observations),
        retrieval_ms=sum(o.retrieval_ms for o in observations),
        overhead_ms=sum(o.overhead_ms for o in observations),
        observations=len(observations),
    )


def percentile(values: Sequence[float], q: float) -> float:
    """Xətti interpolyasiya ilə persentil (q: 0.0-1.0)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
