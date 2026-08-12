"""Konfiqurasiya, açar mühafizəsi və qiymət cədvəli.

AÇARIN QORUNMASI — ÜÇ MEXANİZM VƏ BİRİNİN MƏHDUDİYYƏTİ
-------------------------------------------------------
1. `.gitignore` `.env`-i tutur, `.env.example`-i buraxır.
2. `field(repr=False)` açarı `repr(settings)`-dən çıxarır — traceback-lərdə
   və `print()`-də görünmür.
3. `require_*_key()` placeholder dəyəri tanıyır və 3 addımlı düzəliş mətni
   ilə `ConfigError` atır; OpenAI/Anthropic-in anlaşılmaz 401-i əvəzinə.

MƏHDUDİYYƏT VƏ ONUN CAVABI: `field(repr=False)` YALNIZ `repr()`-i qoruyur.
`dataclasses.asdict(settings)` açarı hələ də qaytarır. Bu layihə konfiqi
HƏQİQƏTƏN diskə (run manifestinə) yazdığı üçün bu nəzəri risk deyil, real
mina idi. Cavab: serializasiyanın YEGANƏ yolu `public_dict()`-dir və o,
sahələri ağ siyahı ilə seçir — yəni gələcəkdə əlavə olunan hər hansı yeni
gizli sahə də default olaraq DIŞARIDA qalır. `tests/test_config.py`
manifestin açar sızdırmadığını kilidləyir.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import load_dotenv

from .errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _repo_relative(path: Path) -> str:
    """Repo daxilindəki yolu nisbi, kənardakını mütləq şəkildə qaytarır."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)

# override=False: real mühit dəyişəni `.env`-i üstələyir. CI-də və
# istehsalda açar mühitdən gəlir, `.env` yalnız lokal rahatlıq üçündür.
load_dotenv(PROJECT_ROOT / ".env", override=False)

OPENAI_PLACEHOLDER = "sk-buraya-oz-acarinizi-yazin"
ANTHROPIC_PLACEHOLDER = "sk-ant-buraya-oz-acarinizi-yazin"

VALID_EFFORTS = ("low", "medium", "high", "xhigh", "max")
VALID_GROUNDING_MODES = ("strict", "llm")


# ---------------------------------------------------------------------------
# Qiymət cədvəli
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPrice:
    """Bir modelin milyon token başına qiyməti.

    `cache_read_multiplier` və `cache_write_multiplier` Anthropic prompt
    keşi üçündür: oxu ~0.1x, yazı ~1.25x (5 dəqiqəlik TTL). Bunları
    nəzərə almamaq ilk çağırışda xərci AZ, qalan çağırışlarda ÇOX göstərir.
    """

    provider: str
    input_per_mtok: float
    output_per_mtok: float
    cache_read_multiplier: float = 1.0
    cache_write_multiplier: float = 1.0
    verified: bool = False

    def cost_usd(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_read_tokens: int = 0,
        cached_write_tokens: int = 0,
    ) -> float:
        """USD xərci.

        `input_tokens` KEŞLƏNMƏMİŞ qalıqdır — Anthropic `usage.input_tokens`
        məhz bunu qaytarır, keş oxunuşu/yazısı ayrı sahələrdədir. Onları
        toplamaq ikiqat sayma olardı.
        """
        total = (
            input_tokens * self.input_per_mtok
            + cached_read_tokens * self.input_per_mtok * self.cache_read_multiplier
            + cached_write_tokens * self.input_per_mtok * self.cache_write_multiplier
            + output_tokens * self.output_per_mtok
        )
        return total / 1_000_000


@dataclass(frozen=True)
class PriceTable:
    version: str
    as_of: str
    models: Mapping[str, ModelPrice]

    def price_for(self, model: str) -> ModelPrice:
        """Naməlum model SƏRT xətadır.

        Səssiz $0.00 daha pisdir: hesabat "xərc 0.31$" yazar, halbuki
        modellərin yarısı hesaba düşməyib və bunu heç kim görməz.
        """
        try:
            return self.models[model]
        except KeyError:
            known = ", ".join(sorted(self.models)) or "(cədvəl boşdur)"
            raise ConfigError(
                f"'{model}' modeli qiymət cədvəlində yoxdur.\n"
                f"Tanınan modellər: {known}\n"
                "Həlli: data/prices.yaml faylına həmin modelin sətrini əlavə edin.\n"
                "Səssiz $0.00 qaytarmırıq — hesabatdakı xərc rəqəmi yanlış olardı."
            ) from None

    @property
    def unverified(self) -> tuple[str, ...]:
        """`verified: false` işarəli modellər — hesabatda banner çıxır."""
        return tuple(sorted(n for n, p in self.models.items() if not p.verified))


def load_prices(path: Path) -> PriceTable:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models_raw = raw.get("models") or {}
    if not isinstance(models_raw, dict):
        raise ConfigError(f"{path}: 'models' sözlük olmalıdır.")

    models: dict[str, ModelPrice] = {}
    for name, spec in models_raw.items():
        if not isinstance(spec, dict):
            raise ConfigError(f"{path}: '{name}' üçün qeyd sözlük olmalıdır.")
        try:
            models[str(name)] = ModelPrice(
                provider=str(spec["provider"]),
                input_per_mtok=float(spec["input_per_mtok"]),
                output_per_mtok=float(spec["output_per_mtok"]),
                cache_read_multiplier=float(spec.get("cache_read_multiplier", 1.0)),
                cache_write_multiplier=float(spec.get("cache_write_multiplier", 1.0)),
                verified=bool(spec.get("verified", False)),
            )
        except KeyError as exc:
            raise ConfigError(f"{path}: '{name}' üçün {exc} sahəsi yoxdur.") from None

    return PriceTable(
        version=str(raw.get("version", "naməlum")),
        as_of=str(raw.get("as_of", "naməlum")),
        models=models,
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name} tam ədəd olmalıdır, '{raw}' verilib.") from None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(f"{name} ədəd olmalıdır, '{raw}' verilib.") from None


# Kontekstə düşən chunk sayının yuxarı həddi. Korpus 19-54 chunk-dır, yəni
# k > 50 retrieval deyil, korpusun tamamını prompta tökmək deməkdir: pullu
# run-da prompt xərcini səssizcə qat-qat artırar və «retrieval ölçdük»
# iddiasını mənasız edər.
RETRIEVAL_TOP_K_MAX = 50


def _sonlu(ad: str, deyer: float) -> float:
    """NaN/Inf BURADA kəsilir — üç qat mina olduğu üçün.

    1. NaN ilə HƏR müqayisə False verir: `bal >= astana` heç vaxt doğru
       olmur, yəni qapı SƏSSİZCƏ hər şeyi rədd edir və nəticə «sistem
       imtina edir» kimi oxunur — halbuki bu, konfiqurasiya səhvidir.
    2. `json.dumps` onu çılpaq `NaN` yazır. Bu, JSON DEYİL (RFC 8259);
       commit edilmiş manifest standart parserlə oxunmur.
    3. `config_hash` həmin yararsız payload üzərindən hesablanır.
    """
    if not math.isfinite(deyer):
        raise ConfigError(
            f"{ad} sonlu ədəd olmalıdır, verilib: {deyer}.\n"
            "NaN/Inf qapını səssizcə tam imtinaya çevirir və manifesti "
            "etibarsız JSON edir."
        )
    return deyer


def _sifir_bir_araliginda(ad: str, deyer: float) -> None:
    """Kosinus oxşarlığı şkalası — məsafə deyil.

    Səhv şkalada verilmiş rəqəm (məsələn 1.8) SUT-da səssizcə «heç nə
    keçmir» davranışına çevrilir və bütün case-lər imtina ilə bitər; bunu
    run bitdikdən sonra pass-rate düşməsi kimi oxumaq asan səhvdir.
    """
    _sonlu(ad, deyer)
    if not 0.0 <= deyer <= 1.0:
        raise ConfigError(
            f"{ad} kosinus oxşarlığı şkalasındadır (0-1), verilib: {deyer}."
        )


def yoxla_retrieval(
    *,
    top_k: int | None = None,
    threshold: float | None = None,
    soft_floor_margin: float | None = None,
    lexical_threshold: float | None = None,
    prefiks: str = "RETRIEVAL_",
) -> None:
    """Retrieval hədlərinin YEGANƏ tərifi.

    `tools/` də bunu çağırır — və bu, təkrarçılıq deyil, məqsəddir: hədlər
    yalnız pullu yolda tətbiq olunsaydı, parametrlərin əslində araşdırıldığı
    ucuz yol (sweep) yoxlamasız qalardı.

    SUT-un öz `validate()`-i ehtiyat qapı DEYİL: orada `soft_floor_margin`
    yoxlaması eyni birtərəfli kor nöqtəyə malikdir və `top_k` yuxarı həddi
    yoxdur — yəni bizim buraxdığımızı o da buraxır.
    """
    if top_k is not None:
        if top_k <= 0:
            raise ConfigError(f"{prefiks}TOP_K müsbət olmalıdır, verilib: {top_k}.")
        if top_k > RETRIEVAL_TOP_K_MAX:
            raise ConfigError(
                f"{prefiks}TOP_K ən çox {RETRIEVAL_TOP_K_MAX} ola bilər, "
                f"verilib: {top_k}. Bundan yuxarısı korpusun tamamını prompta "
                "tökür və pullu run-ın xərcini səssizcə artırır."
            )
    if threshold is not None:
        _sifir_bir_araliginda(f"{prefiks}THRESHOLD", threshold)
    if lexical_threshold is not None:
        _sifir_bir_araliginda(f"{prefiks}LEXICAL_THRESHOLD", lexical_threshold)
    if soft_floor_margin is not None:
        # YUXARI HƏDD NİYƏ VACİBDİR: marja EYNİ 0-1 şkalasından çıxılır.
        # `marja >= astana` yumşaq həddi 0-a endirir, yəni dense qapı tamam
        # sönür və yalnız leksik dəstək qalır. Bu, qanuni konfiqurasiyadır —
        # amma SEÇİLMƏLİDİR, yazı səhvi ilə əldə edilməməlidir.
        _sifir_bir_araliginda(f"{prefiks}SOFT_FLOOR_MARGIN", soft_floor_margin)


def _env_opt_int(name: str) -> int | None:
    """Təyin olunmayıbsa `None` — «SUT-un öz defaultu qalsın» deməkdir.

    Burada default rəqəm qoymaq olmazdı: 4 yazsaydıq, SUT-un defaultu
    gələcəkdə dəyişəndə biz onu səssizcə 4-də dondurardıq və bunu heç bir
    manifest göstərməzdi.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name} tam ədəd olmalıdır, '{raw}' verilib.") from None


def _env_opt_float(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(f"{name} ədəd olmalıdır, '{raw}' verilib.") from None


def _resolve(path_like: str) -> Path:
    """Nisbi yolu layihə kökünə görə həll edir."""
    p = Path(path_like).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


@dataclass(frozen=True)
class Settings:
    """Framework konfiqurasiyası — dəyişməz, hash-lənən, açar-sızdırmayan."""

    openai_api_key: str = field(repr=False)
    anthropic_api_key: str = field(repr=False)

    sut_path: Path
    sut_commit: str
    grounding_mode: str

    generator_model: str
    embedding_model: str

    judge_model: str
    judge_effort: str
    judge_max_tokens: int
    judge_pass_threshold: int
    judge_timeout: float
    judge_max_retries: int
    judge_retry_base_delay: float
    judge_fallback_model: str

    dataset_path: Path
    split_manifest_path: Path
    prices_path: Path
    variants_dir: Path
    runs_dir: Path
    logs_dir: Path

    repeats: int

    # --- Retrieval override-ları (CP7) ---------------------------------------
    # `None` = SUT-un öz dəyəri qalır. Bunlar SUT konfiqurasiyasıdır, prompt
    # deyil: `eval/variants.py` mesaj səviyyəsində işləyir və retrieval qatına
    # çata bilmir. SUT faylı YENƏ DƏ redaktə olunmur — `rag.config.Settings.load`
    # onsuz da `**overrides` qəbul edir, biz həmin mövcud giriş nöqtəsindən
    # istifadə edirik, ona görə pin edilmiş commit iddiası pozulmur.
    retrieval_top_k: int | None = None
    retrieval_threshold: float | None = None
    retrieval_soft_floor_margin: float | None = None

    @classmethod
    def load(cls, **overrides: Any) -> "Settings":
        base = cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
            sut_path=_resolve(_env_str("SUT_PATH", "vendor/week2-rag-document-qa")),
            sut_commit=_env_str("SUT_COMMIT", ""),
            grounding_mode=_env_str("GROUNDING_MODE", "strict"),
            generator_model=_env_str("GENERATOR_MODEL", "gpt-4o-mini"),
            embedding_model=_env_str("EMBEDDING_MODEL", "text-embedding-3-small"),
            judge_model=_env_str("JUDGE_MODEL", "claude-opus-5"),
            judge_effort=_env_str("JUDGE_EFFORT", "low"),
            judge_max_tokens=_env_int("JUDGE_MAX_TOKENS", 4000),
            judge_pass_threshold=_env_int("JUDGE_PASS_THRESHOLD", 2),
            judge_timeout=_env_float("JUDGE_TIMEOUT", 90.0),
            judge_max_retries=_env_int("JUDGE_MAX_RETRIES", 4),
            judge_retry_base_delay=_env_float("JUDGE_RETRY_BASE_DELAY", 1.0),
            judge_fallback_model=_env_str("JUDGE_FALLBACK_MODEL", ""),
            dataset_path=_resolve(_env_str("DATASET_PATH", "data/testset.yaml")),
            split_manifest_path=_resolve(
                _env_str("SPLIT_MANIFEST_PATH", "data/split_manifest.json")
            ),
            prices_path=_resolve(_env_str("PRICES_PATH", "data/prices.yaml")),
            variants_dir=_resolve(_env_str("VARIANTS_DIR", "data/variants")),
            runs_dir=_resolve(_env_str("RUNS_DIR", "runs")),
            logs_dir=_resolve(_env_str("LOGS_DIR", "logs")),
            repeats=_env_int("REPEATS", 1),
            retrieval_top_k=_env_opt_int("RETRIEVAL_TOP_K"),
            retrieval_threshold=_env_opt_float("RETRIEVAL_THRESHOLD"),
            retrieval_soft_floor_margin=_env_opt_float("RETRIEVAL_SOFT_FLOOR_MARGIN"),
        )
        settings = base if not overrides else _replace(base, **overrides)
        settings.validate()
        return settings

    # -- yoxlamalar --------------------------------------------------------

    def validate(self) -> None:
        if self.judge_effort not in VALID_EFFORTS:
            raise ConfigError(
                f"JUDGE_EFFORT '{self.judge_effort}' tanınmır. "
                f"İcazə verilənlər: {', '.join(VALID_EFFORTS)}."
            )
        if self.grounding_mode not in VALID_GROUNDING_MODES:
            raise ConfigError(
                f"GROUNDING_MODE '{self.grounding_mode}' tanınmır. "
                f"İcazə verilənlər: {', '.join(VALID_GROUNDING_MODES)}."
            )
        # 1024 aşağı hədd təsadüfi deyil: Opus 5-də `max_tokens` düşüncəni DƏ
        # əhatə edir və düşüncə default açıqdır, yəni kiçik dəyər düşüncənin
        # ortasında kəsir və BOŞ content-li `stop_reason: max_tokens` verir.
        if self.judge_max_tokens < 1024:
            raise ConfigError(
                f"JUDGE_MAX_TOKENS={self.judge_max_tokens} çox kiçikdir.\n"
                "Anthropic-də max_tokens düşüncə + mətn cəmini məhdudlaşdırır və "
                "düşüncə default olaraq açıqdır. Kiçik dəyər düşüncənin ortasında "
                "kəsir və boş cavab qaytarır. Ən azı 1024, tövsiyə 4000."
            )
        # NİYƏ DÜŞÜNCƏNİ SÖNDÜRMÜRÜK
        #   `thinking: {"type": "disabled"}` effort `low`-da icazəlidir və
        #   büdcəni azaldardı. Amma söndürülmüş düşüncə Opus 5-də cavab
        #   mətninə `<thinking>` teqlərinin sızmasına səbəb ola bilir — bu
        #   hakim üçün ölümcüldür, çünki cavab JSON kimi parse olunur.
        #   Ona görə düşüncə açıq qalır və əvəzinə tavan qaldırılır:
        #   max_tokens tavandır, rezerv deyil — istifadə olunmayan token
        #   pul tələb etmir, yəni bu dəyişiklik xərci artırmır.
        if self.judge_fallback_model:
            # Fallback modelin qiyməti yoxlanılmasa, o, YALNIZ həqiqətən
            # işə düşdükdə — yəni pul xərcləndikdən SONRA, hesabat
            # mərhələsində — `ConfigError` ilə partlayırdı.
            prices = load_prices(self.prices_path)
            try:
                prices.price_for(self.judge_fallback_model)
            except ConfigError as exc:
                raise ConfigError(
                    f"JUDGE_FALLBACK_MODEL='{self.judge_fallback_model}' "
                    f"qiymət cədvəlində yoxdur ({self.prices_path}). "
                    "Fallback yalnız əsl nasazlıqda işə düşür; onu indi "
                    "yoxlamasaq, xəta run bitdikdən SONRA hesabatda çıxar "
                    f"və xərc artıq çəkilmiş olar. Detal: {exc}"
                ) from exc
        if not 0 <= self.judge_pass_threshold <= 3:
            raise ConfigError("JUDGE_PASS_THRESHOLD 0-3 aralığında olmalıdır.")
        if self.repeats < 1:
            raise ConfigError("REPEATS ən azı 1 olmalıdır.")
        if self.judge_max_retries < 0:
            raise ConfigError("JUDGE_MAX_RETRIES mənfi ola bilməz.")
        yoxla_retrieval(
            top_k=self.retrieval_top_k,
            threshold=self.retrieval_threshold,
            soft_floor_margin=self.retrieval_soft_floor_margin,
        )

    def retrieval_overrides(self) -> dict[str, Any]:
        """`rag.config.Settings.load(**overrides)`-a ötürüləcək sahələr.

        Təyin olunmayan parametr sözlüyə DÜŞMÜR — belədə SUT öz dəyərini
        işlədir və biz onu təxmin etmirik.
        """
        pairs = (
            ("top_k", self.retrieval_top_k),
            ("relevance_threshold", self.retrieval_threshold),
            ("soft_floor_margin", self.retrieval_soft_floor_margin),
        )
        return {name: value for name, value in pairs if value is not None}

    def require_openai_key(self) -> str:
        return _require_key(
            self.openai_api_key,
            placeholder=OPENAI_PLACEHOLDER,
            env_name="OPENAI_API_KEY",
            url="https://platform.openai.com/api-keys",
        )

    def require_anthropic_key(self) -> str:
        return _require_key(
            self.anthropic_api_key,
            placeholder=ANTHROPIC_PLACEHOLDER,
            env_name="ANTHROPIC_API_KEY",
            url="https://console.anthropic.com/settings/keys",
        )

    # -- serializasiya -----------------------------------------------------

    def public_dict(self) -> dict[str, Any]:
        """Manifestə yazıla bilən görüntü.

        Ağ siyahı ilə qurulur: `asdict()` kimi "hər şeyi götür, gizliləri
        çıxar" yanaşması gələcəkdə əlavə olunan sahəni səssizcə sızdırardı.
        """
        return {
            # REPO-YA NİSBİ. Mütləq yol iki problem yaradırdı:
            #   1. `config_hash` maşından asılı olurdu — eyni konfiqurasiya
            #      iki kompüterdə fərqli hash verirdi və «konfiqurasiya
            #      eynidirmi?» yoxlaması yalandan uğursuz olurdu;
            #   2. manifest istifadəçi adını və lokal qovluq quruluşunu
            #      repo-ya yazırdı.
            # Repo-dan KƏNARDAKI SUT üçün mütləq yol saxlanılır: o, həqiqətən
            # başqa konfiqurasiyadır və hash-də görünməlidir.
            "sut_path": _repo_relative(self.sut_path),
            "sut_commit": self.sut_commit,
            "grounding_mode": self.grounding_mode,
            "generator_model": self.generator_model,
            "embedding_model": self.embedding_model,
            "judge_model": self.judge_model,
            "judge_effort": self.judge_effort,
            "judge_max_tokens": self.judge_max_tokens,
            "judge_pass_threshold": self.judge_pass_threshold,
            "judge_fallback_model": self.judge_fallback_model,
            "repeats": self.repeats,
            # RETRIEVAL OVERRIDE-LARI YALNIZ TƏYİN OLUNANDA DÜŞÜR.
            #
            # Bu, yuxarıdakı «sahələri açıq sadala» qaydasının pozulması deyil:
            # təyin olunmamış parametr konfiqurasiyanın bir hissəsi DEYİL, onun
            # dəyəri `sut_commit`-in içindədir və o commit onsuz da hash-dədir.
            # Sabit `None` yazsaydıq, hash bütün KÖHNƏ run-lar üçün dəyişərdi və
            # 2026-08-10 baseline-ı ilə hər müqayisə yalandan «konfiqurasiya
            # fərqlidir» xəbərdarlığı verərdi — halbuki heç nə dəyişməyib.
            **{f"retrieval_{k}": v for k, v in sorted(self.retrieval_overrides().items())},
        }

    @property
    def config_hash(self) -> str:
        # `allow_nan=False` — hash-in özü də etibarlı JSON üzərindən
        # hesablanmalıdır. Bu, `yoxla_retrieval`-ın təkrarı deyil: dəyər ora
        # çatmayan yollarla da gələ bilər (məsələn birbaşa `Settings(...)`).
        payload = json.dumps(
            self.public_dict(), sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def live_secrets(self) -> tuple[str, ...]:
        """Redaksiya üçün canlı açar dəyərləri (boş olanlar buraxılır)."""
        return tuple(
            k
            for k in (self.openai_api_key, self.anthropic_api_key)
            if k and k not in (OPENAI_PLACEHOLDER, ANTHROPIC_PLACEHOLDER)
        )


def _replace(settings: Settings, **overrides: Any) -> Settings:
    """`dataclasses.replace` əvəzinə: naməlum sahəni aydın xəta ilə tutur."""
    known = set(settings.__dataclass_fields__)
    unknown = set(overrides) - known
    if unknown:
        raise ConfigError(f"Tanınmayan Settings sahəsi: {', '.join(sorted(unknown))}")
    values = {name: getattr(settings, name) for name in known}
    values.update(overrides)
    return Settings(**values)


def _require_key(value: str, *, placeholder: str, env_name: str, url: str) -> str:
    if not value or value == placeholder:
        raise ConfigError(
            f"{env_name} təyin edilməyib (və ya hələ də nümunə dəyəridir).\n"
            "Düzəliş — 3 addım:\n"
            "  1) cp .env.example .env\n"
            f"  2) {url} ünvanından açar yaradın\n"
            f"  3) .env faylında {env_name}= sətrini həmin açarla doldurun\n"
            ".env faylı .gitignore-dadır, açar repo-ya düşməyəcək."
        )
    return value
