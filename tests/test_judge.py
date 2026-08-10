"""LLM-as-judge — sxem, qərəz mühafizəsi, xəta təsnifatı və retry büdcəsi.

Şəbəkəyə çıxış yoxdur: `FakeAnthropicClient` skriptlənir və hər sorğunun
kwargs-larını saxlayır, yəni "hakimə nə göndərildi" sualı test edilə bilir.
"""

from __future__ import annotations

import json

import pytest

from eval.errors import JudgePermanentError, JudgeTransientError
from eval.judge import (
    JUDGE_SYSTEM,
    VERDICT_SCHEMA,
    AnthropicJudge,
    RetryBudget,
    judge_prompt_sha256,
)
from tests.conftest import (
    FakeAnthropicClient,
    FakeAnthropicResponse,
    make_case,
    make_chunk,
    make_observation,
    make_settings,
)


class FakeStatusError(Exception):
    """SDK istisnalarının duck-type qarşılığı (status_code daşıyır)."""

    def __init__(self, status_code: int, message: str = "xəta") -> None:
        super().__init__(message)
        self.status_code = status_code


def build(
    *,
    responses: list | None = None,
    **settings_overrides,
) -> tuple[AnthropicJudge, FakeAnthropicClient]:
    client = FakeAnthropicClient(responses=responses)
    judge = AnthropicJudge(
        make_settings(**settings_overrides),
        client=client,
        sleep=lambda _seconds: None,
    )
    return judge, client


def judge_case(**kwargs):
    return make_case(
        "dev_open_incident_s1",
        gradable="judge",
        rubric="Cavab hadisəyə cavab addımlarını ardıcıl və mənbəyə söykənmiş şəkildə izah edir.",
        **kwargs,
    )


def verdict_response(**overrides) -> FakeAnthropicResponse:
    payload = {
        "score": 3,
        "faithful": True,
        "complete": True,
        "reason": "bütün addımlar mənbədə var",
        "flags": [],
    }
    payload.update(overrides.pop("payload", {}))
    return FakeAnthropicResponse(text=json.dumps(payload, ensure_ascii=False), **overrides)


# --- sorğunun forması ------------------------------------------------------


def test_verdikt_JSON_dan_oxunur() -> None:
    judge, _ = build(responses=[verdict_response()])
    verdict = judge.judge(judge_case(), make_observation())

    assert verdict.score == 3
    assert verdict.faithful is True
    assert verdict.complete is True
    assert verdict.reason == "bütün addımlar mənbədə var"
    assert verdict.ok is True
    assert verdict.error == ""


def test_kecid_HEDDLE_muqayise_olunur() -> None:
    judge, _ = build(responses=[verdict_response(payload={"score": 2})])
    assert judge.judge(judge_case(), make_observation()).passed is True

    judge, _ = build(
        responses=[verdict_response(payload={"score": 2})], judge_pass_threshold=3
    )
    assert judge.judge(judge_case(), make_observation()).passed is False


def test_cixis_JSON_SXEMI_ile_mecburidir() -> None:
    """Sərbəst mətn parse etmək hakimi qeyri-müəyyən edərdi."""
    judge, client = build()
    judge.judge(judge_case(), make_observation())

    output_config = client.requests[0]["output_config"]
    assert output_config["format"]["type"] == "json_schema"
    assert output_config["format"]["schema"] == VERDICT_SCHEMA
    assert output_config["effort"] == "low"


def test_system_bloku_KESLENIR() -> None:
    """Rubrika sabitdir və hər case-də təkrarlanır — keş oxunuşu 0.1x-dir."""
    judge, client = build()
    judge.judge(judge_case(), make_observation())

    system = client.requests[0]["system"]
    assert system[0]["text"] == JUDGE_SYSTEM
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_temperature_GONDERILMIR() -> None:
    """Opus 5-də `temperature` 400 qaytarır — sükutla göndərilməməlidir."""
    judge, client = build()
    judge.judge(judge_case(), make_observation())

    request = client.requests[0]
    assert "temperature" not in request
    assert "top_p" not in request
    assert request["max_tokens"] == 1500
    assert request["model"] == "claude-opus-5"


def test_hakim_CAVAB_VERQESINI_gormur() -> None:
    """Gözlənilən cavab hakimin sorğusuna DÜŞMƏMƏLİDİR.

    Düşsəydi, "hakim düzgün tanıdı" nəticəsi heç nə sübut etməzdi — hakim
    sadəcə verilmiş cavabı təkrarlayardı.
    """
    case = judge_case(contains=["atlas-cache-edge"], numeric=[{"value": 128, "unit": "GB"}])
    obs = make_observation(
        answer_text="Sistem paylanmış keş istifadə edir [1].",
        chunks=[make_chunk(text="Keş qatı üç zonada işləyir.")],
    )
    judge, client = build()
    judge.judge(case, obs)

    serialized = json.dumps(client.requests[0], ensure_ascii=False)
    assert "atlas-cache-edge" not in serialized
    assert "128" not in serialized


def test_rubrika_ve_ISTINAD_EDILEN_chunklar_gonderilir() -> None:
    case = judge_case()
    obs = make_observation(
        answer_text="Hadisə zamanı əvvəlcə bildiriş göndərilir [1].",
        chunks=[
            make_chunk(label=1, text="İlk addım: növbətçi komandaya bildiriş."),
            make_chunk(label=2, text="İSTİNAD EDİLMƏMİŞ BLOK"),
        ],
        cited_labels=[1],
    )
    judge, client = build()
    judge.judge(case, obs)

    user_text = client.requests[0]["messages"][0]["content"]
    assert case.expected.rubric in user_text
    assert "İlk addım" in user_text
    assert "İSTİNAD EDİLMƏMİŞ BLOK" not in user_text


# --- imtina və kəsilmə -----------------------------------------------------


def test_imtina_HECVAXT_SIFIR_BAL_deyil() -> None:
    """`refusal` hakimin ölçə bilmədiyini bildirir, cavabın pis olduğunu yox.

    0 bala çevirmək sistemin uğursuzluğunu uydurardı: hesabatdakı pass-rate
    hakimin əlçatanlığından asılı olardı.
    """
    judge, _ = build(responses=[FakeAnthropicResponse(text="", stop_reason="refusal")])
    verdict = judge.judge(judge_case(), make_observation())

    assert verdict.score is None
    assert verdict.passed is False
    assert verdict.ok is False
    assert "imtina" in verdict.error.lower()


def test_max_tokens_kesilmesi_AYDIN_xeta_verir() -> None:
    """Boş content + max_tokens = düşüncənin ortasında kəsilmə."""
    judge, _ = build(responses=[FakeAnthropicResponse(text="", stop_reason="max_tokens")])
    verdict = judge.judge(judge_case(), make_observation())

    assert verdict.ok is False
    assert "JUDGE_MAX_TOKENS" in verdict.error


def test_pozulmus_JSON_judge_error_olur() -> None:
    judge, _ = build(responses=[FakeAnthropicResponse(text="bu JSON deyil")])
    verdict = judge.judge(judge_case(), make_observation())

    assert verdict.ok is False
    assert verdict.score is None


def test_aralikdan_kenar_bal_reddedilir() -> None:
    judge, _ = build(responses=[verdict_response(payload={"score": 7})])
    verdict = judge.judge(judge_case(), make_observation())
    assert verdict.ok is False


# --- xəta təsnifatı və retry ----------------------------------------------


def test_kecici_xeta_TEKRAR_edilir() -> None:
    judge, client = build(responses=[FakeStatusError(429), verdict_response()])
    verdict = judge.judge(judge_case(), make_observation())

    assert verdict.ok is True
    assert len(client.requests) == 2


def test_daimi_xeta_TEKRAR_EDILMIR() -> None:
    """401-i təkrarlamaq yalnız vaxt və pul itkisidir."""
    judge, client = build(responses=[FakeStatusError(401)])
    verdict = judge.judge(judge_case(), make_observation())

    assert verdict.ok is False
    assert len(client.requests) == 1


def test_retry_budcesi_BUTUN_RUN_ucun_paylasilir() -> None:
    """Büdcə case başına olsaydı, şəbəkə pisləşəndə xərc case sayına vurulardı."""
    judge, client = build(
        responses=[FakeStatusError(503)] * 10,
        judge_max_retries=2,
    )
    first = judge.judge(judge_case(), make_observation("c1"))
    second = judge.judge(judge_case(), make_observation("c2"))

    assert first.ok is False and second.ok is False
    # 1-ci case: 1 cəhd + 2 təkrar = 3; 2-ci case: büdcə bitib, 1 cəhd.
    assert len(client.requests) == 4


def test_xeta_terifleri_domen_siniflerine_cevrilir() -> None:
    from eval.judge import _to_domain_error

    assert isinstance(_to_domain_error(FakeStatusError(429)), JudgeTransientError)
    assert isinstance(_to_domain_error(FakeStatusError(529)), JudgeTransientError)
    assert isinstance(_to_domain_error(FakeStatusError(400)), JudgePermanentError)
    assert isinstance(_to_domain_error(FakeStatusError(404)), JudgePermanentError)


def test_budce_ancaq_KECICI_xetada_xerclenir() -> None:
    budget = RetryBudget(2)
    judge, client = build(responses=[FakeStatusError(401)])
    judge.budget = budget
    judge.judge(judge_case(), make_observation())
    assert budget.remaining == 2


# --- uçot ------------------------------------------------------------------


def test_usage_ve_KES_tokenleri_qeyd_olunur() -> None:
    judge, _ = build(
        responses=[
            verdict_response(
                input_tokens=400,
                output_tokens=120,
                cache_read_input_tokens=900,
                cache_creation_input_tokens=50,
            )
        ]
    )
    verdict = judge.judge(judge_case(), make_observation())

    assert verdict.input_tokens == 400
    assert verdict.output_tokens == 120
    assert verdict.cached_read_tokens == 900
    assert verdict.cached_write_tokens == 50
    assert verdict.latency_ms >= 0.0


def test_verdikt_HAKIMIN_KIMLIYINI_dasiyir() -> None:
    """Hakim və ya prompt dəyişəndə köhnə verdiktlər fərqlənə bilməlidir."""
    judge, _ = build(responses=[verdict_response(model="claude-opus-5")])
    verdict = judge.judge(judge_case(), make_observation())

    assert verdict.judge_model == "claude-opus-5"
    assert verdict.served_model == "claude-opus-5"
    assert verdict.judge_prompt_sha256 == judge_prompt_sha256()
    assert len(verdict.judge_prompt_sha256) == 64


def test_fallback_model_YALNIZ_ACIQ_teyin_edilende_islenir() -> None:
    judge, client = build(
        responses=[
            FakeAnthropicResponse(text="", stop_reason="refusal"),
            verdict_response(model="claude-sonnet-5"),
        ],
        judge_fallback_model="claude-sonnet-5",
    )
    verdict = judge.judge(judge_case(), make_observation())

    assert verdict.ok is True
    assert verdict.judge_model == "claude-opus-5"
    assert verdict.served_model == "claude-sonnet-5"
    assert client.requests[1]["model"] == "claude-sonnet-5"


def test_fallback_bos_olanda_imtina_XETA_olaraq_qalir() -> None:
    """Səssiz model dəyişikliyi bütün qərəz ölçmələrini etibarsızlaşdırardı."""
    judge, client = build(responses=[FakeAnthropicResponse(text="", stop_reason="refusal")])
    verdict = judge.judge(judge_case(), make_observation())

    assert verdict.ok is False
    assert len(client.requests) == 1


def test_case_kimliyi_verdiktde_saxlanilir() -> None:
    judge, _ = build(responses=[verdict_response()])
    verdict = judge.judge(judge_case(), make_observation("dev_open_incident_s1", repeat=3))
    assert verdict.case_id == "dev_open_incident_s1"
    assert verdict.repeat == 3
