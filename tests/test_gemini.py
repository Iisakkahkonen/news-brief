import io
import json
import urllib.error

import pytest

from news_brief import gemini
from news_brief.gemini import DEFAULT_MODEL, ENDPOINT, GeminiClient
from news_brief.model import ModelError, ModelUnavailable

KEY = "AIza-test-key-123"


class FakeResponse:
    def __init__(self, payload):
        self._raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def reply(text="hello", status="completed"):
    return {"id": "i1", "status": status, "steps": [
        {"type": "model_output", "content": [{"type": "text", "text": text}]}]}


def http_error(code, error_code="", message="boom"):
    body = json.dumps({"error": {"code": error_code, "message": message}}).encode()
    return urllib.error.HTTPError(ENDPOINT, code, "err", {}, io.BytesIO(body))


@pytest.fixture
def net(monkeypatch):
    """Script the network: a list of responses or exceptions, consumed in order."""
    class Net:
        def __init__(self):
            self.script, self.requests, self.sleeps = [], [], []

    state = Net()

    def fake_urlopen(request, timeout):
        state.requests.append(request)
        step = state.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return FakeResponse(step)

    monkeypatch.setattr(gemini.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(gemini.time, "sleep", state.sleeps.append)
    return state


def client(**kw):
    kw.setdefault("min_interval", 0)
    kw.setdefault("retry_wait", 7)
    return GeminiClient(KEY, **kw)


def test_successful_call_returns_the_text(net):
    net.script = [reply("The summary.")]
    assert client().complete("prompt") == "The summary."


def test_text_from_several_parts_and_steps_is_joined_and_other_steps_ignored(net):
    net.script = [{"status": "completed", "steps": [
        {"type": "function_call", "name": "x"},
        {"type": "model_output", "content": [
            {"type": "text", "text": '{"sentences":'}, {"type": "image"},
            {"type": "text", "text": " []}"}]},
    ]}]
    assert client().complete("p") == '{"sentences": []}'


def test_request_shape_key_in_header_only_and_nothing_stored(net):
    net.script = [reply()]
    client(model="gemini-test").complete("my prompt")

    request = net.requests[0]
    assert request.full_url == ENDPOINT and KEY not in request.full_url
    assert request.get_header("X-goog-api-key") == KEY
    assert request.get_method() == "POST"
    body = json.loads(request.data)
    assert KEY not in json.dumps(body)
    assert body["model"] == "gemini-test"
    assert body["input"] == "my prompt"
    assert body["store"] is False
    assert body["generation_config"]["max_output_tokens"] == 4096


def test_default_model_is_set():
    assert client().model == DEFAULT_MODEL


def test_blank_key_is_refused():
    with pytest.raises(ValueError):
        GeminiClient("   ")


def test_per_minute_rate_limit_is_retried_once_after_waiting(net):
    net.script = [http_error(429, "rate_limit_exceeded"), reply("ok")]
    assert client().complete("p") == "ok"
    assert len(net.requests) == 2
    assert 7 in net.sleeps


def test_daily_quota_stops_at_once_without_retry(net):
    net.script = [http_error(429, "quota_exceeded", "daily limit")]
    with pytest.raises(ModelUnavailable, match="daily quota used up"):
        client().complete("p")
    assert len(net.requests) == 1


def test_two_rate_limits_in_a_row_end_the_run_as_unavailable(net):
    net.script = [http_error(429, "too_many_requests"), http_error(429, "too_many_requests")]
    with pytest.raises(ModelUnavailable, match="429 twice"):
        client().complete("p")
    assert len(net.requests) == 2


@pytest.mark.parametrize("code", [401, 402, 403])
def test_auth_payment_and_permission_errors_are_unavailable_and_not_retried(net, code):
    net.script = [http_error(code, "denied", "no")]
    with pytest.raises(ModelUnavailable):
        client().complete("p")
    assert len(net.requests) == 1


def test_403_explains_the_free_tier_eligibility_possibility(net):
    net.script = [http_error(403, "permission_denied")]
    with pytest.raises(ModelUnavailable, match="not eligible for the free tier"):
        client().complete("p")


def test_bad_request_is_a_plain_error_without_retry(net):
    net.script = [http_error(400, "parameter_unknown", "unknown field")]
    with pytest.raises(ModelError, match="parameter_unknown") as info:
        client().complete("p")
    assert not isinstance(info.value, ModelUnavailable)
    assert len(net.requests) == 1


def test_server_error_is_retried_once_then_reported(net):
    net.script = [http_error(503, "service_unavailable"), http_error(503, "service_unavailable")]
    with pytest.raises(ModelError, match="after one retry"):
        client().complete("p")
    assert len(net.requests) == 2


def test_server_error_then_success(net):
    net.script = [http_error(500, "api_error"), reply("fine")]
    assert client().complete("p") == "fine"


def test_network_failure_is_retried_once(net):
    net.script = [urllib.error.URLError("no route"), reply("back")]
    assert client().complete("p") == "back"


def test_api_key_is_scrubbed_from_error_messages(net):
    net.script = [http_error(400, "invalid_request", f"bad key {KEY} supplied")]
    with pytest.raises(ModelError) as info:
        client().complete("p")
    assert KEY not in str(info.value)
    assert "[key removed]" in str(info.value)


@pytest.mark.parametrize("payload, message", [
    (b"not json", "not valid JSON"),
    (b"[1, 2]", "not a JSON object"),
    (reply(status="failed"), "status 'failed'"),
    (reply(status="incomplete"), "status 'incomplete'"),
    ({"status": "completed", "steps": []}, "no text"),
    ({"status": "completed"}, "no text"),
])
def test_unusable_replies_raise_model_error(net, payload, message):
    net.script = [payload]
    with pytest.raises(ModelError, match=message):
        client().complete("p")


def test_calls_are_spaced_by_the_minimum_interval(net, monkeypatch):
    clock = iter([100.0, 101.0, 101.0, 104.0])  # last_call set, then pace check, ...
    monkeypatch.setattr(gemini.time, "monotonic", lambda: next(clock))
    net.script = [reply("a"), reply("b")]
    c = client(min_interval=4.0)

    c.complete("one")
    c.complete("two")

    assert net.sleeps == [pytest.approx(3.0)]


def test_retry_is_announced_and_requests_are_counted(net):
    messages = []
    net.script = [http_error(503, "service_unavailable", "high demand"), reply("fine")]
    c = GeminiClient(KEY, min_interval=0, retry_wait=7, notify=messages.append)

    assert c.complete("p") == "fine"
    assert c.requests_made == 2
    assert "high demand" in messages[0] and "waiting 7s" in messages[0]
