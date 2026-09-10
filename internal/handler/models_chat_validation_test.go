package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tonghaoch/copilot-proxy-go/internal/state"
)

func TestModelsIncludesSupportedEndpoints(t *testing.T) {
	appState := &state.State{}
	appState.SetModels([]state.Model{{
		ID:                 "claude-opus-4.8",
		Name:               "Claude Opus 4.8",
		Vendor:             "anthropic",
		SupportedEndpoints: []string{"/v1/messages", "/responses"},
	}})
	h := New(Dependencies{
		State:   appState,
		Metrics: responsesTestMetrics{},
		Copilot: &chatValidationCopilot{},
		Config:  responsesTestConfig{},
	})

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	recorder := httptest.NewRecorder()
	h.Models(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}

	var resp ModelsListResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}
	if len(resp.Data) != 1 {
		t.Fatalf("expected 1 model, got %d", len(resp.Data))
	}
	if resp.Data[0].ID != "claude-opus-4-8" {
		t.Fatalf("unexpected public model id: %q", resp.Data[0].ID)
	}
	if len(resp.Data[0].SupportedEndpoints) != 2 {
		t.Fatalf("expected supported endpoints in response, got %v", resp.Data[0].SupportedEndpoints)
	}
	if resp.Data[0].SupportedEndpoints[0] != "/v1/messages" || resp.Data[0].SupportedEndpoints[1] != "/responses" {
		t.Fatalf("unexpected supported endpoints: %v", resp.Data[0].SupportedEndpoints)
	}
}

func TestChatCompletionsRejectsUnsupportedEndpointWithoutUpstreamCall(t *testing.T) {
	appState := &state.State{}
	appState.SetModels([]state.Model{{
		ID:                 "gpt-5.6-terra",
		SupportedEndpoints: []string{"/responses"},
	}})
	upstream := &chatValidationCopilot{}
	h := New(Dependencies{
		State:   appState,
		Metrics: responsesTestMetrics{},
		Copilot: upstream,
		Config:  responsesTestConfig{},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(
		`{"model":"gpt-5.6-terra","messages":[{"role":"user","content":"hello"}]}`,
	))
	recorder := httptest.NewRecorder()
	h.ChatCompletions(recorder, req)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", recorder.Code, recorder.Body.String())
	}
	if !strings.Contains(recorder.Body.String(), `is not accessible via the /chat/completions endpoint`) {
		t.Fatalf("unexpected error body: %s", recorder.Body.String())
	}
	if upstream.chatCalls != 0 {
		t.Fatalf("expected no upstream chat call, got %d", upstream.chatCalls)
	}
}

func TestChatCompletionsRejectsUnavailableModelWithoutUpstreamCall(t *testing.T) {
	appState := &state.State{}
	upstream := &chatValidationCopilot{}
	h := New(Dependencies{
		State:   appState,
		Metrics: responsesTestMetrics{},
		Copilot: upstream,
		Config:  responsesTestConfig{},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(
		`{"model":"gpt-5.6-terra","messages":[{"role":"user","content":"hello"}]}`,
	))
	recorder := httptest.NewRecorder()
	h.ChatCompletions(recorder, req)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", recorder.Code, recorder.Body.String())
	}
	if !strings.Contains(recorder.Body.String(), `model \"gpt-5.6-terra\" is unavailable`) {
		t.Fatalf("unexpected error body: %s", recorder.Body.String())
	}
	if upstream.chatCalls != 0 {
		t.Fatalf("expected no upstream chat call, got %d", upstream.chatCalls)
	}
}

func TestChatCompletionsAllowsSupportedEndpointAndCallsUpstream(t *testing.T) {
	appState := &state.State{}
	appState.SetModels([]state.Model{{
		ID:                 "gpt-5.6-terra",
		SupportedEndpoints: []string{"/chat/completions"},
	}})
	upstream := &chatValidationCopilot{}
	h := New(Dependencies{
		State:   appState,
		Metrics: responsesTestMetrics{},
		Copilot: upstream,
		Config:  responsesTestConfig{},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(
		`{"model":"gpt-5.6-terra","messages":[{"role":"user","content":"hello"}]}`,
	))
	recorder := httptest.NewRecorder()
	h.ChatCompletions(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", recorder.Code, recorder.Body.String())
	}
	if upstream.chatCalls != 1 {
		t.Fatalf("expected one upstream chat call, got %d", upstream.chatCalls)
	}
}

func TestChatCompletionsAllowsUnknownEndpointSupportAndCallsUpstream(t *testing.T) {
	appState := &state.State{}
	appState.SetModels([]state.Model{{ID: "gpt-5.6-terra"}})
	upstream := &chatValidationCopilot{}
	h := New(Dependencies{
		State:   appState,
		Metrics: responsesTestMetrics{},
		Copilot: upstream,
		Config:  responsesTestConfig{},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(
		`{"model":"gpt-5.6-terra","messages":[{"role":"user","content":"hello"}]}`,
	))
	recorder := httptest.NewRecorder()
	h.ChatCompletions(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", recorder.Code, recorder.Body.String())
	}
	if upstream.chatCalls != 1 {
		t.Fatalf("expected one upstream chat call, got %d", upstream.chatCalls)
	}
}

func TestChatCompletionsResolvesPublicModelIDForValidation(t *testing.T) {
	appState := &state.State{}
	appState.SetModels([]state.Model{{
		ID:                 "claude-opus-4.8",
		SupportedEndpoints: []string{"/chat/completions"},
	}})
	upstream := &chatValidationCopilot{}
	h := New(Dependencies{
		State:   appState,
		Metrics: responsesTestMetrics{},
		Copilot: upstream,
		Config:  responsesTestConfig{},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(
		`{"model":"claude-opus-4-8","messages":[{"role":"user","content":"hello"}]}`,
	))
	recorder := httptest.NewRecorder()
	h.ChatCompletions(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", recorder.Code, recorder.Body.String())
	}
	if upstream.chatCalls != 1 {
		t.Fatalf("expected one upstream chat call, got %d", upstream.chatCalls)
	}

	var forwarded struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(upstream.body, &forwarded); err != nil {
		t.Fatalf("unmarshal forwarded body: %v", err)
	}
	if forwarded.Model != "claude-opus-4.8" {
		t.Fatalf("expected rewritten copilot model id, got %q", forwarded.Model)
	}
}

type chatValidationCopilot struct {
	chatCalls int
	body      []byte
}

func (*chatValidationCopilot) FetchModels(context.Context) ([]state.Model, error) { return nil, nil }
func (c *chatValidationCopilot) ProxyChatCompletionEx(_ context.Context, body []byte, _ bool, _ bool) (*http.Response, error) {
	c.chatCalls++
	c.body = append([]byte(nil), body...)
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     make(http.Header),
		Body: io.NopCloser(bytes.NewBufferString(
			`{"id":"chat-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`,
		)),
	}, nil
}
func (*chatValidationCopilot) ProxyMessages(context.Context, []byte, string, bool, bool) (*http.Response, error) {
	panic("unexpected ProxyMessages call")
}
func (*chatValidationCopilot) ProxyResponses(context.Context, []byte, bool, bool) (*http.Response, error) {
	panic("unexpected ProxyResponses call")
}
func (*chatValidationCopilot) ProxyEmbeddings(context.Context, []byte) (*http.Response, error) {
	panic("unexpected ProxyEmbeddings call")
}
