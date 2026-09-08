# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""End-to-end tests for how graph-side interrupt behaviour reaches the client.

Everything here drives a real graph through :class:`ResponsesHostServer`, and
is grouped into four sections:

* **Scopes** — *where* ``interrupt()`` is raised (inside a tool, a subgraph).
* **Multiplicity** — parallel pauses in one superstep vs. several pauses in
  one node.
* **Rules** — the constraints the docs place on graph authors. Hosting cannot
  enforce these, so each test pins the observable consequence of breaking (or
  respecting) one, which is what makes the resulting support question
  diagnosable as a graph bug rather than a host bug.
* **Patterns** — the HITL shapes customers actually build: approve/reject,
  review and edit, validate and re-prompt.

All references are to
https://docs.langchain.com/oss/python/langgraph/interrupts
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("azure.ai.agentserver.responses")
pytest.importorskip("starlette")

from langchain_core.messages import AIMessage

from langchain_azure_ai.agents.hosting import ResponsesHostServer
from langchain_azure_ai.agents.hosting._converters import (
    HITL_FUNCTION_NAME,
)

from .conftest import (
    REAL_INTERRUPT_ASYNC_XFAIL,
    ScriptRegistrar,
    approval_requests,
    assistant_text,
    client_for,
    interrupt_value,
    resume_item,
    sentinels,
)
from .graphs import (
    build_approval_routing_graph,
    build_parallel_empty_update_interrupt_graph,
    build_parallel_interrupt_graph,
    build_reordering_interrupt_graph,
    build_reprompt_graph,
    build_review_graph,
    build_sequential_interrupt_graph,
    build_side_effect_graph,
    build_skipping_interrupt_graph,
    build_static_breakpoint_graph,
    build_subgraph_interrupt_graph,
    build_swallowed_interrupt_graph,
    build_tool_interrupt_graph,
    build_uncheckpointed_interrupt_graph,
    build_unserializable_value_graph,
)

# ---------------------------------------------------------------------------
# Scopes — where interrupt() is allowed to be raised
# ---------------------------------------------------------------------------


class TestInterruptInsideATool:
    """https://docs.langchain.com/oss/python/langgraph/interrupts#interrupts-in-tools"""

    def test_surfaces_and_resumes_a_tool_pause(self, script: ScriptRegistrar) -> None:
        """``interrupt()`` inside a ``@tool`` must reach the client, and the
        resume payload must be handed back to the tool so it can act on the
        (possibly edited) arguments."""
        key = "hitl-tool"
        script(
            key,
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_send_1",
                            "name": "send_email",
                            "args": {
                                "to": "alice@example.com",
                                "subject": "Meeting",
                            },
                        }
                    ],
                ),
                AIMessage(content="All done."),
            ],
        )
        host = ResponsesHostServer(build_tool_interrupt_graph(key))
        conversation_id = "conv-tool-interrupt"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={
                    "input": "email alice about the meeting",
                    "conversation": {"id": conversation_id},
                },
            )
            assert first.status_code == 200, first.text
            pending = sentinels(first.json())
            assert len(pending) == 1, first.json()
            # Structured payloads survive the envelope round-trip.
            assert interrupt_value(pending[0]) == {
                "action": "send_email",
                "to": "alice@example.com",
                "subject": "Meeting",
            }

            # Approve, editing the recipient on the way through.
            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [
                        resume_item(
                            pending[0]["call_id"],
                            {"action": "approve", "to": "ops@example.com"},
                        )
                    ],
                },
            )
            assert second.status_code == 200, second.text
            payload = second.json()
            assert payload["status"] == "completed", payload
            assert not sentinels(payload), payload
            tool_outputs = [
                item["output"]
                for item in payload["output"]
                if item.get("type") == "function_call_output"
            ]
            assert any(
                "Email sent to ops@example.com" in str(out) for out in tool_outputs
            ), payload


class TestInterruptInsideASubgraph:
    """https://docs.langchain.com/oss/python/langgraph/interrupts#using-with-subgraphs-called-as-functions"""

    @REAL_INTERRUPT_ASYNC_XFAIL
    def test_surfaces_and_resumes_a_subgraph_pause(self) -> None:
        """An interrupt raised in a nested subgraph bubbles to the parent
        checkpoint, so the host surfaces and resumes it like any other pause."""
        host = ResponsesHostServer(build_subgraph_interrupt_graph())
        conversation_id = "conv-subgraph"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "go", "conversation": {"id": conversation_id}},
            )
            assert first.status_code == 200, first.text
            pending = sentinels(first.json())
            assert len(pending) == 1, first.json()
            assert interrupt_value(pending[0]) == "sub-question"

            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(pending[0]["call_id"], "Ada")],
                },
            )
            assert second.status_code == 200, second.text
            payload = second.json()
            assert payload["status"] == "completed", payload
            assert not sentinels(payload), payload
            assert "sub:Ada" in assistant_text(payload), payload


# ---------------------------------------------------------------------------
# Multiplicity — more than one pause at a time
# ---------------------------------------------------------------------------


class TestParallelInterrupts:
    """https://docs.langchain.com/oss/python/langgraph/interrupts#handling-multiple-interrupts"""

    @REAL_INTERRUPT_ASYNC_XFAIL
    def test_emits_and_resumes_parallel_interrupts(self) -> None:
        """Two branches pause at once → one sentinel pair each, resumable together.

        LangGraph rejects a bare resume value while several interrupts are
        pending ("you must specify the interrupt id when resuming"), so the
        host must fold both answers into an id-keyed resume map.
        """
        host = ResponsesHostServer(build_parallel_interrupt_graph())
        conversation_id = "conv-parallel"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "go", "conversation": {"id": conversation_id}},
            )
            assert first.status_code == 200, first.text
            first_payload = first.json()
            pending = sentinels(first_payload)
            assert len(pending) == 2, first_payload
            call_ids = {interrupt_value(it): it["call_id"] for it in pending}
            assert set(call_ids) == {"question_a", "question_b"}
            # Distinct LangGraph ids — otherwise answers could not be routed.
            assert len(set(call_ids.values())) == 2

            approvals = approval_requests(first_payload)
            assert len(approvals) == 2, first_payload
            assert len({it["id"] for it in approvals}) == 2

            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [
                        resume_item(call_ids["question_a"], "A"),
                        resume_item(call_ids["question_b"], "B"),
                    ],
                },
            )
            assert second.status_code == 200, second.text
            payload = second.json()
            assert payload["status"] == "completed", payload
            assert not sentinels(payload), payload
            text = assistant_text(payload)
            assert "a=A" in text and "b=B" in text, payload

    @REAL_INTERRUPT_ASYNC_XFAIL
    def test_stays_paused_until_every_branch_is_answered(self) -> None:
        """Answering one of two parallel pauses must not resume the other.

        The branch that was answered runs to completion, the other stays
        suspended, and the turn comes back with exactly *one* sentinel — the
        outstanding one. Re-emitting the answered question too would leave
        the client unable to tell what is still owed, and answering it again
        is a no-op (see the duplicate-answer test), so the conversation would
        dead-end.
        """
        host = ResponsesHostServer(build_parallel_interrupt_graph())
        conversation_id = "conv-parallel-partial"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "go", "conversation": {"id": conversation_id}},
            )
            assert first.status_code == 200, first.text
            call_ids = {
                interrupt_value(it): it["call_id"] for it in sentinels(first.json())
            }
            assert set(call_ids) == {"question_a", "question_b"}, first.json()

            # Answer only branch a.
            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(call_ids["question_a"], "A")],
                },
            )
            assert second.status_code == 200, second.text
            second_payload = second.json()
            second_sentinels = sentinels(second_payload)
            assert len(second_sentinels) == 1, second_payload
            assert interrupt_value(second_sentinels[0]) == "question_b"
            # Same interrupt, so the client may reuse either id it was given.
            assert second_sentinels[0]["call_id"] == call_ids["question_b"]
            # Branch a already committed its output while b stays suspended.
            assert "a=A" in assistant_text(second_payload), second_payload
            assert "b=" not in assistant_text(second_payload), second_payload

            third = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(call_ids["question_b"], "B")],
                },
            )
            assert third.status_code == 200, third.text
            third_payload = third.json()
            assert third_payload["status"] == "completed", third_payload
            assert not sentinels(third_payload), third_payload
            text = assistant_text(third_payload)
            assert "a=A" in text and "b=B" in text, third_payload

    @REAL_INTERRUPT_ASYNC_XFAIL
    def test_does_not_reemit_answered_branch_that_returns_empty_update(self) -> None:
        host = ResponsesHostServer(build_parallel_empty_update_interrupt_graph())
        conversation_id = "conv-parallel-empty-update"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "go", "conversation": {"id": conversation_id}},
            )
            assert first.status_code == 200, first.text
            call_ids = {
                interrupt_value(item): item["call_id"]
                for item in sentinels(first.json())
            }
            assert set(call_ids) == {"question_a", "question_b"}

            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(call_ids["question_a"], "A")],
                },
            )
            assert second.status_code == 200, second.text
            second_sentinels = sentinels(second.json())
            assert len(second_sentinels) == 1, second.text
            assert interrupt_value(second_sentinels[0]) == "question_b"

            third = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(call_ids["question_b"], "B")],
                },
            )
            assert third.status_code == 200, third.text
            assert third.json()["status"] == "completed", third.text
            assert not sentinels(third.json()), third.text

    @REAL_INTERRUPT_ASYNC_XFAIL
    def test_ignores_a_repeated_answer_to_the_same_interrupt(self) -> None:
        """Answering an already-resolved interrupt is a no-op, not a rewrite.

        A client that retries — a dropped connection, an impatient user, a
        buggy loop — must not be able to overwrite an answer the graph has
        already consumed, nor knock the still-outstanding branch loose. The
        stale id no longer matches anything pending, so the host falls back
        to re-emitting the outstanding sentinel and the first answer stands.
        """
        host = ResponsesHostServer(build_parallel_interrupt_graph())
        conversation_id = "conv-parallel-duplicate"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "go", "conversation": {"id": conversation_id}},
            )
            assert first.status_code == 200, first.text
            call_ids = {
                interrupt_value(it): it["call_id"] for it in sentinels(first.json())
            }

            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(call_ids["question_a"], "A1")],
                },
            )
            assert second.status_code == 200, second.text
            assert "a=A1" in assistant_text(second.json()), second.text

            # Same interrupt, different answer, after it was already consumed.
            third = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(call_ids["question_a"], "A2")],
                },
            )
            assert third.status_code == 200, third.text
            third_payload = third.json()
            # The outstanding pause is re-emitted, unchanged and still alone.
            third_sentinels = sentinels(third_payload)
            assert len(third_sentinels) == 1, third_payload
            assert interrupt_value(third_sentinels[0]) == "question_b"
            # The graph was never advanced, so the turn carries no new output.
            assert assistant_text(third_payload) == "", third_payload

            fourth = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(call_ids["question_b"], "B")],
                },
            )
            assert fourth.status_code == 200, fourth.text
            fourth_payload = fourth.json()
            assert fourth_payload["status"] == "completed", fourth_payload
            assert not sentinels(fourth_payload), fourth_payload
            text = assistant_text(fourth_payload)
            # The first answer stood; the retry never reached the graph.
            assert "a=A1" in text and "b=B" in text, fourth_payload
            assert "A2" not in text, fourth_payload


class TestSequentialInterrupts:
    """https://docs.langchain.com/oss/python/langgraph/interrupts#do-not-reorder-interrupt-calls-within-a-node"""

    @REAL_INTERRUPT_ASYNC_XFAIL
    def test_walks_sequential_interrupts_in_one_node(self) -> None:
        """Several ``interrupt()`` calls in one node surface one pause at a time.

        LangGraph stores resume values per *task* and matches them to
        ``interrupt()`` calls strictly by index, so both pauses share the
        same ``call_id``. That is a real constraint on clients: within a
        node, the id does not identify *which question* is being asked — only
        the ``value`` envelope does. Pinning it here stops anyone from
        "fixing" the host to mint a fresh id per pause, which would break
        resume matching.
        """
        host = ResponsesHostServer(build_sequential_interrupt_graph())
        conversation_id = "conv-sequential"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "go", "conversation": {"id": conversation_id}},
            )
            assert first.status_code == 200, first.text
            first_sentinels = sentinels(first.json())
            assert len(first_sentinels) == 1, first.json()
            assert interrupt_value(first_sentinels[0]) == "name?"
            call_id = first_sentinels[0]["call_id"]

            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(call_id, "Ada")],
                },
            )
            assert second.status_code == 200, second.text
            second_payload = second.json()
            second_sentinels = sentinels(second_payload)
            assert len(second_sentinels) == 1, second_payload
            assert interrupt_value(second_sentinels[0]) == "city?"
            # Same task → same interrupt id, even though it is a new question.
            assert second_sentinels[0]["call_id"] == call_id, second_payload

            third = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(call_id, "Paris")],
                },
            )
            assert third.status_code == 200, third.text
            third_payload = third.json()
            assert third_payload["status"] == "completed", third_payload
            assert not sentinels(third_payload), third_payload
            # Both stored answers replayed into the right slots.
            assert "Ada@Paris" in assistant_text(third_payload), third_payload

    @REAL_INTERRUPT_ASYNC_XFAIL
    def test_misroutes_answers_when_a_node_skips_an_interrupt(self) -> None:
        """Skipping an ``interrupt()`` on replay silently misbinds answers.

        Resume values are matched by index, so dropping the second question
        shifts everything after it up one slot: the answer given for "age?"
        is handed back as the answer to "city?", and "city?" is never asked.
        The host has no way to detect or repair this — it faithfully relays
        whatever the graph does — so the test exists to make the corruption
        reproducible and attributable to the graph, not to hosting.
        """
        flags = {"ask_age": True}
        host = ResponsesHostServer(build_skipping_interrupt_graph(flags))
        conversation_id = "conv-skip"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "go", "conversation": {"id": conversation_id}},
            )
            assert first.status_code == 200, first.text
            pending = sentinels(first.json())
            assert len(pending) == 1, first.json()
            assert interrupt_value(pending[0]) == "name?"
            call_id = pending[0]["call_id"]

            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(call_id, "Ada")],
                },
            )
            assert second.status_code == 200, second.text
            second_sentinels = sentinels(second.json())
            assert len(second_sentinels) == 1, second.json()
            assert interrupt_value(second_sentinels[0]) == "age?"

            # The branch condition flips before the node replays.
            flags["ask_age"] = False
            third = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(second_sentinels[0]["call_id"], "30")],
                },
            )
            assert third.status_code == 200, third.text
            third_payload = third.json()
            assert third_payload["status"] == "completed", third_payload
            # "city?" was never asked...
            assert not sentinels(third_payload), third_payload
            # ...and the age answer landed in the city slot.
            assert "Ada@30" in assistant_text(third_payload), third_payload

    @REAL_INTERRUPT_ASYNC_XFAIL
    def test_swaps_answers_when_a_node_reorders_interrupts(self) -> None:
        """Reordering ``interrupt()`` calls on replay transposes the answers.

        The sibling case to skipping, and the one the docs rule is named
        after. Skipping *shifts* every later answer up a slot; reordering
        keeps the count intact and instead binds each stored answer to the
        wrong question. Nothing raises and no sentinel is re-emitted, so the
        turn looks perfectly healthy — which is exactly why it needs pinning:
        the only evidence is the transposed text in the final message.
        """
        flags = {"reversed": False}
        host = ResponsesHostServer(build_reordering_interrupt_graph(flags))
        conversation_id = "conv-reorder"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "go", "conversation": {"id": conversation_id}},
            )
            assert first.status_code == 200, first.text
            pending = sentinels(first.json())
            assert len(pending) == 1, first.json()
            assert interrupt_value(pending[0]) == "name?"
            call_id = pending[0]["call_id"]

            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(call_id, "Ada")],
                },
            )
            assert second.status_code == 200, second.text
            second_sentinels = sentinels(second.json())
            assert len(second_sentinels) == 1, second.json()
            assert interrupt_value(second_sentinels[0]) == "city?"

            # The node starts asking in the opposite order before it replays.
            flags["reversed"] = True
            third = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(second_sentinels[0]["call_id"], "Paris")],
                },
            )
            assert third.status_code == 200, third.text
            third_payload = third.json()
            assert third_payload["status"] == "completed", third_payload
            assert not sentinels(third_payload), third_payload
            # Index 0 ("Ada") now feeds "city?" and index 1 ("Paris") feeds
            # "name?" — the answers come back transposed.
            assert "Paris@Ada" in assistant_text(third_payload), third_payload


# ---------------------------------------------------------------------------
# Rules — constraints hosting relays but cannot enforce
# ---------------------------------------------------------------------------


class TestTryExceptAroundInterrupt:
    """https://docs.langchain.com/oss/python/langgraph/interrupts#do-not-wrap-interrupt-calls-in-try%2Fexcept"""

    def test_emits_nothing_when_a_node_swallows_the_interrupt(self) -> None:
        """Regression guard for the most common HITL support question.

        A bare ``except Exception`` around ``interrupt()`` catches either the
        ``GraphInterrupt`` LangGraph uses to suspend or, on Python 3.10, the
        ``RuntimeError`` raised when the runnable context is unavailable.
        Nothing is checkpointed and the host has no pause to surface. The turn
        completes normally with no sentinel, making the failure diagnosable as
        a graph bug rather than a host bug.
        """
        host = ResponsesHostServer(build_swallowed_interrupt_graph())
        with client_for(host) as client:
            resp = client.post(
                "/responses",
                json={"input": "hi", "conversation": {"id": "conv-swallow"}},
            )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["status"] == "completed", payload
        assert not sentinels(payload), payload
        assert not approval_requests(payload), payload
        expected_exception = (
            "RuntimeError" if sys.version_info < (3, 11) else "GraphInterrupt"
        )
        assert f"swallowed:{expected_exception}" in assistant_text(payload), payload


class TestIdempotentSideEffects:
    """https://docs.langchain.com/oss/python/langgraph/interrupts#side-effects-called-before-interrupt-must-be-idempotent"""

    @REAL_INTERRUPT_ASYNC_XFAIL
    def test_resume_replays_the_node_from_its_start(self) -> None:
        """Resuming re-runs the whole node, not just the line after ``interrupt``.

        This is why the docs require pre-interrupt side effects to be
        idempotent. Hosting must not paper over it — a customer debugging
        duplicated writes needs the replay to be observable.
        """
        trace: list[str] = []
        host = ResponsesHostServer(build_side_effect_graph(trace))
        conversation_id = "conv-replay"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "go", "conversation": {"id": conversation_id}},
            )
            assert first.status_code == 200, first.text
            pending = sentinels(first.json())
            assert len(pending) == 1, first.json()
            # Paused before the post-interrupt work ever ran.
            assert trace == ["before"]

            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(pending[0]["call_id"], "yes")],
                },
            )
            assert second.status_code == 200, second.text
            assert "done:yes" in assistant_text(second.json()), second.text
        assert trace == ["before", "before", "after"]


class TestComplexInterruptValues:
    """https://docs.langchain.com/oss/python/langgraph/interrupts#do-not-return-complex-values-in-interrupt-calls"""

    @REAL_INTERRUPT_ASYNC_XFAIL
    def test_degrades_gracefully_for_non_json_interrupt_values(self) -> None:
        """A non-JSON-serializable payload must not take the turn down.

        The docs tell graph authors to keep interrupt payloads simple, but
        hosting cannot enforce that. The envelope falls back to ``str()`` so
        the client still gets a routable sentinel and resume keeps working —
        degraded rendering instead of a failed response.
        """
        host = ResponsesHostServer(build_unserializable_value_graph())
        conversation_id = "conv-unserializable"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "go", "conversation": {"id": conversation_id}},
            )
            assert first.status_code == 200, first.text
            first_payload = first.json()
            pending = sentinels(first_payload)
            assert len(pending) == 1, first_payload
            # Arguments stay valid JSON; the value degrades to its repr.
            value = interrupt_value(pending[0])
            assert isinstance(value, str), first_payload
            assert "pick one" in value, first_payload
            # Both channels still agree so either can be used to resume.
            approvals = approval_requests(first_payload)
            assert len(approvals) == 1, first_payload
            assert approvals[0]["arguments"] == pending[0]["arguments"]

            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(pending[0]["call_id"], "a")],
                },
            )
            assert second.status_code == 200, second.text
            payload = second.json()
            assert payload["status"] == "completed", payload
            assert "picked:a" in assistant_text(payload), payload


class TestCheckpointerRequirement:
    """https://docs.langchain.com/oss/python/langgraph/interrupts#pause-using-interrupt"""

    def test_cannot_surface_an_interrupt_without_a_checkpointer(self) -> None:
        """No checkpointer → no persisted pause → nothing to surface or resume.

        Documents the first prerequisite in the interrupt docs: a HITL graph
        compiled without a checkpointer silently loses its pause, so the turn
        just completes.
        """
        host = ResponsesHostServer(build_uncheckpointed_interrupt_graph())
        with client_for(host) as client:
            resp = client.post(
                "/responses",
                json={"input": "hi", "conversation": {"id": "conv-no-checkpointer"}},
            )
        assert resp.status_code == 200, resp.text
        assert HITL_FUNCTION_NAME not in resp.text


class TestStaticBreakpoints:
    """https://docs.langchain.com/oss/python/langgraph/interrupts#debugging-with-interrupts"""

    def test_does_not_surface_static_interrupt_breakpoints(self) -> None:
        """``interrupt_before`` pauses the graph without producing an ``Interrupt``.

        The docs steer people away from static breakpoints for HITL; this
        pins *why* it matters for hosting: the snapshot's ``next`` is set but
        ``interrupts`` is empty, so there is nothing for the host to emit and
        the client is left with a silently truncated turn.
        """
        host = ResponsesHostServer(build_static_breakpoint_graph())
        with client_for(host) as client:
            resp = client.post(
                "/responses",
                json={"input": "hi", "conversation": {"id": "conv-static"}},
            )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert not sentinels(payload), payload
        assert not approval_requests(payload), payload
        # The node never ran, so there is no output to show either.
        assert "node-ran" not in assistant_text(payload), payload


# ---------------------------------------------------------------------------
# Patterns — the HITL shapes customers actually build
# ---------------------------------------------------------------------------


class TestApproveOrReject:
    """https://docs.langchain.com/oss/python/langgraph/interrupts#approve-or-reject"""

    @pytest.mark.parametrize(
        ("decision", "expected"),
        [(True, "status:approved"), (False, "status:rejected")],
    )
    @REAL_INTERRUPT_ASYNC_XFAIL
    def test_routes_decision_through_command_goto(
        self, decision: bool, expected: str
    ) -> None:
        """``resume=false`` is a real answer, not a missing one.

        Distinct from the ``mcp_approval_response{approve:false}`` path,
        which fails the turn: here the *graph* owns the rejection semantics,
        so a falsy resume value has to reach the node intact and route it to
        the cancel branch.
        """
        host = ResponsesHostServer(build_approval_routing_graph())
        conversation_id = f"conv-approve-{decision}"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={
                    "input": "do the thing",
                    "conversation": {"id": conversation_id},
                },
            )
            assert first.status_code == 200, first.text
            pending = sentinels(first.json())
            assert len(pending) == 1, first.json()

            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(pending[0]["call_id"], decision)],
                },
            )
            assert second.status_code == 200, second.text
            payload = second.json()
            assert payload["status"] == "completed", payload
            assert not sentinels(payload), payload
            assert expected in assistant_text(payload), payload


class TestReviewAndEditState:
    """https://docs.langchain.com/oss/python/langgraph/interrupts#review-and-edit-state"""

    @REAL_INTERRUPT_ASYNC_XFAIL
    def test_carries_state_out_and_the_edit_back_in(self) -> None:
        host = ResponsesHostServer(build_review_graph())
        conversation_id = "conv-review"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={
                    "input": "write something",
                    "conversation": {"id": conversation_id},
                },
            )
            assert first.status_code == 200, first.text
            pending = sentinels(first.json())
            assert len(pending) == 1, first.json()
            assert interrupt_value(pending[0]) == {
                "instruction": "Review and edit this content",
                "content": "Initial draft",
            }

            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(pending[0]["call_id"], "Improved draft")],
                },
            )
            assert second.status_code == 200, second.text
            payload = second.json()
            assert payload["status"] == "completed", payload
            assert not sentinels(payload), payload
            assert "Improved draft" in assistant_text(payload), payload


class TestValidateHumanInput:
    """https://docs.langchain.com/oss/python/langgraph/interrupts#validating-human-input"""

    @REAL_INTERRUPT_ASYNC_XFAIL
    def test_reemits_a_new_sentinel_when_the_graph_pauses_again(self) -> None:
        """A resume turn that itself pauses must emit a *fresh* sentinel.

        Without this the client would have no id to answer the re-prompt with,
        and the conversation would dead-end after one invalid answer.
        """
        host = ResponsesHostServer(build_reprompt_graph())
        conversation_id = "conv-reprompt"
        with client_for(host) as client:
            first = client.post(
                "/responses",
                json={"input": "start", "conversation": {"id": conversation_id}},
            )
            assert first.status_code == 200, first.text
            first_sentinels = sentinels(first.json())
            assert len(first_sentinels) == 1, first.json()
            assert interrupt_value(first_sentinels[0]) == "What is your age?"

            # Invalid answer → the node loops back and pauses again.
            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(first_sentinels[0]["call_id"], "thirty")],
                },
            )
            assert second.status_code == 200, second.text
            second_payload = second.json()
            assert second_payload["status"] == "completed", second_payload
            second_sentinels = sentinels(second_payload)
            assert len(second_sentinels) == 1, second_payload
            assert "not a valid age" in interrupt_value(second_sentinels[0])
            # A new pause is a new LangGraph interrupt, so a new call_id.
            assert second_sentinels[0]["call_id"] != first_sentinels[0]["call_id"], (
                second_payload
            )

            # Valid answer → the graph finishes.
            third = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [resume_item(second_sentinels[0]["call_id"], 30)],
                },
            )
            assert third.status_code == 200, third.text
            third_payload = third.json()
            assert third_payload["status"] == "completed", third_payload
            assert not sentinels(third_payload), third_payload
            assert "age=30" in assistant_text(third_payload), third_payload
