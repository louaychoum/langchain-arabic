"""Key-free booking graph; install examples/requirements-langgraph.txt first."""

from dataclasses import asdict
from typing import TypedDict

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph

from langchain_arabic import ArabicSpeechNormalizer


class BookingFacts(TypedDict):
    order_id: str
    date: str


class BookingState(TypedDict, total=False):
    facts: BookingFacts
    display_text: str
    speech_text: str
    changes: list[dict]
    status: str
    clarification: str


def compose_reply(state: BookingState) -> dict:
    facts = state["facts"]
    # This fake response is deterministic; no API key, booking write or TTS call.
    response = f"رقم الطلب {facts['order_id']}. الموعد {facts['date']}."
    chain = FakeListChatModel(responses=[response]) | StrOutputParser()
    return {"display_text": chain.invoke("أعطني تفاصيل الحجز")}


def prepare_speech(state: BookingState) -> dict:
    normalizer = ArabicSpeechNormalizer()
    result = normalizer.explain(state["display_text"])
    source_date = state["facts"]["date"]
    date_changes = normalizer.explain(source_date).changes
    valid_date = (
        len(date_changes) == 1
        and date_changes[0].kind == "date"
        and date_changes[0].start == 0
        and date_changes[0].end == len(source_date)
    )
    needs_clarification = not valid_date or any(
        c.kind in {"retained_date", "retained_time"} for c in result.changes
    )
    return {
        "speech_text": "" if needs_clarification else result.text,
        "changes": [asdict(c) for c in result.changes],
        "status": "needs_clarification" if needs_clarification else "ready",
        "clarification": "",
    }


def clarify_date(state: BookingState) -> dict:
    return {"clarification": "ما تاريخ الحجز بصيغة سنة-شهر-يوم؟"}


def build_graph():
    graph = StateGraph(BookingState)
    graph.add_node("compose", compose_reply)
    graph.add_node("speech", prepare_speech)
    graph.add_node("clarify", clarify_date)
    graph.add_edge(START, "compose")
    graph.add_edge("compose", "speech")
    graph.add_conditional_edges(
        "speech", lambda state: state["status"], {"ready": END, "needs_clarification": "clarify"}
    )
    graph.add_edge("clarify", END)
    return graph.compile()


def main() -> None:
    graph = build_graph()
    for booking_date in ["2026-11-10", "03/04/2026"]:
        result = graph.invoke({"facts": {"order_id": "001234", "date": booking_date}})
        print(result)
        if booking_date == "03/04/2026":
            assert result["status"] == "needs_clarification" and not result["speech_text"]
        else:
            assert result["status"] == "ready"
            assert "صفر صفر واحد اثنان ثلاثة أربعة" in result["speech_text"]


if __name__ == "__main__":
    main()
