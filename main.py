from uuid import uuid4
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from pydantic import BaseModel, Field

load_dotenv()

# 1. llm+structured output
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

class TicketAnalysis(BaseModel):
    category: Literal[
        "billing", "technical", "general"
    ]
    priority: Literal["low", "medium", "high"]
    summary: str
    refund_amount: float = Field(
        description="requested refund amount, or 0"
    )

classifier = llm.with_structured_output(TicketAnalysis)


# 2. Graph State
class TicketState(TypedDict, total=False):
    ticket: str
    category: str
    priority: str
    summary: str
    refund_amount: float
    draft: str
    final_response: str
    approved: bool


# 3. Classification node
def classify_ticket(state: TicketState):
    print("\n 1] Analyzing customer ticket...")

    result = classifier.invoke(
        [
            (
                "system",
                "Classify the customer support ticket"
                "Extract only explicitly requested refunds"
                "user 0 when no amount is stated"
                "do not invent facts"
            ),
            (
                "human", state["ticket"]
            ),
        ]
    )

    print(f"Category: {result.category}")
    print(f"Priority: {result.priority}")
    print(f"Refund Amount: {result.refund_amount}")

    return {
        "category": result.category,
        "priority": result.priority,
        "summary": result.summary,
        "refund_amount": result.refund_amount,
    }


# 4. Draft Response node
def draft_response(state: TicketState):
    print("\n 2] Drafting response...")

    response = llm.invoke(
        [
            (
                "system",
                "You are a helpful customer support agent."
                "Draft a short, professional reply."
                "Do not promise a refund, payment, "
                "or confirmed resolution. "
                "Do not invent accoutn details."
            ),
            (
                "human",
                f"Ticket: {state['ticket']}\n"
                f"Summary: {state['summary']}\n"
                f"Category: {state['category']}"
            )
        ]
    )

    return {"draft": str(response.content)}


# 5. Conditional routing
def route_ticket(state: TicketState):
    high_risk = (
        state["priority"] == "high"
        or state["refund_amount"] > 1000
    )

    return "review" if high_risk else "auto"


# 6. Human approval node
def human_review(state: TicketState):
    print("\n [3] Human review Required")

    decision = interrupt({
        "reason": "High-risk customer ticket",
        "ticket": state["ticket"],
        "draft": state["draft"],
        "refund_amount": state["refund_amount"],
    })

    approved = decision["approved"]

    return {
        "approved": approved,
        "final_response": (
            state["draft"]
            if approved
            else "Your request has been escalated"
                 "to our support team for manual review. "
        )
    }


# 7. Auto response node
def auto_respond(state: TicketState):
    print("\n [3] Low-risk ticket. Auto-approved")

    return {
        "approved": True,
        "final_response": state["draft"],
    }


# 8. Build the graph

builder = StateGraph(TicketState)

builder.add_node("classify", classify_ticket)
builder.add_node("draft", draft_response)
builder.add_node("human_review", human_review)
builder.add_node("auto_respond", auto_respond)

builder.add_edge(START, "classify")
builder.add_edge("classify", "draft")

builder.add_conditional_edges(
    "draft",
    route_ticket,
    {
        "review": "human_review",
        "auto": "auto_respond",
    },
)

builder.add_edge("human_review", END)
builder.add_edge("auto_respond", END)

graph = builder.compile(
    checkpointer = InMemorySaver()
)


# 9. run the workflow

def main():

    ticket = input("\n Enter your customer ticket:")

    config = {
        "configurable" : {
            "thread_id": str(uuid4())
        }
    }

    result = graph.invoke(
        {"ticket": ticket},
        config=config,
    )

    if "__interrupt__" in result:
        print("\n ----- HUMAN REVIEW------")
        print(result["__interrupt__"][0].value)

        choice = input(
            "\n Approve response? (yes/no)"
        ).strip().lower()

        result = graph.invoke(
            Command(
                resume={"approved": choice == "yes"}
            ),
            config=config,
        )
    
    print("\n --- final result----")
    print(result["final_response"])


if __name__ == "__main__":
    main()