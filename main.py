from uuid import uuid4
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from pydantic import BaseModel, Field

load_dotenv()


# NEW: Mock customer purchase database
MOCK_ORDERS = {
    "ORD101": {
        "customer": "Rahul Sharma",
        "product": "Logitech M331 Wireless Mouse",
        "price": 799,
    },
    "ORD102": {
        "customer": "Priya Patil",
        "product": "Samsung Galaxy Buds FE",
        "price": 4999,
    },
    "ORD103": {
        "customer": "Amit Deshmukh",
        "product": "Dell 24-inch Monitor",
        "price": 10999,
    },
    "ORD104": {
        "customer": "Sneha Kulkarni",
        "product": "USB-C Charging Cable",
        "price": 1000,
    },
}


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
    order_id: str
    product: str
    product_price: float
    customer: str


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
                "human",
                f"Customer ticket: {state['ticket']}\n"
                f"Purchased product: {state['product']}\n"
                f"Purchase price: {state['product_price']}\n"
                "If the customer requests a full refund"
                "without specifying an amount, use the"
                "purchase price as the refund amount"
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

    amount = state['refund_amount']

    if amount > 0:
        prefix = (
            f" Hi {state['customer']}, we recieved your"
            f"refund request of INR.{amount:g} for "
            f"{state['product']} (order {state['order_id']})"
        )

        if amount > state['product_price']:
            message = (
                "The requested amount exceeds the purchase"
                f"price of INR{state['product_price']:g}"
                "Our support team will review the request"
                "before processing"
            )
        elif amount >= 1000:
            message = (
                "Your request requires manual approval. "
                "We will update you after review"
            )
        else:
            message = (
                "Your request has been recorded for processing. "
                "This does not confirm a refund has been issued."
            )
        
        return { "draft": prefix + message }


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
    amount = state['refund_amount']

    if amount > state['product_price']:
        return "review"

    return "review" if amount >= 1000 else "auto"


# 6. Human approval node
def human_review(state: TicketState):
    print("\n [3] Human review Required")

    decision = interrupt({
        "reason": "Refund requires manual review",
        "order_id": state["order_id"],
        "customer": state["customer"],
        "product": state["product"],
        "purchase_price": state["product_price"],
        "draft": state["draft"],
        "refund_amount": state["refund_amount"],
    })

    approved = decision["approved"]

    if state["refund_amount"] > state["product_price"]:
        approved = False

    if approved:
        final_response = (
            f"Hi {state['customer']}, your refund request"
            f"of INR {state['refund_amount']:g} for"
            f"{state['product']}"
            "has been approved for further processing"
            "No payment has been issued yet."
        )
    else: 
        final_response = (
            f"Hi {state['customer']}, your refund request"
            f"for {state['product']}"
            f"order {state['order_id']}"
            "was not approved and has been escalated"
            "for further review"
        )

    return {
        "approved": approved,
        "final_response": final_response
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

graph.get_graph().draw_mermaid_png(
    output_file_path="workflow.png"
)

# 9. run the workflow

def main():

    print("\n Available orders: ", list(MOCK_ORDERS))

    order_id = input("\n Enter Order ID: ").strip().upper()

    if order_id not in MOCK_ORDERS:
        print("Invalid order id")
        return
    
    order = MOCK_ORDERS[order_id]

    print(f"Customer: {order['customer']}")
    print(f"Product: {order['product']}")
    print(f"Price: INR{order['price']}")


    ticket = input("\n Enter your customer ticket:")

    config = {
        "configurable" : {
            "thread_id": str(uuid4())
        }
    }

    result = graph.invoke(
        {
            "ticket": ticket,
            "order_id": order_id,
            "customer": order['customer'],
            "product": order['product'],
            "product_price": order['price'],
        },
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