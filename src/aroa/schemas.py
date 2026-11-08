"""Structured outputs. Every model call that decides something returns one of these, validated by Pydantic."""

from typing import Literal

from pydantic import BaseModel, Field

Route = Literal["chat", "documents", "alarms", "web", "project", "review"]


class Routing(BaseModel):
    route: Route = Field(
        description="chat: conversation, questions the assistant can answer from memory or general knowledge; "
        "documents: questions about the user's indexed documents; alarms: set, list or remove reminders; "
        "web: needs current information from the internet; project: read or change files in a project folder."
    )
    search_query: str = Field(
        description="Standalone search query in the user's language for memory or documents. "
        "Resolve pronouns and references to earlier turns."
    )
    project: str = Field(default="", description="Project name when route is project, else empty.")
    policy: Literal["allowed", "blocked"] = Field(
        description="blocked when the request asks for something the firm's policy forbids (see the policy)."
    )
    policy_reason: str = Field(default="", description="One sentence for the user when blocked, else empty.")


class Grading(BaseModel):
    relevant: list[int] = Field(description="Numbers of the fragments that help answer the question.")


class Quote(BaseModel):
    source: int = Field(description="Fragment number.")
    text: str = Field(description="A sentence copied verbatim from that fragment.")


class DocumentAnswer(BaseModel):
    answer: str = Field(description="Answer in Markdown, citing fragments as [1], [2]...")
    quotes: list[Quote] = Field(description="One verbatim quote for each cited fragment.")
    coverage: Literal["full", "partial", "none"] = Field(
        description="full: the fragments answer the question. partial: they answer part of it and the answer "
        "says what is missing. none: they do not answer it at all."
    )


ClaimKind = Literal[
    "file_written", "file_read", "command_run", "url_read", "search_done", "alarm_created", "alarm_deleted", "other"
]


class Claim(BaseModel):
    kind: ClaimKind = Field(description="What the assistant says it did.")
    target: str = Field(description="The file path, URL, command, alarm text or action named in the claim.")


class Claims(BaseModel):
    claims: list[Claim] = Field(description="Every action the reply states as done. Empty if it states none.")


class Facts(BaseModel):
    facts: list[str] = Field(
        description="Durable facts about the user, their preferences, people, projects or decisions, "
        "one short sentence each in the user's language. Empty if the turn contains none."
    )


class PolicyReview(BaseModel):
    verdict: Literal["allowed", "blocked"] = Field(description="Final verdict after reading the policy again.")
    reason: str = Field(default="", description="One sentence naming the forbidden action when blocked.")


class FindingItem(BaseModel):
    kind: str = Field(description="One of the kinds listed in the instructions; 'otro' if none fits.")
    statement: str = Field(
        description="The finding in one sentence, in the user's language, with the figures it needs."
    )
    quote: str = Field(description="A verbatim quote from the fragment that proves the statement, copied exactly.")


class Findings(BaseModel):
    findings: list[FindingItem] = Field(
        default_factory=list, description="Empty when the fragment has nothing for the goal."
    )
