"""TVShowAgents — a LangGraph multi-agent system for TV renewal decisions.

Pipeline: five tool-calling analyst agents (performance, audience, market,
financial, critical) gather live data from the TMDB API → a renew advocate
and a cancel advocate debate the evidence → a programming director
recommends → a network executive makes the final call
(RENEW / RENEW_CONDITIONAL / FINAL_SEASON / CANCEL).

Demonstrates: typed LangGraph state, conditional routing on tool calls,
ToolNode usage, cheap-vs-deep model tiering (Haiku for analysts, Sonnet for
decisions), and structured multi-agent debate.

Ported from the original Colab notebook (UT Dallas FIN 6327 group project).

Usage:
    export ANTHROPIC_API_KEY=...
    export TMDB_API_KEY=...      # TMDB read access token
    python tvshow_agents.py "Stranger Things"
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "deep_think_llm": "claude-sonnet-4-5",
    "quick_think_llm": "claude-3-5-haiku-latest",
    "max_debate_rounds": 1,
    "renew_threshold": 70,
    "conditional_threshold": 50,
    "final_season_threshold": 30,
    "analyst_weights": {
        "performance": 0.30,
        "audience": 0.25,
        "market": 0.20,
        "financial": 0.15,
        "critical": 0.10,
    },
}

# ---------------------------------------------------------------------------
# TMDB helpers
# ---------------------------------------------------------------------------

TMDB_BASE_URL = "https://api.themoviedb.org/3"


def get_tmdb_headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ.get('TMDB_API_KEY')}",
        "accept": "application/json",
    }


def tmdb_request(endpoint: str, params: dict | None = None) -> dict:
    url = f"{TMDB_BASE_URL}{endpoint}"
    try:
        response = requests.get(url, headers=get_tmdb_headers(), params=params or {})
        response.raise_for_status()
        time.sleep(0.25)  # be polite to the API
        return response.json()
    except Exception as exc:  # noqa: BLE001
        print(f"TMDB Error: {exc}")
        return {}


def search_tv_show(show_name: str) -> Optional[int]:
    data = tmdb_request("/search/tv", {"query": show_name})
    results = data.get("results", [])
    return results[0]["id"] if results else None


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class RenewalDebateState(TypedDict):
    renew_history: str
    cancel_history: str
    history: str
    current_response: str
    judge_decision: str
    count: int


class AgentState(MessagesState):
    show_id: str
    show_title: str
    tmdb_id: int
    network: str
    genre: str
    current_season: int
    episodes_aired: int
    production_cost_per_episode: float
    showrunner: str
    cast: List[str]
    sender: str
    performance_report: str
    audience_report: str
    market_report: str
    financial_report: str
    critical_report: str
    renewal_debate_state: RenewalDebateState
    renewal_recommendation: str
    decision_agent_plan: str
    final_decision: str


# ---------------------------------------------------------------------------
# TMDB data tools
# ---------------------------------------------------------------------------


@tool
def get_tmdb_show_details(tmdb_id: int) -> Dict[str, Any]:
    """Get comprehensive TV show details from TMDB."""
    data = tmdb_request(f"/tv/{tmdb_id}")
    if not data:
        return {"error": "Could not fetch show details"}
    return {
        "name": data.get("name"),
        "vote_average": data.get("vote_average", 0),
        "vote_count": data.get("vote_count", 0),
        "popularity": data.get("popularity", 0),
        "number_of_seasons": data.get("number_of_seasons", 0),
        "number_of_episodes": data.get("number_of_episodes", 0),
        "status": data.get("status", "Unknown"),
        "networks": [n["name"] for n in data.get("networks", [])],
        "genres": [g["name"] for g in data.get("genres", [])],
    }


@tool
def get_tmdb_season_ratings(tmdb_id: int, num_seasons: int) -> Dict[str, Any]:
    """Get ratings for each season to analyze trends."""
    season_ratings = []
    for season_num in range(1, min(num_seasons + 1, 11)):
        data = tmdb_request(f"/tv/{tmdb_id}/season/{season_num}")
        if data:
            season_ratings.append({
                "season": season_num,
                "vote_average": data.get("vote_average", 0),
                "episode_count": len(data.get("episodes", [])),
            })
    if len(season_ratings) >= 2:
        first = season_ratings[0]["vote_average"]
        last = season_ratings[-1]["vote_average"]
        trend = ("rising" if last > first + 0.5
                 else "declining" if last < first - 0.5 else "stable")
    else:
        trend = "insufficient_data"
    return {
        "season_ratings": season_ratings,
        "rating_trend": trend,
        "average_rating": (sum(s["vote_average"] for s in season_ratings)
                           / len(season_ratings)) if season_ratings else 0,
    }


@tool
def get_tmdb_reviews(tmdb_id: int) -> Dict[str, Any]:
    """Get user reviews for sentiment analysis."""
    data = tmdb_request(f"/tv/{tmdb_id}/reviews")
    reviews = data.get("results", [])
    if not reviews:
        return {"review_count": 0, "sentiment": "no_reviews", "sample_reviews": []}
    positive = sum(1 for r in reviews
                   if (r.get("author_details", {}).get("rating") or 5) >= 7)
    negative = sum(1 for r in reviews
                   if (r.get("author_details", {}).get("rating") or 5) < 5)
    sentiment = ("mostly_positive" if positive > negative
                 else "mostly_negative" if negative > positive else "mixed")
    return {
        "review_count": len(reviews),
        "sentiment": sentiment,
        "positive_count": positive,
        "negative_count": negative,
        "sample_reviews": [{
            "author": r.get("author"),
            "rating": r.get("author_details", {}).get("rating"),
            "content_preview": r.get("content", "")[:200] + "...",
        } for r in reviews[:3]],
    }


@tool
def get_tmdb_similar_shows(tmdb_id: int) -> Dict[str, Any]:
    """Get similar shows to analyze competitive landscape."""
    data = tmdb_request(f"/tv/{tmdb_id}/similar")
    similar = data.get("results", [])[:10]
    comparable_shows = [{
        "name": show.get("name"),
        "vote_average": show.get("vote_average", 0),
        "popularity": show.get("popularity", 0),
    } for show in similar]
    if comparable_shows:
        avg_rating = sum(s["vote_average"] for s in comparable_shows) / len(comparable_shows)
        avg_popularity = sum(s["popularity"] for s in comparable_shows) / len(comparable_shows)
    else:
        avg_rating = avg_popularity = 0
    return {
        "similar_shows": comparable_shows,
        "genre_avg_rating": round(avg_rating, 2),
        "genre_avg_popularity": round(avg_popularity, 2),
    }


@tool
def get_tmdb_recommendations(tmdb_id: int) -> Dict[str, Any]:
    """Get recommendation strength."""
    data = tmdb_request(f"/tv/{tmdb_id}/recommendations")
    recommendations = data.get("results", [])
    return {
        "recommendation_count": len(recommendations),
        "recommendation_strength": ("high" if len(recommendations) > 15
                                    else "medium" if len(recommendations) > 5
                                    else "low"),
    }


# ---------------------------------------------------------------------------
# Analyst agents (tool-calling)
# ---------------------------------------------------------------------------


def _analyst_node(llm, tools, system_message: str, human_message: str, report_key: str):
    """Shared factory: build an analyst node that may call tools."""
    def node(state):
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message.format(**state)),
            ("human", human_message),
        ])
        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke({})
        report = result.content if result.content else ""
        return {"messages": [result], report_key: report}
    return node


def create_performance_analyst(llm):
    return _analyst_node(
        llm, [get_tmdb_show_details, get_tmdb_season_ratings],
        system_message=(
            "You are a TV performance analyst evaluating '{show_title}' "
            "(TMDB ID: {tmdb_id}).\n"
            "Use the available tools to gather data on:\n"
            "1) Overall ratings and vote counts\n"
            "2) Season-by-season rating trends\n"
            "3) Current popularity score\n"
            "Analyze the data and provide a performance score from 0-100 "
            "with your reasoning."
        ),
        human_message=("Analyze this show's performance using the available tools. "
                       "After gathering data, provide your score and analysis."),
        report_key="performance_report",
    )


def create_audience_analyst(llm):
    return _analyst_node(
        llm, [get_tmdb_reviews, get_tmdb_recommendations],
        system_message=(
            "You are an audience analyst evaluating '{show_title}' "
            "(TMDB ID: {tmdb_id}).\n"
            "Use the available tools to gather data on:\n"
            "1) User reviews and sentiment\n"
            "2) Recommendation strength and algorithm favorability\n"
            "Analyze the data and provide an audience engagement score "
            "from 0-100 with your reasoning."
        ),
        human_message=("Analyze this show's audience reception using the available "
                       "tools. After gathering data, provide your score and analysis."),
        report_key="audience_report",
    )


def create_market_analyst(llm):
    return _analyst_node(
        llm, [get_tmdb_similar_shows],
        system_message=(
            "You are a market analyst evaluating '{show_title}' "
            "(TMDB ID: {tmdb_id}).\n"
            "Use the available tools to gather data on:\n"
            "1) Similar shows in the same genre\n"
            "2) How this show compares to genre averages\n"
            "Analyze the competitive landscape and provide a market position "
            "score from 0-100 with your reasoning."
        ),
        human_message=("Analyze this show's market position using the available "
                       "tools. After gathering data, provide your score and analysis."),
        report_key="market_report",
    )


def create_financial_analyst(llm):
    return _analyst_node(
        llm, [get_tmdb_show_details],
        system_message=(
            "You are a financial analyst evaluating '{show_title}' "
            "(TMDB ID: {tmdb_id}).\n"
            "Production cost per episode: ${production_cost_per_episode}M\n"
            "Episodes aired: {episodes_aired}\n"
            "Use the available tools to get popularity and rating data. "
            "Calculate ROI proxy using: (Popularity x Rating) / (Cost per episode)\n"
            "Provide a financial viability score from 0-100 with your reasoning."
        ),
        human_message=("Analyze this show's financial performance using the available "
                       "tools. After gathering data, provide your score and analysis."),
        report_key="financial_report",
    )


def create_critical_analyst(llm):
    return _analyst_node(
        llm, [get_tmdb_show_details, get_tmdb_reviews],
        system_message=(
            "You are a critical analyst evaluating '{show_title}' "
            "(TMDB ID: {tmdb_id}).\n"
            "Use the available tools to assess:\n"
            "1) Critical acclaim (rating quality vs quantity)\n"
            "2) Prestige value and cultural impact potential\n"
            "3) Review sentiment from critics\n"
            "Provide a critical value score from 0-100 with your reasoning."
        ),
        human_message=("Analyze this show's critical reception using the available "
                       "tools. After gathering data, provide your score and analysis."),
        report_key="critical_report",
    )


# ---------------------------------------------------------------------------
# Debate + decision agents
# ---------------------------------------------------------------------------


def create_renew_advocate(llm):
    def renew_advocate_node(state):
        debate_state = state["renewal_debate_state"]
        prompt = f"""Argue why '{state['show_title']}' should be RENEWED.

Evidence available:
Performance: {state['performance_report']}
Audience: {state['audience_report']}
Market: {state['market_report']}

Make your strongest case for renewal:"""
        response = llm.invoke(prompt)
        new_debate_state = {
            "renew_history": debate_state.get("renew_history", "") + "\n" + response.content,
            "cancel_history": debate_state.get("cancel_history", ""),
            "history": debate_state.get("history", "") + f"\n[RENEW ADVOCATE]: {response.content}",
            "current_response": response.content,
            "judge_decision": "",
            "count": debate_state.get("count", 0) + 1,
        }
        return {"renewal_debate_state": new_debate_state}
    return renew_advocate_node


def create_cancel_advocate(llm):
    def cancel_advocate_node(state):
        debate_state = state["renewal_debate_state"]
        prompt = f"""Argue why '{state['show_title']}' should be CANCELED or given a FINAL SEASON.

Evidence available:
Financial: {state['financial_report']}
Critical: {state['critical_report']}
Market: {state['market_report']}

Make your strongest case for cancellation or ending:"""
        response = llm.invoke(prompt)
        new_debate_state = {
            "renew_history": debate_state.get("renew_history", ""),
            "cancel_history": debate_state.get("cancel_history", "") + "\n" + response.content,
            "history": debate_state.get("history", "") + f"\n[CANCEL ADVOCATE]: {response.content}",
            "current_response": response.content,
            "judge_decision": "",
            "count": debate_state.get("count", 0) + 1,
        }
        return {"renewal_debate_state": new_debate_state}
    return cancel_advocate_node


def create_programming_director(llm):
    def programming_director_node(state):
        debate_state = state["renewal_debate_state"]
        prompt = f"""You are the Programming Director. Review all evidence and make a recommendation.

ANALYST REPORTS:
Performance: {state['performance_report']}
Audience: {state['audience_report']}
Market: {state['market_report']}
Financial: {state['financial_report']}
Critical: {state['critical_report']}

DEBATE SUMMARY:
{debate_state.get('history', '')}

Based on all evidence, recommend ONE of these decisions:
- RENEW (strong performance across metrics)
- RENEW_CONDITIONAL (renew with specific requirements)
- FINAL_SEASON (one more season to conclude)
- CANCEL (end immediately)

Your recommendation:"""
        response = llm.invoke(prompt)
        new_debate_state = {**debate_state, "judge_decision": response.content}
        return {"renewal_debate_state": new_debate_state,
                "renewal_recommendation": response.content}
    return programming_director_node


def create_network_executive(llm):
    def network_executive_node(state):
        prompt = f"""You are the Network Executive making the FINAL DECISION on '{state['show_title']}'.

PROGRAMMING DIRECTOR'S RECOMMENDATION:
{state['renewal_recommendation']}

Make your final decision. Format your response with the decision in bold:
**RENEW** / **RENEW_CONDITIONAL** / **FINAL_SEASON** / **CANCEL**

Your final decision and reasoning:"""
        response = llm.invoke(prompt)
        return {"decision_agent_plan": response.content,
                "final_decision": response.content}
    return network_executive_node


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def _route_on_tool_calls(state):
    """Conditional edge: route to tool node if the last message called tools."""
    messages = state.get("messages", [])
    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        return "call_tools"
    return "continue"


def build_tvshow_agents_graph(config: dict):
    quick_llm = ChatAnthropic(model=config["quick_think_llm"])
    deep_llm = ChatAnthropic(model=config["deep_think_llm"])

    analysts = [
        ("performance", create_performance_analyst(quick_llm),
         ToolNode([get_tmdb_show_details, get_tmdb_season_ratings])),
        ("audience", create_audience_analyst(quick_llm),
         ToolNode([get_tmdb_reviews, get_tmdb_recommendations])),
        ("market", create_market_analyst(quick_llm),
         ToolNode([get_tmdb_similar_shows])),
        ("financial", create_financial_analyst(quick_llm),
         ToolNode([get_tmdb_show_details])),
        ("critical", create_critical_analyst(quick_llm),
         ToolNode([get_tmdb_show_details, get_tmdb_reviews])),
    ]

    workflow = StateGraph(AgentState)

    for name, node, tool_node in analysts:
        workflow.add_node(f"{name}_analyst", node)
        workflow.add_node(f"{name}_tools", tool_node)
    workflow.add_node("renew_advocate", create_renew_advocate(quick_llm))
    workflow.add_node("cancel_advocate", create_cancel_advocate(quick_llm))
    workflow.add_node("programming_director", create_programming_director(deep_llm))
    workflow.add_node("network_executive", create_network_executive(deep_llm))

    # Analysts run sequentially; each may detour through its tool node.
    workflow.add_edge(START, "performance_analyst")
    next_stops = ["audience_analyst", "market_analyst", "financial_analyst",
                  "critical_analyst", "renew_advocate"]
    for (name, _, _), nxt in zip(analysts, next_stops):
        workflow.add_conditional_edges(
            f"{name}_analyst", _route_on_tool_calls,
            {"call_tools": f"{name}_tools", "continue": nxt},
        )
        workflow.add_edge(f"{name}_tools", nxt)

    # Debate and decision flow
    workflow.add_edge("renew_advocate", "cancel_advocate")
    workflow.add_edge("cancel_advocate", "programming_director")
    workflow.add_edge("programming_director", "network_executive")
    workflow.add_edge("network_executive", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


class TVShowAgents:
    def __init__(self, config: dict | None = None, debug: bool = False):
        self.config = config or CONFIG
        self.debug = debug
        self.graph = build_tvshow_agents_graph(self.config)

    def evaluate_show_by_name(self, show_name: str, **kwargs):
        print(f"\n{'=' * 80}\nSearching TMDB for '{show_name}'...\n{'=' * 80}")
        tmdb_id = search_tv_show(show_name)
        if not tmdb_id:
            print(f"Show '{show_name}' not found on TMDB")
            return None
        show_data = tmdb_request(f"/tv/{tmdb_id}")
        show = {
            "show_id": f"tmdb_{tmdb_id}",
            "title": show_data.get("name", show_name),
            "tmdb_id": tmdb_id,
            "network": kwargs.get("network")
                or show_data.get("networks", [{}])[0].get("name", "Unknown"),
            "genre": kwargs.get("genre")
                or ", ".join(g["name"] for g in show_data.get("genres", [])[:2]),
            "current_season": kwargs.get("current_season")
                or show_data.get("number_of_seasons", 1),
            "episodes_aired": kwargs.get("episodes_aired")
                or show_data.get("number_of_episodes", 0),
            "production_cost_per_episode": kwargs.get("production_cost_per_episode", 3.0),
            "showrunner": kwargs.get("showrunner", "Unknown"),
            "cast": kwargs.get("cast", ["Ensemble"]),
        }
        print(f"\n Show: {show['title']}")
        print(f" Network: {show['network']} | Genre: {show['genre']}")
        print(f" Seasons: {show['current_season']} | Episodes: {show['episodes_aired']}")
        print("Running analysis...")
        return self.evaluate_show(show)

    def evaluate_show(self, show: dict):
        init_state = {
            "messages": [], "show_id": show["show_id"], "show_title": show["title"],
            "tmdb_id": show["tmdb_id"], "network": show["network"], "genre": show["genre"],
            "current_season": show["current_season"],
            "episodes_aired": show["episodes_aired"],
            "production_cost_per_episode": show["production_cost_per_episode"],
            "showrunner": show["showrunner"], "cast": show["cast"], "sender": "system",
            "performance_report": "", "audience_report": "", "market_report": "",
            "financial_report": "", "critical_report": "",
            "renewal_debate_state": {
                "renew_history": "", "cancel_history": "", "history": "",
                "current_response": "", "judge_decision": "", "count": 0,
            },
            "renewal_recommendation": "", "decision_agent_plan": "", "final_decision": "",
        }
        final_state = self.graph.invoke(init_state)
        decision_text = final_state.get("final_decision", "")

        decision_type = "UNKNOWN"
        if "**RENEW**" in decision_text and "CONDITIONAL" not in decision_text:
            decision_type = "RENEW"
        elif "RENEW_CONDITIONAL" in decision_text:
            decision_type = "RENEW_CONDITIONAL"
        elif "FINAL_SEASON" in decision_text:
            decision_type = "FINAL_SEASON"
        elif "**CANCEL**" in decision_text:
            decision_type = "CANCEL"

        self._print_decision(show["title"], decision_type, decision_text)
        return {"decision_type": decision_type,
                "full_decision": decision_text, "state": final_state}

    @staticmethod
    def _print_decision(title: str, decision_type: str, decision_text: str) -> None:
        badges = {
            "RENEW": "🟢 RENEW",
            "RENEW_CONDITIONAL": "🟡 CONDITIONAL RENEWAL",
            "FINAL_SEASON": "🟠 FINAL SEASON",
            "CANCEL": "🔴 CANCELED",
        }
        print(f"\n{badges.get(decision_type, decision_type)}")
        print("=" * 80 + "\n")
        print(decision_text)
        print("\n" + "=" * 80)


if __name__ == "__main__":
    show = sys.argv[1] if len(sys.argv) > 1 else "Stranger Things"
    tsa = TVShowAgents(config=CONFIG)
    tsa.evaluate_show_by_name(show)
