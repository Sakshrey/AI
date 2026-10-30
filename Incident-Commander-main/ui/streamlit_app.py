from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.state import AegisIncidentState
from app.graph import (
    supervisor_step,
    log_step,
    metrics_step,
    database_step,
    deployment_step,
    correlation_step,
    human_review_step,
    human_rework_step,
    report_step,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Software Incident Commander",
    page_icon="🚨",
    layout="wide",
)

st.title("🚨 AI Software Incident Commander")

st.caption(
    "Multi-agent incident investigation, root-cause analysis, "
    "critic review, and human-in-the-loop approval"
)

# ============================================================
# OFFLINE DEMO NODES
# ============================================================

def demo_rag_step(state: AegisIncidentState) -> dict[str, Any]:

    return {
        "rag_context": [
            {
                "filename": "http_500_database_errors.md",
                "category": "runbook",
                "content": (
                    "HTTP 500 errors can occur when database "
                    "connection pools are exhausted."
                ),
            },
            {
                "filename": "database_connection_pool.md",
                "category": "runbook",
                "content": (
                    "Connection pool exhaustion can occur when "
                    "pool configuration is insufficient or incorrect."
                ),
            },
            {
                "filename": "INC-0001.md",
                "category": "incident",
                "content": (
                    "A previous incident involved HTTP 500 errors "
                    "and database connection pool exhaustion."
                ),
            },
        ],
        "investigation_status": "rag_complete",
    }


def demo_root_cause_step(
    state: AegisIncidentState,
) -> dict[str, Any]:

    feedback = state.get("human_feedback", "")
    iteration = int(state.get("iteration", 0))

    logs = state.get("logs", []) or []
    metrics = state.get("metrics", []) or []
    database_events = state.get("database_events", []) or []
    deployments = state.get("deployments", []) or []
    correlations = state.get("correlations", []) or []

    evidence_counts = {
        "logs": len(logs),
        "metrics": len(metrics),
        "database": len(database_events),
        "deployments": len(deployments),
        "correlations": len(correlations),
    }

    available_sources = [
        name
        for name, count in evidence_counts.items()
        if count > 0
    ]

    if not available_sources:

        root_cause = (
            "Insufficient evidence to determine the root cause. "
            "The investigation did not load any log, metric, database, "
            "deployment, or correlation records."
        )

        rationale = (
            "No investigation evidence was available. The database "
            "connection-pool hypothesis must not be presented as "
            "confirmed until the data sources are loaded and "
            "relevant records are examined."
        )

        confidence = 0.0

        missing_evidence = (
            "Load application logs, API metrics, database events, "
            "deployment history, and timestamp-based correlations "
            "before selecting a root cause."
        )

        hypotheses = []

    else:

        root_cause = (
            "Database connection-pool exhaustion is a candidate "
            "explanation for the API HTTP 500 errors. Check the "
            "loaded database, log, metric, and deployment records; "
            "this demo result is not proof of direct causation."
        )

        rationale = (
            "The following evidence sources were loaded: "
            + ", ".join(available_sources)
            + ". The demo hypothesis is based on the sample incident "
            "dataset. Verify exact timestamps and production "
            "connection-pool settings before treating the deployment "
            "as the cause."
        )

        confidence = min(
            0.75,
            0.15 * len(available_sources),
        )

        missing_evidence = (
            "Verify production pool-size and connection-limit "
            "configuration, exact deployment timing, and whether "
            "pool exhaustion precedes the HTTP 500 spike."
        )

        hypotheses = [
            {
                "type": "candidate",
                "root_cause": root_cause,
                "confidence": confidence,
                "rationale": rationale,
                "missing_evidence": missing_evidence,
                "evidence_counts": evidence_counts,
            }
        ]

    if feedback:

        rationale += (
            " Human reviewer feedback was considered: "
            + feedback
        )

    return {
        "root_cause": root_cause,
        "confidence": confidence,
        "hypotheses": hypotheses,
        "root_cause_rationale": rationale,
        "missing_evidence": missing_evidence,
        "root_cause_agent_message": (
            "Offline demo RCA generated using loaded evidence counts: "
            + str(evidence_counts)
        ),
        "investigation_status": "root_cause_analysis_complete",
        "next_action": "critic_review",
        "iteration": iteration,
        "demo_evidence_counts": evidence_counts,
    }


def demo_critic_step(
    state: AegisIncidentState,
) -> dict[str, Any]:

    counts = state.get("demo_evidence_counts", {}) or {}

    total_records = sum(
        int(value or 0)
        for value in counts.values()
    )

    if total_records == 0:

        verdict = "REWORK"

        feedback = (
            "Verdict: REWORK. No log, metric, database, deployment, "
            "or correlation records were loaded. The proposed root "
            "cause is unsupported. Fix data loading and rerun the "
            "investigation before approving an RCA."
        )

    else:

        verdict = "PASS WITH CAVEAT"

        feedback = (
            "Verdict: PASS WITH CAVEAT. Sample evidence records were "
            "loaded from the following sources: "
            + ", ".join(
                name
                for name, count in counts.items()
                if count
            )
            + ". This supports investigating database connection-pool "
            "exhaustion, but does not establish direct causation. "
            "Verify production configuration and exact event timing."
        )

    return {
        "critic_feedback": feedback,
        "critic_verdict": verdict,
        "investigation_status": "critic_review_complete",
        "next_action": "generate_final_report",
    }


# ============================================================
# OFFLINE GRAPH
# ============================================================

@st.cache_resource
def get_offline_graph():

    workflow = StateGraph(AegisIncidentState)

    workflow.add_node("supervisor", supervisor_step)
    workflow.add_node("rag", demo_rag_step)
    workflow.add_node("logs", log_step)
    workflow.add_node("metrics", metrics_step)
    workflow.add_node("database", database_step)
    workflow.add_node("deployment", deployment_step)
    workflow.add_node("correlation", correlation_step)
    workflow.add_node("root_cause", demo_root_cause_step)
    workflow.add_node("critic", demo_critic_step)
    workflow.add_node("human_review", human_review_step)
    workflow.add_node("human_rework", human_rework_step)
    workflow.add_node("report", report_step)

    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", "rag")
    workflow.add_edge("rag", "logs")
    workflow.add_edge("logs", "metrics")
    workflow.add_edge("metrics", "database")
    workflow.add_edge("database", "deployment")
    workflow.add_edge("deployment", "correlation")
    workflow.add_edge("correlation", "root_cause")
    workflow.add_edge("root_cause", "critic")

    workflow.add_conditional_edges(
        "critic",
        lambda state: "human_review",
        {
            "human_review": "human_review",
        },
    )

    def human_route(state: AegisIncidentState):

        if state.get("human_decision") == "rework":
            return "human_rework"

        return "report"

    workflow.add_conditional_edges(
        "human_review",
        human_route,
        {
            "human_rework": "human_rework",
            "report": "report",
        },
    )

    workflow.add_edge("human_rework", "root_cause")
    workflow.add_edge("report", END)

    return workflow.compile(
        checkpointer=MemorySaver()
    )


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state["result"] = None

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = None

if "awaiting_review" not in st.session_state:
    st.session_state["awaiting_review"] = False

if "interrupt_data" not in st.session_state:
    st.session_state["interrupt_data"] = None

if "mode" not in st.session_state:
    st.session_state["mode"] = "Offline / Demo"


def get_config():

    return {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }


def save_result(result: dict[str, Any]):

    st.session_state["result"] = result

    if "__interrupt__" in result:

        interrupt = result["__interrupt__"][0]

        st.session_state["awaiting_review"] = True
        st.session_state["interrupt_data"] = interrupt.value

    else:

        st.session_state["awaiting_review"] = False
        st.session_state["interrupt_data"] = None


def reset_investigation():

    st.session_state["result"] = None
    st.session_state["thread_id"] = None
    st.session_state["awaiting_review"] = False
    st.session_state["interrupt_data"] = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Investigation Settings")

    mode = st.radio(
        "Execution Mode",
        [
            "Offline / Demo",
            "Live AI",
        ],
        index=0,
    )

    st.session_state["mode"] = mode

    st.divider()

    st.markdown("### Workflow")

    st.markdown(
        """
1. 🧭 Supervisor
2. 📚 RAG
3. 📋 Logs
4. 📈 Metrics
5. 🗄️ Database
6. 🚀 Deployment
7. 🔗 Correlation
8. 🎯 Root Cause
9. 🧐 Critic
10. 🧑 Human Review
11. 📄 Final Report
"""
    )

    st.divider()

    if mode == "Offline / Demo":

        st.success("Demo mode: 0 Groq API calls")

    else:

        st.warning(
            "Live mode uses your configured Groq API quota."
        )

    if st.button(
        "🔄 Reset Investigation",
        use_container_width=True,
    ):

        reset_investigation()
        st.rerun()


# ============================================================
# INCIDENT INPUT
# ============================================================

st.header("📝 Incident Details")

col1, col2, col3 = st.columns(3)

with col1:

    incident_id = st.text_input(
        "Incident ID",
        value="INC-001",
    )

with col2:

    service = st.text_input(
        "Service",
        value="api",
    )

with col3:

    severity = st.selectbox(
        "Severity",
        [
            "SEV-1",
            "SEV-2",
            "SEV-3",
            "SEV-4",
        ],
        index=1,
    )


description = st.text_area(
    "Incident Description",
    value="API returning HTTP 500 errors",
    height=100,
)


# ============================================================
# START INVESTIGATION
# ============================================================

if st.button(
    "🚀 Start Investigation",
    type="primary",
    use_container_width=True,
):

    if not incident_id.strip():

        st.error("Please enter an Incident ID.")
        st.stop()

    if not service.strip():

        st.error("Please enter a service name.")
        st.stop()

    if not description.strip():

        st.error("Please enter an incident description.")
        st.stop()

    reset_investigation()

    st.session_state["thread_id"] = (
        f"streamlit-{incident_id.strip()}-"
        f"{uuid.uuid4().hex[:8]}"
    )

    execution_mode = (
        "live"
        if mode == "Live AI"
        else "offline"
    )

    initial_state: AegisIncidentState = {
        "incident_id": incident_id.strip(),
        "description": description.strip(),
        "service": service.strip(),
        "severity": severity,
        "execution_mode": execution_mode,
        "iteration": 0,
        "execution_history": [],
        "logs": [],
        "metrics": [],
        "database_events": [],
        "deployments": [],
    }

    if mode == "Offline / Demo":

        graph = get_offline_graph()

    else:

        from app.graph import incident_graph

        graph = incident_graph

    with st.spinner(
        "🔎 Incident Commander is investigating..."
    ):

        try:

            result = graph.invoke(
                initial_state,
                config=get_config(),
            )

            save_result(result)

        except Exception as exc:

            st.error("Investigation failed.")
            st.exception(exc)
            st.stop()

    st.rerun()


# ============================================================
# GET RESULT
# ============================================================

result = st.session_state.get("result")

if not result:

    st.info(
        "Enter the incident details and click "
        "**Start Investigation** to begin."
    )

    st.stop()


# ============================================================
# HUMAN REVIEW
# ============================================================

if st.session_state.get("awaiting_review"):

    interrupt_data = (
        st.session_state.get("interrupt_data")
        or {}
    )

    st.divider()

    st.header("🧑 Human Review Required")

    st.warning(
        "The AI investigation is paused. Review the proposed "
        "root cause before generating the final report."
    )

    review_col1, review_col2 = st.columns(2)

    with review_col1:

        st.markdown("### 🎯 Proposed Root Cause")

        st.info(
            interrupt_data.get(
                "root_cause",
                result.get("root_cause", "Not available."),
            )
        )

        confidence = interrupt_data.get(
            "confidence",
            result.get("confidence", 0.0),
        )

        if isinstance(confidence, (int, float)):

            st.metric(
                "RCA Confidence",
                f"{confidence * 100:.0f}%",
            )

        else:

            st.metric(
                "RCA Confidence",
                str(confidence),
            )

    with review_col2:

        st.markdown("### 🧐 Critic Review")

        st.write(
            interrupt_data.get(
                "critic_feedback",
                result.get(
                    "critic_feedback",
                    "No critic feedback.",
                ),
            )
        )

    st.markdown("### 🔍 Evidence to Verify")

    missing_evidence = result.get(
        "missing_evidence",
        "Verify the production database connection-pool configuration.",
    )

    st.info(missing_evidence)

    hypotheses = result.get("hypotheses", [])

    if hypotheses:

        with st.expander("View Hypotheses"):

            for hypothesis in hypotheses:

                if isinstance(hypothesis, dict):
                    st.json(hypothesis)
                else:
                    st.write(hypothesis)

    feedback = st.text_area(
        "Reviewer Feedback",
        placeholder=(
            "Optional for approval. Required when requesting rework."
        ),
        key="human_feedback_input",
    )

    approve_col, rework_col = st.columns(2)

    if mode == "Offline / Demo":

        graph = get_offline_graph()

    else:

        from app.graph import incident_graph

        graph = incident_graph

    with approve_col:

        if st.button(
            "✅ Approve RCA",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "📄 Generating final incident report..."
            ):

                try:

                    resumed = graph.invoke(
                        Command(
                            resume={
                                "decision": "approve",
                                "feedback": feedback.strip(),
                            }
                        ),
                        config=get_config(),
                    )

                    save_result(resumed)

                except Exception as exc:

                    st.error(
                        "Unable to approve the investigation."
                    )

                    st.exception(exc)
                    st.stop()

            st.rerun()

    with rework_col:

        if st.button(
            "🔄 Request RCA Rework",
            use_container_width=True,
        ):

            if not feedback.strip():

                st.error(
                    "Please provide feedback explaining "
                    "what should be reconsidered."
                )

            else:

                with st.spinner(
                    "🔄 Reworking root-cause analysis..."
                ):

                    try:

                        resumed = graph.invoke(
                            Command(
                                resume={
                                    "decision": "rework",
                                    "feedback": feedback.strip(),
                                }
                            ),
                            config=get_config(),
                        )

                        save_result(resumed)

                    except Exception as exc:

                        st.error(
                            "Unable to request RCA rework."
                        )

                        st.exception(exc)
                        st.stop()

                st.rerun()

    st.stop()


# ============================================================
# INVESTIGATION RESULT
# ============================================================

st.divider()

st.header("🔎 Investigation Result")

status = result.get(
    "investigation_status",
    "unknown",
)

result_incident_id = result.get(
    "incident_id",
    incident_id,
)

result_service = result.get(
    "service",
    service,
)

confidence = result.get("confidence", 0)
iterations = result.get("iteration", 0)

if isinstance(confidence, (int, float)):

    confidence_percent = f"{confidence * 100:.0f}%"

else:

    confidence_percent = str(confidence)


if status in {
    "incident_report_complete",
    "final_report_complete",
}:

    st.success("✅ Investigation completed")

else:

    st.warning(f"Investigation status: {status}")


# ============================================================
# SUMMARY CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric("Incident", result_incident_id)

with col2:

    st.metric("Service", result_service)

with col3:

    st.metric("Confidence", confidence_percent)

with col4:

    st.metric("Iterations", iterations)


# ============================================================
# ROOT CAUSE
# ============================================================

st.subheader("🎯 Root Cause")

st.info(
    result.get(
        "root_cause",
        "Root cause was not determined.",
    )
)

rationale = result.get("root_cause_rationale", "")

if rationale:

    with st.expander("Why this RCA was selected"):

        st.write(rationale)


# ============================================================
# CRITIC
# ============================================================

st.subheader("🧐 Critic Review")

st.write(
    result.get(
        "critic_feedback",
        "No critic feedback available.",
    )
)


# ============================================================
# HUMAN DECISION
# ============================================================

human_decision = result.get("human_decision")

if human_decision:

    if human_decision == "approve":

        st.success("🧑 Human Decision: APPROVED")

    else:

        st.warning(
            f"🧑 Human Decision: {human_decision.upper()}"
        )

human_feedback = result.get("human_feedback")

if human_feedback:

    with st.expander("Human Reviewer Feedback"):

        st.write(human_feedback)


# ============================================================
# EVIDENCE SUMMARY
# ============================================================

st.subheader("🔎 Investigation Evidence")

logs = result.get("logs", []) or []
metrics = result.get("metrics", []) or []
database_events = result.get("database_events", []) or []
deployments = result.get("deployments", []) or []
correlations = result.get("correlations", []) or []
hypotheses = result.get("hypotheses", []) or []

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Logs", len(logs))

with col2:
    st.metric("Metrics", len(metrics))

with col3:
    st.metric("Database", len(database_events))

with col4:
    st.metric("Deployments", len(deployments))

with col5:
    st.metric("Correlations", len(correlations))

with col6:
    st.metric("Hypotheses", len(hypotheses))


# ============================================================
# FINDINGS
# ============================================================

st.subheader("📌 Findings")

log_findings = result.get("log_findings", []) or []
metrics_findings = result.get("metrics_findings", []) or []
database_findings = result.get("database_findings", []) or []
deployment_findings = result.get("deployment_findings", []) or []

tabs = st.tabs(
    [
        "Logs",
        "Metrics",
        "Database",
        "Deployments",
        "Correlations",
        "Hypotheses",
    ]
)


with tabs[0]:

    if log_findings:

        for finding in log_findings:
            st.write(f"• {finding}")

    else:

        st.info("No log findings available.")

    if logs:

        with st.expander("View Raw Logs"):
            st.json(logs)


with tabs[1]:

    if metrics_findings:

        for finding in metrics_findings:
            st.write(f"• {finding}")

    else:

        st.info("No metric findings available.")

    if metrics:

        with st.expander("View Raw Metrics"):
            st.json(metrics)


with tabs[2]:

    if database_findings:

        for finding in database_findings:
            st.write(f"• {finding}")

    else:

        st.info("No database findings available.")

    if database_events:

        with st.expander("View Raw Database Events"):
            st.json(database_events)


with tabs[3]:

    if deployment_findings:

        for finding in deployment_findings:
            st.write(f"• {finding}")

    else:

        st.info("No deployment findings available.")

    if deployments:

        with st.expander("View Raw Deployments"):
            st.json(deployments)


with tabs[4]:

    if correlations:

        for correlation in correlations:
            st.write(f"• {correlation}")

    else:

        st.info("No correlations available.")


with tabs[5]:

    if hypotheses:

        for hypothesis in hypotheses:

            if isinstance(hypothesis, dict):
                st.json(hypothesis)
            else:
                st.write(f"• {hypothesis}")

    else:

        st.info("No hypotheses available.")


# ============================================================
# AGENT TIMELINE
# ============================================================

st.subheader("🤖 Agent Investigation Timeline")

execution_history = result.get("execution_history", []) or []

st.caption(
    f"{len(execution_history)} agent events "
    "recorded during the investigation"
)

if execution_history:

    for index, event in enumerate(
        execution_history,
        start=1,
    ):

        if isinstance(event, dict):

            agent = event.get("agent", "Agent")
            status_value = event.get("status", "completed")
            message = event.get("message", "")

            if status_value == "completed":
                icon = "✅"
            elif status_value == "started":
                icon = "🔵"
            elif status_value == "rework":
                icon = "🔄"
            elif status_value == "failed":
                icon = "❌"
            else:
                icon = "⚪"

            with st.container(border=True):

                st.markdown(
                    f"### {icon} {index}. {agent} Agent"
                )

                st.caption(
                    f"Status: {status_value.upper()}"
                )

                if message:
                    st.write(message)

        else:

            with st.container(border=True):

                st.markdown(
                    f"### ✅ {index}. {event}"
                )

else:

    st.info("No agent execution information available.")


# ============================================================
# FINAL REPORT
# ============================================================

st.divider()

st.header("📋 Final Incident Report")

final_report = result.get("final_report", "")

if final_report:

    st.markdown(final_report)

    st.download_button(
        label="📥 Download Incident Report",
        data=final_report,
        file_name=(
            f"{result_incident_id}_"
            "incident_report.md"
        ),
        mime="text/markdown",
        use_container_width=True,
    )

else:

    st.warning(
        "The Final Report Agent did not produce a final report."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

if st.session_state["mode"] == "Offline / Demo":

    st.caption(
        "Demo mode • Deterministic offline investigation • "
        "No Groq API calls"
    )

else:

    st.caption(
        "Live AI mode • Uses the configured Groq model"
    )
