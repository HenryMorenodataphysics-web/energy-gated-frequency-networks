from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.egfn_context import (
    add_repair,
    ingest_document,
    initialize_database,
    load_evidence,
    retrieve_context,
    retrieve_knowledge,
)
from src.audio_diagnosis import diagnose_wav


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env.local")
EVIDENCE_PATH = ROOT / "evidence" / "egfn_evidence.json"
DATABASE_PATH = ROOT / "evidence" / "egfn_context.sqlite3"
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
CHECKPOINT_PATH = ROOT / "outputs" / "mimii_hierarchical_gating_ablation_v7" / "runs" / "hierarchical" / "seed_42" / "best_checkpoint.pt"

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "findings": {"type": "array", "items": {"type": "string"}},
        "recommendation": {"type": "string"},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "findings", "recommendation", "limitations", "citations"],
    "additionalProperties": False,
}


def deterministic_explanation(context: list[dict]) -> dict:
    return {
        "summary": f"Se recuperaron {len(context)} bloques de evidencia verificable de EGFN.",
        "findings": [
            "Las métricas mostradas provienen de artefactos JSON locales.",
            "La explicación no sustituye la evaluación experimental ni modifica sus resultados.",
        ],
        "limitations": ["El modo determinista no interpreta automáticamente cada métrica."],
        "citations": [item["source_path"] for item in context],
    }


def openai_explanation(context: list[dict], knowledge: list[dict]) -> dict:
    from openai import OpenAI

    prompt = (
        "IMPORTANT: respond in clear English for a non-technical person. Start "
        "with a clear conclusion about whether there may be a valve failure. "
        "Do not use AUC, seeds, ablations, or technical jargon in visible fields. "
        "Include a practical recommendation and do not invent a physical cause "
        "if the evidence does not identify one.\n\n"
        "Responde en español y explica el diagnóstico del audio. La entrada "
        "audio_diagnosis es la fuente prioritaria. Explica si el score supera el "
        "umbral y qué significa el estado. No mezcles resultados históricos, "
        "ablaciones, AUCs o seeds con este diagnóstico salvo que se solicite "
        "explícitamente. No inventes una causa física concreta: si la evidencia "
        "no identifica el componente defectuoso, dilo claramente. Usa solo la "
        "evidencia JSON y los fragmentos de conocimiento proporcionados; cita sus "
        "source_path o filename.\n\n"
        + json.dumps({"evidence": context, "knowledge": knowledge}, ensure_ascii=False)
    )
    client = OpenAI()
    response = client.responses.create(
        model=MODEL,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "egfn_explanation",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
    )
    return json.loads(response.output_text)


st.set_page_config(page_title="EGFN Evidence", layout="wide")
st.title("EGFN Evidence Analyzer")
st.caption("Evidence JSON -> SQLite -> retrieval -> structured explanation")

if not EVIDENCE_PATH.exists():
    st.warning("evidence/egfn_evidence.json is missing. Run scripts/build_evidence_json.py first.")
    st.stop()

evidence = load_evidence(EVIDENCE_PATH)
initialize_database(DATABASE_PATH, evidence)

with st.popover("Knowledge base"):
    st.header("Knowledge base")
    uploaded = st.file_uploader("Upload a manual, report, or repair record", type=["pdf", "txt", "md", "json"])
    if uploaded and st.button("Save document"):
        try:
            document_id = ingest_document(DATABASE_PATH, uploaded.name, uploaded.getvalue())
            st.success(f"Document saved: {document_id}")
        except (RuntimeError, ValueError) as exc:
            st.error(str(exc))
    st.subheader("Record a repair")
    machine_id = st.text_input("Machine / ID")
    symptom = st.text_area("Symptom")
    action = st.text_area("Action taken")
    outcome = st.text_area("Outcome")
    if st.button("Save repair"):
        if all((machine_id, symptom, action, outcome)):
            add_repair(DATABASE_PATH, machine_id, symptom, action, outcome)
            st.success("Repair saved")
        else:
            st.warning("Complete all repair fields.")

st.subheader("Audio diagnosis")
st.caption("Calibrated model available: valve. Fan support will be added with a dedicated checkpoint.")
audio_file = st.file_uploader("Engine audio", type=["wav"], key="engine_audio")
diagnosis = None
if st.button("Check for failure", type="primary"):
    if audio_file is None:
        st.warning("Upload a WAV file first.")
    elif not CHECKPOINT_PATH.is_file():
        st.error("No calibrated EGFN model is available.")
    else:
        try:
            diagnosis = diagnose_wav(
                audio_file.getvalue(),
                CHECKPOINT_PATH,
                "mimii/valve/unknown/unknown",
            )
            st.session_state["last_diagnosis"] = diagnosis
        except Exception as exc:
            st.error(f"The audio could not be analyzed: {exc}")
diagnosis = st.session_state.get("last_diagnosis", diagnosis)
if diagnosis:
    st.subheader("Detection result")
    if diagnosis["status"] == "possible_failure":
        st.error("Signals consistent with a possible valve failure")
    else:
        st.success("No clear valve-failure signals were detected")
    with st.expander("Technical details"):
        st.json(diagnosis)

if diagnosis and st.button("Explain the result"):
    if not os.getenv("OPENAI_API_KEY"):
        st.error("The explanation service is not configured. Add an API key to .env.local.")
    else:
        try:
            with st.spinner("Generating explanation..."):
                context = [
                    {
                        "evidence_id": "audio-diagnosis",
                        "source_path": diagnosis["checkpoint"],
                        "title": "audio_diagnosis",
                        "payload": diagnosis,
                    }
                ]
                knowledge = retrieve_knowledge(DATABASE_PATH, "valve")
                result = openai_explanation(context, knowledge)
            st.subheader("Explanation")
            st.write(result["summary"])
            st.markdown("**What was observed**")
            for finding in result["findings"]:
                st.markdown(f"- {finding}")
            st.markdown(f"**What to do next:** {result['recommendation']}")
            if result["limitations"]:
                with st.expander("What this analysis cannot confirm"):
                    for limitation in result["limitations"]:
                        st.markdown(f"- {limitation}")
            if result["citations"]:
                with st.expander("Sources used"):
                    for citation in result["citations"]:
                        st.code(citation)
        except Exception as exc:
            error_text = str(exc)
            if "insufficient_quota" in error_text or "current quota" in error_text:
                st.warning(
                    "The EGFN diagnosis completed, but the explanation service has no "
                    "available quota. Check billing and project limits."
                )
            elif "429" in error_text:
                st.warning("The explanation service is temporarily rate-limited. Try again later.")
            else:
                st.error(f"The explanation could not be generated: {exc}")
