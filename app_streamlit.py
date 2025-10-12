# app_streamlit.py
import os
import json
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from src.manifest import load_manifest
from src.agentic import answer
from src.config import settings

st.set_page_config(page_title="EU Navigator (PLP)", layout="wide")
st.set_option("client.showErrorDetails", True)

PROGRESS_PATH = Path("user_progress.json")


def load_progress():
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {}

def save_progress(state):
    PROGRESS_PATH.write_text(json.dumps(state, indent=2))

def force_bullets(text: str) -> str:
    return (text or "").replace("•", "\n\n•").strip()

#Download Button for the pdfs
def download_button_for_pdf(pdf_path: str, label_prefix: str, key_suffix: str):
    p = Path(pdf_path).expanduser()
    if not p.exists():
        st.warning(f"PDF not found on disk: {pdf_path}")
        return
    data = p.read_bytes()
    
    #Unique key for this downlaod button
    uniq = hashlib.md5((str(p.resolve()) + "|" + key_suffix).encode()).hexdigest()
    st.download_button(
        label=f"{label_prefix}: {p.name}",
        data=data,
        file_name=p.name,
        mime="application/pdf",
        key=f"dl_{uniq}",
        use_container_width=False,
    )

#Loading Manifest
rows = load_manifest(str(settings.manifest_csv))
df = pd.DataFrame([r.__dict__ for r in rows])
pdf_by_id = {r.doc_id: r.pdf_path for r in rows}


#UI 
st.title("EU Navigator - Personal Learning Portal")
tabs = st.tabs(["Learn", "Ask EU Navigator", "Progress"])


#Learn Tab
with tabs[0]:
    st.subheader("Course Modules & Documents")
    if df.empty:
        st.error("Manifest is empty. Check settings.manifest_csv and your data folders.")
    else:
        modules = sorted(df["module"].unique().tolist())
        m = st.selectbox("Select module", modules)

        mdf = df[df["module"] == m].copy()
        progress = load_progress()
        progress.setdefault(m, {})

        left, right = st.columns([0.5, 0.5], gap="large")

        with left:
            st.write("**Documents**")
            for _, r in mdf.iterrows():
                doc_id = r["doc_id"]
                title = r["title"]
                done = progress[m].get(doc_id, {}).get("done", False)
                col1, col2 = st.columns([0.75, 0.25])
                with col1:
                    st.write(f"**{title}**  \n`{doc_id}`")
                with col2:
                    new_done = st.checkbox("Done", value=done, key=f"done_{m}_{doc_id}")
                    progress[m].setdefault(doc_id, {})["done"] = new_done
            save_progress(progress)

            total = len(mdf)
            completed = sum(1 for r in mdf["doc_id"] if progress[m].get(r, {}).get("done"))
            st.progress(0 if total == 0 else completed / total)
            st.caption(f"{completed}/{total} completed")

        with right:
            st.write("**Download current document**")
            which = st.selectbox(
                "Select document to download",
                mdf["doc_id"].tolist(),
                format_func=lambda x: df[df["doc_id"] == x]["title"].iloc[0],
                key=f"open_{m}",
            )
            pdf_path = df[df["doc_id"] == which]["pdf_path"].iloc[0]
            download_button_for_pdf(
                pdf_path,
                label_prefix="Download",
                key_suffix=f"learn_{m}_{which}",
            )

            st.markdown("### ✍️ Reflection / Notes")
            curr = progress[m].setdefault(which, {}).get("note", "")
            new_note = st.text_area("Notes for this document", value=curr, height=140, key=f"note_{m}_{which}")
            progress[m][which]["note"] = new_note
            save_progress(progress)

#Ask Tab
with tabs[1]:
    st.subheader("Ask EU Navigator")
    q = st.text_input("Question", placeholder="e.g., Is text-and-data mining lawful for AI training in the EU?")
    module_opt = st.selectbox("(Optional) Focus module", ["", "Equality_Foundations", "Data_IP_TDM", "AI_Cyber_Gov"])

    if st.button("Answer"):
        if not q.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving and synthesizing..."):
                out = answer(q, module=module_opt or None)

            st.markdown("### Answer")
            st.markdown(force_bullets(out.get("answer", "")))

            st.divider()
            st.subheader("💬 Reviewer Notes")
            st.info(out.get("review", ""))

            st.markdown("### Sources")
            srcs = out.get("sources", [])
            if not srcs:
                st.caption("No sources returned.")
            else:
                for i, s in enumerate(srcs):
                    did = s.get("doc_id", "UNKNOWN")
                    label = f"[{did}: {s.get('section','') or ''}] — {s.get('title','')} ({s.get('module','')})"
                    st.write(f"- {label}")
                    src_pdf = pdf_by_id.get(did)
                    if src_pdf and Path(src_pdf).exists():
                        download_button_for_pdf(
                            src_pdf,
                            label_prefix=f"Download {did}",
                            key_suffix=f"src_{i}_{did}",
                        )
                    else:
                        st.caption("  ↳ PDF not available or missing on disk.")

#Progress Tab
with tabs[2]:
    st.subheader("Progress Overview")
    progress = load_progress()

    records = []
    for m in sorted(progress.keys()):
        for doc_id, info in progress[m].items():
            rowmask = (df["doc_id"] == doc_id)
            title = df[rowmask]["title"].iloc[0] if rowmask.any() else doc_id
            records.append({
                "module": m,
                "doc_id": doc_id,
                "title": title,
                "done": bool(info.get("done", False)),
                "note": info.get("note", ""),
            })

    if records:
        pdf = pd.DataFrame(records)
        st.dataframe(pdf, use_container_width=True)
    else:
        st.info("No progress recorded yet.")
