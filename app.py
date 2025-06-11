import streamlit as st
import os
import io
from docx import Document
import base64
from pdf2image import convert_from_bytes
from PIL import Image
import re
import pandas as pd
from dataclasses import dataclass, replace

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
load_dotenv()

# == SAP GenAI Configuration ==
from dotenv import load_dotenv
load_dotenv()
Client_ID = os.getenv("AICORE_CLIENT_ID")
Client_Secret = os.getenv("AICORE_CLIENT_SECRET")
Auth_URL = os.getenv("AICORE_AUTH_URL")
Resource_Group = os.getenv("AICORE_RESOURCE_GROUP")
Base_URL = os.getenv("AICORE_BASE_URL")
LLM_DEPLOYMENT_ID = os.getenv("LLM_DEPLOYMENT_ID")

llm = ChatOpenAI(deployment_id=LLM_DEPLOYMENT_ID)
parser = JsonOutputParser()

# == OCR-based Image Extraction ==
def extract_text_from_image(image_bytes):
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    message = HumanMessage(content=[
        {"type": "text", "text": "Extract all text from this page correctly."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
    ])
    return llm.invoke([message]).content

def extract_pdf_content(pdf_file):
    images = convert_from_bytes(pdf_file.read())
    content = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        text = extract_text_from_image(buf.getvalue())
        content.append(text.strip())
    return "\n\n".join(content)

def extract_docx_content(docx_file):
    doc = Document(docx_file)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_file_content(file):
    if not file:
        return ""
    file.seek(0)
    if file.type == "application/pdf":
        return extract_pdf_content(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_docx_content(file)
    return "Unsupported file type."

def extract_register_number(text):
    match = re.search(r"(Register\s*No\.?\s*[:\-]?\s*)([A-Za-z0-9]+)", text, re.IGNORECASE)
    if match:
        return match.group(2)
    match = re.search(r"\b[A-Za-z0-9]{6,}\b", text)
    return match.group(0) if match else "Unknown"

eval_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an exam evaluator. You will be given:\n"
     "- The full student answer script.\n"
     "- The answer key (with question numbers and correct answers).\n"
     "- The full question paper (with question numbers, sections, and marks allocation).\n\n"
     "Instructions:\n"
     "-Evaluate correctly for all student answers based on the provided answer key and question paper pattern is important (if multiple student answer sheet) Question paper pattern is same for all answer sheet so evaluate based on that pattern.\n"
     "1. Carefully extract the question pattern, section structure (including choices/options), and mark allocation from the question paper.Understand the question paper pattern (choices,marks,total marks) based on that evaluate student answer sheet\n"
     "2. For each question in the pattern, extract the student's answer (if present). and evaluate it based on the answer key with question paper mark allocation\n"
     "3. Compare the student's answer with the answer key:\n"
     "   - Award **full marks only if the answer is fully correct and complete**.\n"
     "   - Award **partial marks** for partially correct or incomplete answers, based on the quality and completeness.\n"
     "   - Award **zero marks** for incorrect or missing answers.\n"
     "   - Be strict and do not give marks for irrelevant or wrong content.\n"
     "4. If the question paper has sections or choices (e.g., 'Answer any 2 from Section B'), only award marks for the attempted options as per the pattern, and do not double-count marks for choices.\n"
     "5. Always include every question from the pattern in your output, even if not answered.\n"
     "6. Output a JSON object like this:\n"
     "{{\"evaluations\": [{{\"question\": \"1\", \"mark\": 4, \"total_mark\": 5}}], \"total_score\": 54}}\n"
     "Only output the JSON object. Do not include any explanation or extra text."
    ),
    ("human", "Student Answer:\n{student_answer}\n\nAnswer Key:\n{answer_key}\n\nQuestion Paper:\n{question_paper}")
])
evaluation_chain = eval_prompt | llm | parser

# == LangGraph ==
@dataclass
class EvalState:
    student_answer: str
    answer_key: str
    question_paper: str
    result: dict = None

def agent_node(state: EvalState):
    result = evaluation_chain.invoke({
        "student_answer": state.student_answer,
        "answer_key": state.answer_key,
        "question_paper": state.question_paper
    })
    return replace(state, result=result)

graph = StateGraph(state_schema=EvalState)
graph.add_node("evaluate", agent_node)
graph.set_entry_point("evaluate")
graph.add_edge("evaluate", END)
langgraph_agent = graph.compile()

# == Streamlit App ==
st.set_page_config(page_title="📘 Answer Evaluator", layout="centered")
st.title("📘 Student Answer Evaluation")

col1, col2, col3 = st.columns(3)
with col1:
    answer_key_file = st.file_uploader("📂 Upload Answer Key", type=["pdf", "docx"])
with col2:
    student_files = st.file_uploader("🧾 Upload Student Answer Sheets", type=["pdf", "docx"], accept_multiple_files=True)
with col3:
    question_file = st.file_uploader("📄 Upload Question Paper", type=["pdf", "docx"])

student_texts = [extract_file_content(f) for f in student_files] if student_files else []
answer_key_text = extract_file_content(answer_key_file) if answer_key_file else ""
question_text = extract_file_content(question_file) if question_file else ""

if 'evaluated_results' not in st.session_state:
    st.session_state.evaluated_results = None

if st.button("🚀 Start Evaluation") and answer_key_text and student_texts and question_text:
    # Only evaluate if not already done
    if st.session_state.evaluated_results is None:
        all_results = []
        for idx, student_text in enumerate(student_texts):
            student_no = f"Student No {idx+1}"

            state = EvalState(
                student_answer=student_text,
                answer_key=answer_key_text,
                question_paper=question_text
            )
            final_state = langgraph_agent.invoke(state)
            result = final_state.get("result", final_state)

            st.markdown(f"### {student_no}")
            if isinstance(result, dict) and "evaluations" in result:
                df = pd.json_normalize(result["evaluations"])
                df.insert(0, "Student No", student_no)
                # Add Total Marks column (sum of all marks for this student)
                total_marks = df["mark"].sum()
                df["Total Marks"] = total_marks
                all_results.append(df)
                st.dataframe(df, hide_index=True)
            else:
                st.warning("❗ No evaluation data returned for this student.")

        # Save results to session state
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            st.session_state.evaluated_results = final_df
    else:
        st.info("✅ Already evaluated. See results below.")

# Display results and download button if available
if st.session_state.evaluated_results is not None:
    st.subheader("All Evaluations")
    st.dataframe(st.session_state.evaluated_results, hide_index=True)
    excel_bytes = io.BytesIO()
    with pd.ExcelWriter(excel_bytes, engine='xlsxwriter') as writer:
        st.session_state.evaluated_results.to_excel(writer, index=False, sheet_name="Evaluations")
    st.download_button(
        label="⬇️ Download All Results as Excel",
        data=excel_bytes.getvalue(),
        file_name="all_evaluations.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
