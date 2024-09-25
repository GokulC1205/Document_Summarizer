import streamlit as st
from PyPDF2 import PdfReader
import os
import requests
from pypdf import PdfReader
import tiktoken
from langchain.prompts import(
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import(
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
st.set_page_config(layout="wide")

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('API_KEY')
@st.cache_resource
def summarizer(file_path):
  text_body=""
  text_body=extract_text_from_pdf(file_path)

  def num_tokens_from_string(string: str,encoding_name:str) -> int:
    encoding=tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens=len(encoding.encode(string))
    return num_tokens
  num_tokens=num_tokens_from_string(text_body,"gpt-3.5-turbo")
  print(num_tokens)

  context_template="You are a helpful AI Resarcher that specializes in analyzing ML,AI and LLM papers.\
  Please use all your expertise to approach this task. Output your content in markdown format and include titles where relevant"

  system_message_prompt=SystemMessagePromptTemplate.from_template(context_template)
  human_template = "Please summarize this paper focusing on the key important takeaways for each section. \
  Expand the summary on methods so they can be clearly understood. \n\n PAPER: \n\n{paper_content}"
  human_message_prompt=HumanMessagePromptTemplate(
      prompt=PromptTemplate(
          template=human_template,
          input_variables=["Paper Content"]
      )
  )

  chat_prompt_template =ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])


  chat =ChatOpenAI(model_name="gpt-3.5-turbo",temperature=1,openai_api_key=api_key)

  summary_chain=LLMChain(llm=chat,prompt=chat_prompt_template)
  with get_openai_callback() as cb:
    output=summary_chain.run(text_body)

  return output

@st.cache_resource
def text_summary(text_body):
    def num_tokens_from_string(string: str,encoding_name:str) -> int:
        encoding=tiktoken.encoding_for_model("gpt-3.5-turbo")
        num_tokens=len(encoding.encode(string))
        return num_tokens
    num_tokens=num_tokens_from_string(text_body,"gpt-3.5-turbo")
    print(num_tokens)

    context_template="You are a helpful AI Resarcher that specializes in analyzing ML,AI and LLM papers.\
    Please use all your expertise to approach this task. Output your content in markdown format and include titles where relevant"

    system_message_prompt=SystemMessagePromptTemplate.from_template(context_template)
    human_template = "Please summarize this paper focusing on the key important takeaways for each section. \
    Expand the summary on methods so they can be clearly understood. \n\n PAPER: \n\n{paper_content}"

    human_message_prompt=HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=human_template,
            input_variables=["Paper Content"]
        )
    )

    chat_prompt_template =ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])


    chat =ChatOpenAI(model_name="gpt-3.5-turbo",temperature=1,openai_api_key=api_key)

    summary_chain=LLMChain(llm=chat,prompt=chat_prompt_template)
    with get_openai_callback() as cb:
        output=summary_chain.run(text_body)

    return output



def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    print(f"Number of pages:  {len(reader.pages)}")
    parts=[]
    def visitor_body(text,cm,tm,fontDict,fontSize):
        y=tm[5]
        if y>50 and y<720:
            parts.append(text)
    for page in reader.pages:
        page.extract_text(visitor_text=visitor_body)
    text_body="".join(parts)
    print("hi")
    return text_body

choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Summarize Document"])

if choice == "Summarize Text":
    st.subheader("Summarize Text using Langchain and ChatGPT")
    input_text = st.text_area("Enter your text here")
    if input_text is not None:
        if st.button("Summarize Text"):
            col1, col2 = st.columns([1,1])
            with col1:
                st.markdown("**Your Input Text**")
                st.info(input_text)
            with col2:
                st.markdown("**Summary Result**")
                result = text_summary(input_text)
                st.success(result)

elif choice == "Summarize Document":
    st.subheader("Summarize Document using Langchain and ChatGPT")
    input_file = st.file_uploader("Upload your document here", type=['pdf'])
    if input_file is not None:
        if st.button("Summarize Document"):
            with open("doc_file.pdf", "wb") as f:
                f.write(input_file.getbuffer())
            col1, col2 = st.columns([1,1])
            with col1:
                st.info("File uploaded successfully")
                extracted_text = extract_text_from_pdf("doc_file.pdf")
                st.markdown("**Extracted Text is Below:**")
                st.info(extracted_text)
            with col2:
                st.markdown("**Summary Result**")
                text = extract_text_from_pdf("doc_file.pdf")
                doc_summary = text_summary(text)
                st.success(doc_summary)