# Load necessary packages
import os
import streamlit as st
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load the api keys into environmnet variable
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]

# Create a temp directory in current folder
os.makedirs("temp", exist_ok=True)


# Load the groq model and cohere
embedding_model = CohereEmbeddings(model="embed-english-v3.0")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5, max_tokens=500)


# load and split pdf
def load_and_split_pdf(path: str):
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


# Create and store vector embeddings
@st.cache_resource(show_spinner="creating vectorstore ...")
def get_vectorstore(splits, persist_dir):
    return Chroma.from_documents(
        splits, embedding=embedding_model, persist_directory=persist_dir
    )


# get model response
def get_response(retriever, query: str):
    system_prompt = (
        "You are a helpful assistant. Use the retrieved context to answer the question "
        "as completely as possible. If the answer is not in the context, say that you don't know, "
        "but always explain your reasoning based on the available information. "
        "Provide detailed answers and examples when relevant.\n\n{context}"
    )
    prompt = ChatPromptTemplate(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    stuff_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, stuff_chain)
    response = rag_chain.invoke({"input": query})
    return response["answer"]


def list_documents():
    return [
        folder.replace("chroma_db_", "")
        for folder in os.listdir(".")
        if folder.startswith("chroma_db_")
    ]


# --- Streamlit app ----

# Intialize blank streamlit page
st.set_page_config(page_title="RAG Project")
st.title("RAG Project")

# add db and selected doc into state
if "db" not in st.session_state:
    st.session_state.db = None
if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None

# Just create a query for user to input
query = st.text_input("Ask any question about document : ")
# Create a button to provide answer
submit = st.button("answer")

# Create a sidebar to upload pdfs
with st.sidebar:
    st.header("Upload documents here")
    # Upload a pdf
    uploaded_file = st.file_uploader("Upload pdf", type=["pdf"])
    if uploaded_file:
        temp_file = os.path.join("temp", uploaded_file.name)
        # Save pdf to temp folder
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getvalue())
        # Create a button to create embeddings
        if st.button("Create Embeddings"):
            with st.spinner("Processing ... "):
                splits = load_and_split_pdf(temp_file)
                persist_dir = f"./chroma_db_{uploaded_file.name}"
                st.session_state.db = get_vectorstore(splits, persist_dir)
                st.session_state.selected_doc = uploaded_file.name
                st.success(f"Embeddings for {uploaded_file.name} created")

        # List the current pdfs
        docs = list_documents()
        if docs:
            selected_doc = st.selectbox(
                "Selected Documents",
                docs,
                index = (
                    docs.index(st.session_state.selected_doc)
                    if st.session_state.selected_doc in docs
                    else 0
                )
            )
            st.session_state.selected_doc = selected_doc
            st.session_state.db = Chroma(
                persist_directory=f"./chroma_db_{selected_doc}",
                embedding_function=embedding_model,
            )
        else:
            st.info("No persisted documents found. Upload a PDF to get started.")

# -----------------
# Handle QA
# -----------------
if submit:
    if st.session_state.db is None:
        st.error(
            "Please select a document or upload a PDF and create embeddings first."
        )
    else:
        with st.spinner("Responding..."):
            retriever = st.session_state.db.as_retriever()
            response = get_response(retriever, query)
            st.write(response)