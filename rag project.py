# ============================================================
#                    REAL RAG WITH CHROMA
# ============================================================
#
# WHAT THIS PROJECT DOES:
#
# This is a REAL Retrieval Augmented Generation (RAG) system.
#
# Workflow:
#
# User Question
#       ↓
# LLM generates 2 sub-questions from user question
#       ↓
# Convert ALL questions into embedding vectors
#       ↓
# Search similar chunks from Chroma Vector DB for each question
#       ↓
# Deduplicate and merge all retrieved chunks into one context
#       ↓
# Send context + ORIGINAL question to LLM
#       ↓
# Generate grounded answer
#
#
# ============================================================
#                    REQUIRED INSTALLATIONS
# ============================================================
#
# pip install langchain
# pip install langchain-community
# pip install langchain-huggingface
# pip install langchain-chroma
# pip install chromadb
# pip install langchain-groq
# pip install sentence-transformers
# pip install python-dotenv
#

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

from langchain_groq import ChatGroq

from datetime import datetime
import os
import sys

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


# ============================================================
#                    STEP 3: CREATE EMBEDDING MODEL
# ============================================================


print("Loading embedding model...\n")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.\n")

persist_dir = "./chroma_db"
if not os.path.exists(persist_dir):

# ========================loading documents========================

    print("\nLoading documents...\n")

    loader = TextLoader(
        "knowledge_base.txt",
        encoding="utf-8"
    )
    documents = loader.load()

    print("Documents loaded successfully.\n")

    # ============================================================
    #                    STEP 2: SPLIT DOCUMENTS
    # ============================================================

    print("Splitting documents into chunks...\n")

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=300,
        chunk_overlap=60
    )
    
    docs = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(docs)}\n")

    # ============================================================
    #                    STEP 4: CREATE CHROMA VECTOR DATABASE
    # ============================================================


    print("Creating Chroma vector database...\n")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    print("Chroma vector database created.\n")
else:
    print("Loading existing Chroma vector database...\n")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model
    )


# ============================================================
#                    STEP 5: CREATE RETRIEVER
# ============================================================


print("Creating retriever...\n")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}     # fetch top 3 chunks per query
)

print("Retriever ready.\n")

# ============================================================
#                    STEP 6: CREATE LLM
# ============================================================

print("Loading LLM...\n")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)

print("LLM loaded.\n")


# ============================================================
#                    STEP 7: CREATE SUB-QUESTIONS
# ============================================================

SUB_QUESTION_PROMPT = """You are a search query generator for a restaurant chatbot.
Your job is to generate search queries that help find relevant information from the restaurant's knowledge base (menu, prices, dishes, ingredients, offers, etc.).
Given the customer's message, generate 2 or 3 different search queries about the restaurant that would retrieve useful context.
Write only the 2 or 3 queries, one per line, no numbering, no extra text.

Examples:
  User: which is cheaper, chicken biryani or mutton biryani?
  querie 1: What is the price of chicken biryani?
  querie 2: What is the price of mutton biryani?

  User : is now the restorent is opend?
  Sub-Q 1  : What are the working hours of the restaurant?
  Sub-Q 2  : what is current time?
  Sub-Q 3  : Is the restaurant open today?

Customer message: {question}    

2 restaurant-related search queries:"""

def generate_sub_questions(question):
    prompt = SUB_QUESTION_PROMPT.format(question=question)
    response = llm.invoke(prompt)
    lines = [line.strip() for line in response.content.strip().split("\n") if line.strip()]
    return lines 


# ============================================================
#                    CHAT LOOP
# ============================================================

def get_bot_response(query, history):
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

    # --------------------------------------------------------
    # STEP A: Generate sub-questions from user's question
    # --------------------------------------------------------
    sub_questions = generate_sub_questions(query)
    all_questions = [query] + sub_questions   # original + 2 generated = 3 total

    # Print the generated sub-questions so you can see them
    print("\n\033[96m[Multi-Query] All queries used for retrieval:")
    print(f"  Original : {query}")
    for i, q in enumerate(sub_questions, 1):
        print(f"  Sub-Q {i}  : {q}")
    print("\033[0m")

    # --------------------------------------------------------
    # STEP B: Retrieve top-3 chunks for EACH question
    #         and deduplicate by content
    # --------------------------------------------------------
    seen_contents = set()
    all_chunks = []

    for q in all_questions:
        docs = retriever.invoke(q)
        for doc in docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                all_chunks.append(doc)

    #print(f"\033[96m[Multi-Query] Total unique chunks retrieved: {len(all_chunks)} → capped to 5\033[0m\n")

    # Cap to max 5 chunks to keep the context focused
    all_chunks = all_chunks[:5]

    # --------------------------------------------------------
    # STEP C: Build context from all unique chunks
    # --------------------------------------------------------
    context = "\n\n".join([doc.page_content for doc in all_chunks])
    dt=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # --------------------------------------------------------
    # STEP D: Send ONLY the original question + context to LLM
    # --------------------------------------------------------
    answer_prompt = f"""Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Chat History:
{history_str}

Context: {context}

Question: {query}

present date and time: {dt}
Answer:"""

    response = llm.invoke(answer_prompt)
    return response.content, all_chunks


if __name__ == "__main__":
    print("=" * 60)
    print("        REAL RAG CHATBOT WITH CHROMA STARTED")
    print("=" * 60)

    print("\nType 'exit' to quit.\n")

    history=[{"role":"system","content":"you are the chatbot of Royal Spice Resturant"}]

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("\nGoodbye.\n")
            break
        
        
        history.append({"role":"user","content":query})
        print("\nSearching relevant knowledge...\n")

        answer, source_docs = get_bot_response(query, history)
        
        history.append({"role":"assistant","content":answer})
        print("\033[93mmodel response:", end=" ")
        try:
            print(answer)
        except UnicodeEncodeError:
            print(answer.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))
        print("\033[0m")

        # ========================================================
        # RETRIEVED CHUNKS
        # ========================================================

        print("\n")
        print("=" * 60)
        print("RETRIEVED CONTEXT CHUNKS")
        print("=" * 60)

        for i, doc in enumerate(source_docs, start=1):

            print(f"\nChunk {i}:\n")
            try:
                print(doc.page_content)
            except UnicodeEncodeError:
                print(doc.page_content.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))
            print("\n" + "-" * 60)
