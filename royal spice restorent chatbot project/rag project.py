# ============================================================
#             HYBRID RAG WITH CHROMA & KEYWORD SEARCH
# ============================================================
#
# WHAT THIS PROJECT DOES:
#
# This is a Hybrid Retrieval Augmented Generation (RAG) system.
#
# Workflow:
#
# User Question
#       ↓
# LLM generates 2 sub-questions from user question
#       ↓
# Convert ALL questions into embedding vectors
#       ↓
# Search similar chunks from Chroma Vector DB (Semantic Relevance Check)
#       ↓
# Search matching chunks from JSON dictionary (Keyword Matching)
#       ↓
# Deduplicate and merge all retrieved chunks (Semantic + Keyword)
#       ↓
# Send context + ORIGINAL question to LLM
#       ↓
# Generate grounded answer
#
# ============================================================

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from datetime import datetime
import os
import sys
import json
import re
import shutil
import concurrent.futures

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ============================================================
#                    LOAD JSON KNOWLEDGE BASE
# ============================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
kb_path = os.path.join(current_dir, "knowledge_base.json")

print(f"Loading knowledge base from {kb_path}...\n")
if not os.path.exists(kb_path):
    print("Warning: knowledge_base.json not found! Creating an empty one.")
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump({}, f)

with open(kb_path, "r", encoding="utf-8") as f:
    kb_dict = json.load(f)

# ============================================================
#                    CREATE EMBEDDING MODEL
# ============================================================

print("Loading embedding model...\n")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model loaded.\n")

persist_dir = os.path.join(current_dir, "chroma_db")

# Clear the database folder on startup to ensure a completely clean build
if os.path.exists(persist_dir):
    try:
        shutil.rmtree(persist_dir)
        print("Deleted existing chroma_db directory on startup to ensure a clean rebuild.")
    except Exception as e:
        print(f"Note: Could not clear existing chroma_db directory on startup: {e}")

# ============================================================
#                    INDEX DOCUMENTS IN CHROMA
# ============================================================

print("\nCreating chunks from JSON dictionary keys...\n")
docs = []
for key, val in kb_dict.items():
    content = f"{key}:\n{val}"
    docs.append(Document(
        page_content=content, 
        metadata={"source": "knowledge_base.json", "key": key}
    ))

print(f"Total chunks created: {len(docs)}\n")

print("Creating Chroma vector database...\n")
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=persist_dir
)
print("Chroma vector database created.\n")

# ============================================================
#                    CREATE RETRIEVER
# ============================================================

print("Creating retriever...\n")
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}     # fetch top 3 chunks per query
)
print("Retriever ready.\n")

# ============================================================
#                    CREATE LLM
# ============================================================

print("Loading LLM...\n")
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.5
)
print("LLM loaded.\n")

# ============================================================
#                    KEYWORD SEARCH LOGIC
# ============================================================

def keyword_search(query, kb_dict, top_n=2):
    """
    Search the JSON knowledge base using custom keyword matching.
    Only matches keywords against dictionary keys for performance.
    """
    # Normalize query: lowercase and find alphanumeric words
    query_words = set(re.findall(r'\w+', query.lower()))
    
    # Common stop words to exclude from keyword match
    stop_words = {
        'is', 'now', 'the', 'restaurant', 'opened', 'open', 'close', 'closed',
        'are', 'and', 'or', 'a', 'an', 'to', 'in', 'of', 'for', 'on', 'with', 
        'at', 'by', 'from', 'what', 'which', 'who', 'where', 'when', 'how',
        'do', 'does', 'did', 'you', 'i', 'we', 'they', 'he', 'she', 'it', 'me',
        'my', 'your', 'us', 'our', 'what is', 'tell', 'show', 'give', 'please'
    }
    
    keywords = query_words - stop_words
    if not keywords:
        keywords = query_words # Fallback if all words were stop words

    results = []
    for key, val in kb_dict.items():
        score = 0
        key_lower = key.lower()
        
        # Check matches in keys only
        for word in keywords:
            word_pattern = r'\b' + re.escape(word) + r'\b'
            
            # Exact word match in key: highest score
            if re.search(word_pattern, key_lower):
                score += 5
            elif word in key_lower:
                score += 3
                
        if score > 0:
            results.append((score, key, val))
            
    # Sort by score descending
    results.sort(key=lambda x: x[0], reverse=True)
    
    # Return as LangChain Document objects
    matched_docs = []
    for score, key, val in results[:top_n]:
        content = f"{key}:\n{val}"
        matched_docs.append(Document(
            page_content=content, 
            metadata={"source": "keyword_match", "key": key, "score": score}
        ))
    return matched_docs

# ============================================================
#                    CREATE SUB-QUESTIONS
# ============================================================

SUB_QUESTION_PROMPT = """You are a search query generator for a restaurant chatbot.
Your job is to generate search queries that help find relevant information from the restaurant's knowledge base (menu, prices, dishes, ingredients, offers, etc.).
Given the customer's message, generate 2 different search queries about the restaurant that would retrieve useful context.
Write only the 2 or 3 queries, one per line, no numbering, no extra text.

Examples:
  User: which is cheaper, chicken biryani or mutton biryani?
  querie 1: What is the price of chicken biryani?
  querie 2: What is the price of mutton biryani?

  User : is now the restorent is opend?
  Sub-Q 1  : What are the working hours of the restaurant?
  Sub-Q 2  : what is current time?
 

Customer message: {question}    

2 restaurant-related search queries:"""

def generate_sub_questions(question):
    prompt = SUB_QUESTION_PROMPT.format(question=question)
    response = llm.invoke(prompt)
    lines = [line.strip() for line in response.content.strip().split("\n") if line.strip()]
    return lines 

# ============================================================
#                    CHAT LOOP / RESPONSE GENERATION
# ============================================================

def run_keyword_search(query, sub_questions, kb_dict):
    keyword_docs = keyword_search(query, kb_dict, top_n=2)
    for sq in sub_questions:
        keyword_docs.extend(keyword_search(sq, kb_dict, top_n=1))
    return keyword_docs

def run_vector_search(all_questions):
    vector_docs = []
    for q in all_questions:
        docs = retriever.invoke(q)
        vector_docs.extend(docs)
    return vector_docs


def get_bot_response(query, history):
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

    # Generate sub-questions
    sub_questions = generate_sub_questions(query)
    all_questions = [query] + sub_questions

    # Print queries used
    print("\n\033[96m[Multi-Query] All queries used for retrieval:")
    print(f"  Original : {query}")
    for i, q in enumerate(sub_questions, 1):
        print(f"  Sub-Q {i}  : {q}")
    print("\033[0m")

    # Run searches parallelly using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_keyword = executor.submit(run_keyword_search, query, sub_questions, kb_dict)
        future_vector = executor.submit(run_vector_search, all_questions)
        
        keyword_docs = future_keyword.result()
        vector_docs = future_vector.result()

    # --------------------------------------------------------
    # DISPLAY SEARCH RESULTS SEPARATELY IN THE TERMINAL
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("        PARALLEL SEARCH ENGINE DIAGNOSTICS")
    print("=" * 60)

    # 1. Keyword Search Output
    print("\n\033[93m[1. Keyword Search (Keys Only)] Chunks found:")
    if not keyword_docs:
        print("  None")
    for idx, doc in enumerate(keyword_docs, 1):
        key = doc.metadata.get("key", "N/A")
        score_info = f" (Score: {doc.metadata['score']})" if "score" in doc.metadata else ""
        print(f"  Match {idx}: Key: {key}{score_info}")
        print("  Content:")
        try:
            print(f"    {doc.page_content.replace(chr(10), chr(10) + '    ')}")
        except UnicodeEncodeError:
            safe_content = doc.page_content.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8')
            print(f"    {safe_content.replace(chr(10), chr(10) + '    ')}")
        print("  " + "-" * 50)
    print("\033[0m")

    # 2. Vector Search Output
    print("\n\033[95m[2. Vector Search (Semantic)] Chunks found:")
    if not vector_docs:
        print("  None")
    for idx, doc in enumerate(vector_docs, 1):
        key = doc.metadata.get("key", "N/A")
        print(f"  Match {idx}: Key: {key}")
        print("  Content:")
        try:
            print(f"    {doc.page_content.replace(chr(10), chr(10) + '    ')}")
        except UnicodeEncodeError:
            safe_content = doc.page_content.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8')
            print(f"    {safe_content.replace(chr(10), chr(10) + '    ')}")
        print("  " + "-" * 50)
    print("\033[0m")

    print("=" * 60 + "\n")

    # Determine response mode and final chunks
    # Merge and deduplicate by key name
    seen_keys = set()
    all_chunks = []

    # Prioritize keyword matches first
    for doc in keyword_docs + vector_docs:
        key_id = doc.metadata.get("key") or doc.page_content
        if key_id not in seen_keys:
            seen_keys.add(key_id)
            all_chunks.append(doc)

    # Cap to max 3 chunks
    all_chunks = all_chunks[:3]
    context = "\n\n".join([doc.page_content for doc in all_chunks])
    print(f"\033[94m[Response Mode] Answer taken from retrieved chunks (using {len(all_chunks)} unique context chunks).\033[0m\n")

    # Send to LLM
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
    print("        HYBRID RAG CHATBOT WITH CHROMA STARTED")
    print("=" * 60)

    print("\nType 'exit' to quit.\n")

    history = []

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("\nGoodbye.\n")
            break
        
        history.append({"role": "user", "content": query})
        print("\nSearching relevant knowledge...\n")

        answer, source_docs = get_bot_response(query, history)
        
        history.append({"role": "assistant", "content": answer})
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
            print(f"\nChunk {i}:")
            try:
                print(doc.page_content)
            except UnicodeEncodeError:
                print(doc.page_content.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))
            print("\n" + "-" * 60)
