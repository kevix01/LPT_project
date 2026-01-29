import os
import getpass
import pprint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
#from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver

# selezione LLM (modello di gemini)
os.environ["GOOGLE_API_KEY"] = "AIzaSyCkiBHbt_ZqGqAUodkyEK-E4fdni_PWYIY"
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# --- SETUP EMBEDDINGS ---
# scelta del modello da usare per creare gli emedding
flag_embedding = input("Scegli:\n'g' per usare il modello embedding 'gemini-embedding-001',\n'h'per usare il modello embedding  'all-MiniLM-L6-v2' di Hugging Face.\n")
if flag_embedding == "g":
    # selezione modello di gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
else:
    # Usiamo all-MiniLM-L6-v2: veloce e potente per RAG
    print("Caricamento modello di embedding locale...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}, # Usa 'cuda' se hai una GPU NVIDIA
        encode_kwargs={'normalize_embeddings': False}
    )

# -- CARICAMENTO DOCUMENTI PDF --
file_path = "./05 - Context Free Languages.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
# - stampa metadati del pdf caricato
#pprint.pp(docs[0].metadata)
# - stampa contenuto di una pagina
#print(docs[0].page_content[:10]) #stampa primi 10 caratteri della prima pagina

# splitting documenti PDF
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)
print(f"Documento diviso in {len(all_splits)} chunk.")
# print(all_splits[8])

# -- CREAZIONE VECTOR STORE --
if flag_embedding=="g":
    print("Gli embedding vengono creati usando il modello: ",embeddings.model)
    vector_store_dump_file = "vector_store_gemini.dump"
else:
    print("Gli embedding vengono creati usando il modello 'all-MiniLM-L6-v2' di hugging face")
    vector_store_dump_file = "vector_store_hf.dump"

if os.path.exists(vector_store_dump_file): # verifica se il vector store dump è già stato creato
    # caricamento vector_store da file
    vector_store = InMemoryVectorStore.load(vector_store_dump_file,embeddings)
    print(f"Vector store caricato da {vector_store_dump_file}")
else:
    # creazione vector store
    vector_store = InMemoryVectorStore(embeddings)
    # Crea una lista di ID
    ids = [f"doc_{i}" for i in range(len(all_splits))]
    document_ids = vector_store.add_documents(documents=all_splits, ids=ids)
    print(f"Vector store creato")
    vector_store.dump(vector_store_dump_file)
    print(f"Vector store dump to {vector_store_dump_file}")
#print(document_ids)

""" # -- Implementazione chatbot RAG con memoria --
# Define a new graph with a specific format for state (memory)
# MessagesState is a predefined schema for storing messages
workflow = StateGraph(state_schema=MessagesState)

# define a trimmer to avoid long prompts
# when history contains many message
trimmer = trim_messages(
    max_tokens = 100,       # limit in tokens
    strategy = "last",      # keep last messages
    token_counter = model,  # the model provides token count
    allow_partial=False,    # do not trunctae messages
    start_on="human"        # first message must be by human
)

# function excuted by node
def call_model(state: MessagesState):
    trimmed_messages = trimmer.invoke(state['messages'])
    response = model.invoke(trimmed_messages)
    return {"messages": response}

# define an execution graph with only one node (model)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# add checkpoint to save state in memory
app = workflow.compile(checkpointer=MemorySaver())

# configuration to link execution to a thread_id
config = {"configurable": {"thread_id": "user1"}}

# build template for RAG query
# template has 2 placeholders
# question - for the user query
# context - for the retrieved context
prompt_template = ChatPromptTemplate.from_messages(
    [("system", Sei un assistente universitario. Hai accesso a un tool che recupera il contesto da una dispensa riguardante le grammatiche context free. Usa il tool per cercare informazioni e POI rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE sui documenti trovati.), ("user",Question: {question} Context: {context} Answer:)])
# human-loop
while True:
    # get user input
    question = input("Q > ")
    # exit loop if 'quit' or 'exit' command is given
    if question=="quit" or question=="exit":
        break

    #search for similar contexts
    context = vector_store.similarity_search(question, k=4)

    # instantiate the prompt with placeholder values
    prompt = app.invoke(
        {"question": question,
          "context": "\n".join(doc.page_content for doc in context)})
    # call model
    response = model.invoke(prompt)
    # print answer
    print("A > ",response.content) """

# --- 2. DEFINIZIONE DEL PROMPT TEMPLATE ---
# Aggiungiamo MessagesPlaceholder per gestire la memoria (chat history)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """Sei un assistente universitario esperto in Grammatiche Context Free.
        Rispondi alla domanda basandoti ESCLUSIVAMENTE sul contesto fornito qui sotto.
        Se non trovi la risposta nel contesto, dillo chiaramente."""),
        
        # Qui verrà inserita la cronologia dei messaggi (memoria)
        MessagesPlaceholder(variable_name="history"),
        
        # L'ultimo messaggio contiene il contesto recuperato e la domanda
        ("human", """Context: {context}
        
        Question: {question}
        
        Answer:""")
    ]
)

# --- 3. DEFINIZIONE DEL GRAFO ---

workflow = StateGraph(state_schema=MessagesState)

# Trimmer per evitare di superare il limite di token
trimmer = trim_messages(
    max_tokens=2000, # Aumentato leggermente per includere contesto
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

def call_model(state: MessagesState):
    # 1. Recupero l'ultimo messaggio dell'utente
    last_human_message = state['messages'][-1]
    question = last_human_message.content
    
    # 2. Recupero il contesto relativo alla domanda dell'utente (RAG)
    docs = vector_store.similarity_search(question, k=4)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    # 3. Gestione della Memoria con il Trimmer
    # Trimmamo i messaggi PRECEDENTI all'attuale (la history)
    previous_messages = state['messages'][:-1]
    trimmed_history = trimmer.invoke(previous_messages)
    
    # 4. Creazione della Chain (Prompt -> Model)
    # Passiamo al template: history (messaggi vecchi), context (doc trovati), question (domanda attuale)
    chain = prompt_template | model
    
    response = chain.invoke({
        "history": trimmed_history,
        "context": context_text,
        "question": question
    })
    
    # Restituiamo la risposta che verrà aggiunta allo stato (memoria)
    return {"messages": [response]}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

app = workflow.compile(checkpointer=MemorySaver())

# --- 4. ESECUZIONE (Chat Loop) ---

config = {"configurable": {"thread_id": "studente_1"}}

print("\n--- Chatbot Avviato (scrivi 'quit' per uscire) ---")

while True:
    question = input("Q > ")
    if question.lower() in ["quit", "exit"]:
        break

    # Invio il messaggio al Grafo
    # Il grafo ora si occupa di cercare i documenti e formattare il prompt
    input_message = HumanMessage(content=question)
    
    # stream o invoke
    for event in app.stream({"messages": [input_message]}, config=config):
        # Stampiamo l'output del modello quando arriva
        if "model" in event:
            print("A > ", event["model"]["messages"][0].content)
            print("-" * 50)