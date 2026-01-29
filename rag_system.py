import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from qdrant_client import QdrantClient


class RAGSystem:
    """Sistema RAG con memoria per l'inferenza"""

    def __init__(self, collection_name: str, embedding_model: str = "gemini"):
        """
        Inizializza il sistema RAG.

        Args:
            collection_name: Nome della collezione Qdrant da utilizzare
            embedding_model: 'gemini' per GoogleGenerativeAI o 'hf' per HuggingFace
        """
        self.collection_name = collection_name
        self.embedding_model_type = embedding_model

        # Setup LLM
        os.environ["GOOGLE_API_KEY"] = "AIzaSyC24Wq37BNGGbO-qDOR1MR28UQ93BPhOd4"
        self.model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        # Setup embeddings
        self._setup_embeddings()

        # Setup Qdrant client
        self.client = QdrantClient(url="http://localhost")

        # Setup vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )

        # Verifica che la collezione esista
        if not self.client.collection_exists(self.collection_name):
            raise ValueError(f"La collezione '{self.collection_name}' non esiste in Qdrant. "
                             "Usa create_collection.py per crearla.")

        # Inizializza il workflow
        self._setup_workflow()

    def _setup_embeddings(self):
        """Configura il modello di embedding in base alla scelta"""
        if self.embedding_model_type == "gemini":
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            print("Usando il modello embedding: gemini-embedding-001")
        elif self.embedding_model_type == "hf":
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )
            print("Usando il modello embedding: all-MiniLM-L6-v2")
        else:
            raise ValueError("embedding_model deve essere 'gemini' o 'hf'")

    def _setup_workflow(self):
        """Configura il grafo di workflow con memoria"""
        # Prompt template
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", """Sei un assistente universitario esperto in Grammatiche Context Free.
                Rispondi alla domanda basandoti ESCLUSIVAMENTE sul contesto fornito qui sotto.
                Se non trovi la risposta nel contesto, dillo chiaramente."""),

                MessagesPlaceholder(variable_name="history"),

                ("human", """Context: {context}

                Question: {question}

                Answer:""")
            ]
        )

        # Trimmer per gestire la memoria
        trimmer = trim_messages(
            max_tokens=2000,
            strategy="last",
            token_counter=self.model,
            include_system=True,
            allow_partial=False,
            start_on="human"
        )

        # Definizione del workflow
        workflow = StateGraph(state_schema=MessagesState)

        def call_model(state: MessagesState):
            # Recupero l'ultimo messaggio dell'utente
            last_human_message = state['messages'][-1]
            question = last_human_message.content

            # Recupero il contesto (RAG)
            docs = self.vector_store.similarity_search(question, k=4)
            context_text = "\n\n".join([doc.page_content for doc in docs])

            # Gestione della memoria con trimmer
            previous_messages = state['messages'][:-1]
            trimmed_history = trimmer.invoke(previous_messages)

            # Creazione della chain
            chain = prompt_template | self.model

            response = chain.invoke({
                "history": trimmed_history,
                "context": context_text,
                "question": question
            })

            return {"messages": [response]}

        workflow.add_edge(START, "model")
        workflow.add_node("model", call_model)

        self.app = workflow.compile(checkpointer=MemorySaver())

    def chat(self, thread_id: str = "default_thread"):
        """
        Avvia una chat interattiva.

        Args:
            thread_id: ID del thread per la memoria della conversazione
        """
        config = {"configurable": {"thread_id": thread_id}}

        print(f"\n--- Chatbot Avviato (Collezione: {self.collection_name}) ---")
        print("Scrivi 'quit' per uscire")
        print("-" * 50)

        while True:
            question = input("\nQ > ")
            if question.lower() in ["quit", "exit"]:
                break

            input_message = HumanMessage(content=question)

            # Esegui l'inferenza
            for event in self.app.stream({"messages": [input_message]}, config=config):
                if "model" in event:
                    print("\nA > ", event["model"]["messages"][0].content)
                    print("-" * 50)

    def query(self, question: str, thread_id: str = "default_thread") -> str:
        """
        Esegue una singola query senza interfaccia interattiva.

        Args:
            question: La domanda da porre
            thread_id: ID del thread per la memoria

        Returns:
            La risposta del sistema
        """
        config = {"configurable": {"thread_id": thread_id}}
        input_message = HumanMessage(content=question)

        result = None
        for event in self.app.stream({"messages": [input_message]}, config=config):
            if "model" in event:
                result = event["model"]["messages"][0].content

        return result


if __name__ == "__main__":
    # Esempio di utilizzo
    print("=== Sistema RAG con Memoria ===\n")

    # Scelta della collezione
    collection_name = input("Inserisci il nome della collezione Qdrant da utilizzare: ").strip()

    # Scelta del modello di embedding
    embedding_model = input("\nScegli il modello di embedding:\n"
                         "1. [gemini] Gemini (gemini-embedding-001)\n"
                         "2. [hf]     HuggingFace (all-MiniLM-L6-v2)\n"
                         ).strip()

    # Inizializzazione e chat
    try:
        rag_system = RAGSystem(collection_name=collection_name, embedding_model=embedding_model)
        rag_system.chat(thread_id="user_1")
    except ValueError as e:
        print(f"Errore: {e}")
        print("Assicurati che Qdrant sia in esecuzione e che la collezione esista.")