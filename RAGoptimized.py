import os
from typing import List, Dict, Any, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, StateGraph, MessagesState
from langchain_core.messages import HumanMessage, AIMessage, trim_messages, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from qdrant_client import QdrantClient


# Definiamo uno stato personalizzato che estende MessagesState
class RAGState(TypedDict):
    messages: List[BaseMessage]  # tutti i messaggi della conversazione
    original_question: str # domanda originale
    optimized_query: str # domanda ottimizzata
    context: str # chunk più rilevanti rispetto alla query ottimizzata
    conversation_context: List[BaseMessage]  # contesto conversazionale da inserire nel prompt per la generazione della query ottimizzata


class RAGSystemWithQueryGeneration:
    """Sistema RAG che genera query ottimizzate per la ricerca"""

    def __init__(self, collection_name: str, embedding_model: str = "gemini"):
        self.collection_name = collection_name
        self.embedding_model_type = embedding_model

        # Setup LLM
        os.environ["GOOGLE_API_KEY"] = "AIzaSyDVw4dD0bYpQWYspzX3lajwn9q2kSY_hLY"
        self.model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

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

        if not self.client.collection_exists(self.collection_name):
            raise ValueError(f"La collezione '{self.collection_name}' non esiste in Qdrant.")

        # Inizializza i prompt
        self._setup_query_generation_prompts()

        # Inizializza il workflow
        self._setup_workflow()

    def _setup_embeddings(self):
        """Configura il modello di embedding"""
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

    def _setup_query_generation_prompts(self):
        """Configura i prompt per la generazione di query"""

        # Prompt per generare query con contesto conversazionale
        self.query_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """Sei un esperto di ricerca di informazioni. Il tuo compito è trasformare 
            l'ultima domanda dell'utente in una query di ricerca ottimizzata per un sistema di recupero 
            vettoriale.

            **ISTRUZIONI:**
            1. Analizza l'intera conversazione per comprendere il contesto
            2. Risolvi pronomi e riferimenti (es: "questo", "quello", "l'esempio sopra")
            3. Estrai i concetti chiave dalla domanda attuale E dalla conversazione precedente
            4. **TRADUCI SEMPRE LA QUERY IN INGLESE**
            5. Restituisci SOLO la query ottimizzata in inglese

            **ESEMPI CON CONTESTO:**
            - Conversazione precedente: Utente: "Cosa sono le reti RNN?"
            - Domanda attuale: "E come vengono addestrate?"
            - Query: "RNN training process"

            - Conversazione precedente: Utente: "In cosa consiste la tecnica di dropout?"
            - Domanda attuale: "Che vantaggi ci garantisce?"
            - Query: "Dropout technique advantages"

            - Conversazione precedente: Utente: "Gli MLP preservano informazioni spaziali dell'immagine in input?"
            - Domanda attuale: "E le CNN?"
            - Query: "Mantaining of spatial information for CNN"

            Ricorda: usa il contesto della conversazione per rendere la query più precisa e specifica."""),

            MessagesPlaceholder(variable_name="conversation_context"),

            ("human", "Domanda originale: {question}\n\nQuery ottimizzata (in inglese):")
        ])

        # Prompt per la risposta finale
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """Sei un assistente universitario esperto in Reti Neurali.
            Rispondi alla domanda basandoti ESCLUSIVAMENTE sul contesto fornito.

            Se il contesto non contiene informazioni sufficienti, rispondi:
            "Non ho trovato informazioni sufficienti nel materiale fornito per rispondere a questa domanda."

            IMPORTANTE: Non inventare informazioni. Usa solo ciò che trovi nel contesto."""),

            MessagesPlaceholder(variable_name="conversation_context"),

            ("human", """Contesto recuperato:
            {context}

            Domanda dell'utente: {original_question}

            Risposta:""")
        ])

    def _get_conversation_context(self, messages: List[BaseMessage], max_tokens: int = 600) -> List[BaseMessage]:
        """
        Estrae il contesto conversazionale.
        Mantiene i messaggi più recenti entro il limite di token.
        """
        if len(messages) <= 1:
            return []

        # Prendi tutti i messaggi tranne l'ultimo (la domanda attuale)
        previous_messages = messages[:-1]

        # Usa trim_messages per gestire i token
        trimmer = trim_messages(
            max_tokens=max_tokens,
            strategy="last",  # Mantiene i messaggi più recenti
            token_counter=self.model,
            include_system=False,
            allow_partial=True
        )

        trimmed_context = trimmer.invoke(previous_messages)
        return trimmed_context

    def _generate_optimized_query(self, question: str, conversation_context: List[BaseMessage]) -> str:
        """
        Genera una query ottimizzata usando il contesto della conversazione.
        """
        # Crea la chain con il prompt che include la conversation_context
        chain = self.query_generation_prompt | self.model

        # Invoca con conversation_context come variabile del prompt
        response = chain.invoke({
            "conversation_context": conversation_context,
            "question": question
        })

        optimized_query = response.content.strip()
        return optimized_query

    def _retrieve_relevant_docs(self, query: str, k: int = 4) -> List[str]:
        """Recupera documenti (chunk) rilevanti dal vector store"""
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            doc_contents = [doc.page_content for doc in docs]
            return doc_contents
        except Exception as e:
            print(f"Errore nel recupero documenti: {e}")
            return ["Errore nel recupero documenti."]

    def _setup_workflow(self):
        """Configura il grafo di workflow"""
        workflow = StateGraph(state_schema=RAGState)

        def generate_query_and_retrieve(state: RAGState):
            """
            Fase 1: Estrae contesto, genera query e recupera documenti.
            """
            messages = state.get('messages', [])
            if not messages:
                return {
                    "messages": [],
                    "original_question": "",
                    "optimized_query": "",
                    "context": "",
                    "conversation_context": []
                }

            last_message = messages[-1]
            original_question = last_message.content

            # Estrai il contesto conversazionale
            conversation_context = self._get_conversation_context(messages)

            # Genera query ottimizzata CON contesto conversazionale
            optimized_query = self._generate_optimized_query(
                original_question,
                conversation_context
            )

            # Recupera documenti rilevanti rispetto alla query ottimizzata
            doc_contents = self._retrieve_relevant_docs(optimized_query, k=4)
            context_text = "\n\n".join(doc_contents)

            return {
                "messages": messages,
                "original_question": original_question,
                "optimized_query": optimized_query,
                "context": context_text,
                "conversation_context": conversation_context
            }

        def generate_answer(state: RAGState):
            """
            Fase 2: Genera la risposta usando contesto (chunk rilevanti) e contesto conversazionale.
            """
            original_question = state.get("original_question", "")
            context_text = state.get("context", "")
            conversation_context = state.get("conversation_context", [])
            messages = state.get("messages", [])

            # Crea la chain per la risposta
            answer_chain = self.answer_prompt | self.model

            # Genera la risposta
            response = answer_chain.invoke({
                "conversation_context": conversation_context,
                "context": context_text,
                "original_question": original_question
            })

            # Aggiungi la risposta ai messaggi
            new_messages = messages + [response]

            return {"messages": new_messages}

        # Aggiungi nodi e edge
        workflow.add_node("query_generator", generate_query_and_retrieve)
        workflow.add_node("answer_generator", generate_answer)

        workflow.add_edge(START, "query_generator")
        workflow.add_edge("query_generator", "answer_generator")

        # Compila l'app
        self.app = workflow.compile(checkpointer=MemorySaver())

    def chat(self, thread_id: str = "default_thread", debug_mode: bool = False):
        """Avvia una chat interattiva"""
        config = {"configurable": {"thread_id": thread_id}}

        print(f"\n{'=' * 60}")
        print(f"Chatbot RAG Avviato (con contesto conversazionale)")
        print(f"Collezione: {self.collection_name}")
        print(f"Embedding: {self.embedding_model_type}")
        print(f"{'=' * 60}")
        print("Scrivi 'quit' oppure 'exit' per uscire")
        if debug_mode:
            print("Modalità debug: ON")
        else:
            print("Scrivi 'debug' per attivare debug")
        print("-" * 60)

        debug = debug_mode
        messages = []  # Inizializza la lista dei messaggi

        while True:
            question = input("\n Q > ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("Arrivederci!")
                break

            if question.lower() == "debug":
                debug = not debug
                print(f"Modalità debug {'attivata' if debug else 'disattivata'}")
                continue

            if not question:
                continue

            input_message = HumanMessage(content=question)
            messages.append(input_message)  # Aggiungi alla history

            print("\n Ricerca in corso...")

            try:
                # Invocazione della app
                final_state = self.app.invoke(
                    {"messages": messages},  # Passa tutta la history
                    config=config
                )

                # Recupera la risposta aggiornata
                messages = final_state.get('messages', [])

                # Mostra debug se attivo
                if debug:
                    print(f"\n[DEBUG] Query ottimizzata: {final_state.get('optimized_query', 'N/A')}")
                    print(
                        f"[DEBUG] Contesto conversazionale: {len(final_state.get('conversation_context', []))} messaggi")
                    if final_state.get('context'):
                        print(f"[DEBUG] Documenti recuperati: {len(final_state.get('context', '').split('\\n\\n'))}")

                # Recupera e mostra la risposta
                if messages:
                    last_message = messages[-1]
                    print(f"\n A > {last_message.content}")
                else:
                    print("\n A > Nessuna risposta generata")

                print("-" * 60)

            except Exception as e:
                print(f"\n Errore: {e}")

def main():
    print("=== Sistema RAG con Contesto Conversazionale ===\n")

    # Configurazione
    collection_name = input("Inserisci il nome della collezione Qdrant: ").strip()

    print("\nScegli il modello di embedding:")
    print("1. Gemini (gemini-embedding-001)")
    print("2. HuggingFace (all-MiniLM-L6-v2)")

    embed_choice = input("Scelta [1/2]: ").strip()
    embedding_model = "gemini" if embed_choice == "1" else "hf"

    debug_choice = input("\nAttivare modalità debug? (s/n): ").strip().lower()
    debug_mode = debug_choice == 's'

    # Inizializzazione e chat
    try:
        rag_system = RAGSystemWithQueryGeneration(
            collection_name=collection_name,
            embedding_model=embedding_model
        )
        rag_system.chat(thread_id="user_1", debug_mode=debug_mode)
    except ValueError as e:
        print(f"\n Errore: {e}")
    except Exception as e:
        print(f"\n Errore imprevisto: {e}")


if __name__ == "__main__":
    main()