# rag_system_improved.py
import os
from typing import List, Dict, Any, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_llm7 import ChatLLM7
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, StateGraph, MessagesState
from langchain_core.messages import HumanMessage, AIMessage, trim_messages, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from qdrant_client import QdrantClient


# Definiamo uno stato personalizzato che estende MessagesState
class RAGState(TypedDict):
    messages: List[BaseMessage]  # Tutti i messaggi della conversazione
    original_question: str
    optimized_query: str
    context: str
    conversation_context: List[BaseMessage]  # History per contesto conversazionale


class RAGSystemWithQueryGeneration:
    """Sistema RAG che genera query ottimizzate per la ricerca"""

    def __init__(self, collection_name: str, embedding_model: str = "gemini"):
        self.collection_name = collection_name
        self.embedding_model_type = embedding_model

        # Setup LLM
        # os.environ["GOOGLE_API_KEY"] = "AIzaSyC24Wq37BNGGbO-qDOR1MR28UQ93BPhOd4"
        # self.model = ChatGoogleGenerativeAI(model="gemini-3-flash", temperature=0.3)
        os.environ["LLM7_API_KEY"] = "hQftHz8uiu+HWzvXoI/aUppTi4rPQFP/QgAFez5mpirIWUVbBsh6KE2H7cFHuMdWqNhsQ5JOMsY8FfOohzqpKajYvwOxJYAB0oyxKhLjmLB4wJcSIK64oxu1zfzqXPsCPLG8oIGcUq5qyb0="
        self.model = ChatLLM7(
            # api_key="la_tua_api_key",  # la tua chiave qui
            base_url="https://api.llm7.io/v1",
            model="default"  # o "fast", "pro", oppure un model ID specifico
        )

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
            - Conversazione precedente: Utente: "Cosa sono le grammatiche context-free?"
            - Domanda attuale: "E per quelle regolari?"
            - Query: "definition regular grammars"

            - Conversazione precedente: Utente: "Mi spieghi l'algoritmo CYK"
            - Domanda attuale: "Come si applica alle grammatiche ambigue?"
            - Query: "CYK algorithm ambiguous grammars application"

            - Conversazione precedente: Utente: "Quali sono le proprietà di chiusura?"
            - Domanda attuale: "E per l'intersezione?"
            - Query: "closure properties intersection context-free languages"

            Ricorda: usa il contesto della conversazione per rendere la query più precisa e specifica."""),

            MessagesPlaceholder(variable_name="conversation_context"),

            ("human", "Domanda originale: {question}\n\nQuery ottimizzata (in inglese):")
        ])

        # Prompt per la risposta finale
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """Sei un assistente universitario esperto in Grammatiche Context Free.
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

    def _get_conversation_context(self, messages: List[BaseMessage], max_tokens: int = 1000) -> List[BaseMessage]:
        """
        Estrae il contesto conversazionale rilevante.
        Mantiene i messaggi più recenti entro il limite di token.
        """
        if len(messages) <= 1:
            return []

        # Prendi tutti i messaggi tranne l'ultimo (la domanda attuale)
        previous_messages = messages[:-1]

        return previous_messages

        # TODO: da capire se possiamo usarlo
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

        print("History:", conversation_context)

        # Invoca con conversation_context come variabile del prompt
        response = chain.invoke({
            "conversation_context": conversation_context,
            "question": question
        })

        optimized_query = response.content.strip()
        return optimized_query

    def _retrieve_relevant_docs(self, query: str, k: int = 4) -> List[str]:
        """Recupera documenti rilevanti"""
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            doc_contents = [doc.page_content for doc in docs]
            return doc_contents
        except Exception as e:
            print(f"[ERRORE] Errore nel recupero documenti: {e}")
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

            # Genera query ottimizzata CON contesto
            optimized_query = self._generate_optimized_query(
                original_question,
                conversation_context
            )

            # Recupera documenti
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
            Fase 2: Genera la risposta usando contesto e conversazione.
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
        print("Scrivi 'quit' per uscire")
        if debug_mode:
            print("Modalità debug: ON")
        else:
            print("Scrivi 'debug' per attivare debug")
        print("-" * 60)

        debug = debug_mode
        messages = []  # Inizializza la lista dei messaggi

        while True:
            question = input("\n📝 Q > ").strip()

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

            print("\n🔍 Ricerca in corso...")

            try:
                # Usa stream o invoke con update per mantenere lo stato
                final_state = self.app.invoke(
                    {"messages": messages},  # Passa TUTTA la history
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
                    print(f"\n🤖 A > {last_message.content}")
                else:
                    print("\n🤖 A > Nessuna risposta generata")

                print("-" * 60)

            except Exception as e:
                print(f"\n❌ Errore: {e}")

    def query(self, question: str, thread_id: str = "default_thread") -> Dict[str, Any]:
        """Esegue una singola query"""
        config = {"configurable": {"thread_id": thread_id}}
        input_message = HumanMessage(content=question)

        try:
            final_state = self.app.invoke(
                {"messages": [input_message]},
                config=config
            )

            result = {
                "answer": final_state.get('messages', [])[-1].content if final_state.get(
                    'messages') else "Nessuna risposta",
                "optimized_query": final_state.get("optimized_query", "N/A"),
                "context_sources": len(final_state.get("context", "").split("\n\n")),
                "conversation_context_length": len(final_state.get("conversation_context", [])),
                "success": True
            }

            return result

        except Exception as e:
            return {
                "answer": f"Errore: {e}",
                "optimized_query": "N/A",
                "context_sources": 0,
                "conversation_context_length": 0,
                "success": False
            }


def main():
    """Funzione principale"""
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
        print(f"\n❌ Errore: {e}")
    except Exception as e:
        print(f"\n❌ Errore imprevisto: {e}")


if __name__ == "__main__":
    main()