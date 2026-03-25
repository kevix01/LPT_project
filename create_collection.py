import os
import glob
import time
import math
from typing import List
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance


class CollectionCreator:
    """Crea e popola collezioni Qdrant con documenti PDF"""

    def __init__(self, embedding_model: str = "gemini"):
        """
        Inizializza il creatore di collezioni.

        Args:
            embedding_model: 'gemini' per GoogleGenerativeAI o 'hf' per HuggingFace
        """
        load_dotenv()

        self.embedding_model_type = embedding_model

        # Setup embeddings
        self._setup_embeddings()

        # Setup Qdrant client
        self.client = QdrantClient(url="http://localhost")

    def _setup_embeddings(self):
        """Configura il modello di embedding"""
        if self.embedding_model_type == "gemini":
            # Carica la chiave API da variabile d'ambiente (file .env)
            api_key = os.getenv("GOOGLE_API_KEY", "")
            if not api_key:
                raise ValueError("La variabile d'ambiente GOOGLE_API_KEY non è impostata. Inseriscila nel file .env.")
            os.environ["GOOGLE_API_KEY"] = api_key
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

    @staticmethod
    def load_pdfs_from_folder(folder_path: str = "./docs") -> List:
        """
        Carica tutti i PDF da una cartella.

        Args:
            folder_path: Percorso della cartella contenente i PDF

        Returns:
            Lista di documenti caricati, divisi in pagine
        """
        # Trova tutti i file PDF nella cartella
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(f"Nessun file PDF trovato in {folder_path}")

        print(f"Trovati {len(pdf_files)} file PDF:")
        for pdf in pdf_files:
            print(f"  - {os.path.basename(pdf)}")

        # Carica tutti i documenti
        all_docs = []
        for pdf_file in pdf_files:
            print(f"Caricamento: {os.path.basename(pdf_file)}...")
            loader = PDFPlumberLoader(pdf_file)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"  Caricate {len(docs)} pagine")

        return all_docs

    @staticmethod
    def split_documents(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
        """
        Divide i documenti in chunk.

        Args:
            documents: Lista di documenti suddivisi già in pagine, da dividere in chunk
            chunk_size: Dimensione di ogni chunk in caratteri
            chunk_overlap: Sovrapposizione tra chunk in caratteri

        Returns:
            Lista di chunk
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

        all_splits = text_splitter.split_documents(documents)
        print(f"Documenti divisi in {len(all_splits)} chunk")

        return all_splits
    
    def create_collection(self, collection_name: str, documents: List):
        """
        Crea una nuova collezione Qdrant e la popola con i chunk forniti.
        Processa i chunk in batch da 80 per rispettare i rate limits,
        attendendo 60 secondi tra un batch e l'altro.

        Args:
            collection_name: Nome della nuova collezione
            documents: Lista di chunk (Document objects) da indicizzare
        """
        # --- 1. Gestione esistenza collezione ---
        if self.client.collection_exists(collection_name):
            overwrite = input(f"La collezione '{collection_name}' esiste già. Sovrascriverla? (s/n): ")
            if overwrite.lower() != 's':
                print("Operazione annullata.")
                return
            else:
                self.client.delete_collection(collection_name)
                print(f"Collezione '{collection_name}' esistente eliminata.")

        # --- 2. Calcolo dimensione vettori ---
        try:
            # Test rapido per ottenere la dimensione
            sample_embedding = self.embeddings.embed_query("Query di prova")
            vector_size = len(sample_embedding)
        except Exception as e:
            print(f"Errore nel test degli embedding: {e}")
            return

        # --- 3. Creazione Collezione su Qdrant ---
        print(f"Creazione collezione '{collection_name}'...")
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

        # Inizializza il Vector Store
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings
        )

        # --- 4. Aggiunta documenti in Batch ---
        print("Inizio indicizzazione dei chunk in batch...")
        
        _BATCH_SIZE = 80
        total_chunks = len(documents)
        # Calcola quanti batch totali avremo (utile per i log)
        total_batches = math.ceil(total_chunks / _BATCH_SIZE)
        all_document_ids = []

        # Ciclo da 0 alla fine, avanzando di BATCH_SIZE alla volta
        for i in range(0, total_chunks, _BATCH_SIZE):
            # Seleziona i chunk da processare
            batch_docs = documents[i : i + _BATCH_SIZE]
            current_batch_num = (i // _BATCH_SIZE) + 1
            
            print(f"--> Processando batch {current_batch_num}/{total_batches} ({len(batch_docs)} chunk)...")

            try:
                # Aggiunge il batch corrente
                ids = vector_store.add_documents(documents=batch_docs)
                all_document_ids.extend(ids)
                print(f"    Batch {current_batch_num} completato.")
            except Exception as e:
                print(f"!!! Errore nel batch {current_batch_num}: {e}")
            
            # Controlla se ci sono ancora documenti dopo questo batch
            if i + _BATCH_SIZE < total_chunks:
                print("    Attesa di 60 secondi per reset Rate Limit API...")
                time.sleep(60)
            else:
                print("    Ultimo batch completato.")

        # --- 5. Riepilogo Finale ---
        print(f"Collezione '{collection_name}' creata con successo!")
        print(f"  - Chunk totali elaborati: {len(all_document_ids)} su {total_chunks}")
        print(f"  - Dimensione vettori: {vector_size}")

    def create_collection_from_pdfs(self, collection_name: str, folder_path: str = "./docs"):
        """
        Crea una collezione a partire da tutti i PDF in una cartella.
        Richiama i metodi definiti sopra.

        Args:
            collection_name: nome della nuova collezione
            folder_path: percorso della cartella contenente i PDF
        """
        print(f"\n=== Creazione Collezione: {collection_name} ===")

        # 1. Carica PDF
        documents = self.load_pdfs_from_folder(folder_path)

        # 2. Divide in chunk
        splits = self.split_documents(documents)

        # 3. Crea collezione
        self.create_collection(collection_name, splits)


def main():
    print("=== Creatore Collezioni Qdrant ===\n")

    # Scelta del modello di embedding
    print("Scegli il modello di embedding:")
    print("1. [gemini] Gemini (gemini-embedding-001) - richiede API key Google")
    print("2. [hf]     HuggingFace (all-MiniLM-L6-v2) - locale")

    embedding_model = input("\n").strip()

    # Inizializza il creatore
    creator = CollectionCreator(embedding_model=embedding_model)

    # Nome della collezione
    collection_name = input("\nInserisci il nome per la nuova collezione: ").strip()

    # Percorso della cartella
    folder_path = input("Percorso della cartella contenente i PDF [default: ./docs]: ").strip()
    if not folder_path:
        folder_path = "./docs"

    # Verifica che la cartella esista
    if not os.path.exists(folder_path):
        print(f"Errore: La cartella '{folder_path}' non esiste.")
        print("Assicurati che la cartella esista e contenga file PDF.")
        return

    try:
        # Crea la collezione
        creator.create_collection_from_pdfs(collection_name, folder_path)
        print("\nOperazione completata con successo!")
    except Exception as e:
        print(f"Errore durante la creazione della collezione: {e}")

if __name__ == "__main__":
    main()
