import os
import re
import json
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Any, Dict, Union

# Imports LangChain
from langchain_community.document_loaders import (
    PyPDFLoader, PyMuPDFLoader, 
    JSONLoader, TextLoader,
    UnstructuredFileLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# ==========================
# CONFIGURATION
# ==========================
class Config:
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    MIN_DOC_LENGTH = 10
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SUPPORTED_FORMATS = {'.pdf', '.json', '.txt', '.md'}

# ==========================
# SETUP LOGGING
# ==========================
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

# ==========================
# TEXT PROCESSOR
# ==========================
class TextProcessor:
    """Processeur de texte pour tous les formats"""
    
    @staticmethod
    def deep_clean_text(text: str) -> str:
        """Nettoyage approfondi du texte"""
        if not text or not isinstance(text, str):
            return ""
        
        # Suppression des caractères problématiques
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        text = re.sub(r'[\u200b-\u200f\u2028-\u202e\ufeff]', '', text)
        
        # Gestion des césures de mots en fin de ligne
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Normalisation des espaces
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()

# ==========================
# MULTI-FORMAT DOCUMENT LOADER
# ==========================
class MultiFormatLoader:
    """Chargeur de documents multi-format (PDF, JSON, TXT)"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.processor = TextProcessor()
    
    def load_all_documents(self, data_path: Path) -> List[Document]:
        """Charge tous les documents supportés"""
        documents = []
        
        if not data_path.exists():
            self.logger.error(f"❌ Dossier introuvable: {data_path}")
            return documents
        
        # Chargement par type de fichier
        documents.extend(self._load_pdf_documents(data_path))
        documents.extend(self._load_json_documents(data_path))
        documents.extend(self._load_text_documents(data_path))
        
        self.logger.info(f"📚 Total documents chargés: {len(documents)}")
        return documents
    
    def _load_pdf_documents(self, data_path: Path) -> List[Document]:
        """Charge les documents PDF"""
        pdf_files = list(data_path.glob("*.pdf"))
        documents = []
        
        for pdf_file in pdf_files:
            try:
                self.logger.info(f"📖 Chargement PDF: {pdf_file.name}")
                
                # Essayer PyMuPDF d'abord (meilleur)
                try:
                    loader = PyMuPDFLoader(str(pdf_file))
                except ImportError:
                    loader = PyPDFLoader(str(pdf_file))
                
                docs = loader.load()
                
                # Nettoyage
                for doc in docs:
                    doc.page_content = self.processor.deep_clean_text(doc.page_content)
                    if doc.page_content and len(doc.page_content) >= Config.MIN_DOC_LENGTH:
                        doc.metadata["source_type"] = "pdf"
                        documents.append(doc)
                
                self.logger.info(f"   ✅ {len(docs)} pages")
                
            except Exception as e:
                self.logger.error(f"   ❌ Erreur PDF {pdf_file.name}: {e}")
        
        return documents
    
    def _load_json_documents(self, data_path: Path) -> List[Document]:
        """Charge les documents JSON avec différentes stratégies"""
        json_files = list(data_path.glob("*.json"))
        documents = []
        
        for json_file in json_files:
            try:
                self.logger.info(f"📊 Chargement JSON: {json_file.name}")
                json_docs = self._load_single_json_file(json_file)
                documents.extend(json_docs)
                
            except Exception as e:
                self.logger.error(f"   ❌ Erreur JSON {json_file.name}: {e}")
        
        return documents
    
    def _load_single_json_file(self, json_file: Path) -> List[Document]:
        """Charge un fichier JSON avec différentes stratégies"""
        documents = []
        
        try:
            # Essayer de charger comme JSON brut d'abord
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Stratégie 1: JSON avec une liste de textes
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, str):
                        content = self.processor.deep_clean_text(item)
                        if content:
                            documents.append(Document(
                                page_content=content,
                                metadata={
                                    "source": json_file.name,
                                    "source_type": "json",
                                    "item_index": i,
                                    "content_type": "text_list"
                                }
                            ))
                    elif isinstance(item, dict):
                        # Extraire tout le texte du dictionnaire
                        content = self._extract_text_from_dict(item)
                        if content:
                            documents.append(Document(
                                page_content=content,
                                metadata={
                                    "source": json_file.name,
                                    "source_type": "json", 
                                    "item_index": i,
                                    "content_type": "dict_list"
                                }
                            ))
            
            # Stratégie 2: JSON avec un objet unique
            elif isinstance(data, dict):
                content = self._extract_text_from_dict(data)
                if content:
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": json_file.name,
                            "source_type": "json",
                            "content_type": "single_dict"
                        }
                    ))
            
            self.logger.info(f"   ✅ {len(documents)} éléments JSON")
            
        except json.JSONDecodeError:
            self.logger.warning(f"   ⚠️ {json_file.name}: JSON invalide")
        except Exception as e:
            self.logger.error(f"   ❌ Erreur traitement JSON: {e}")
        
        return documents
    
    def _extract_text_from_dict(self, data: Dict) -> str:
        """Extrait récursivement le texte d'un dictionnaire JSON"""
        texts = []
        
        def extract_recursive(obj, current_path=""):
            if isinstance(obj, str):
                cleaned = self.processor.deep_clean_text(obj)
                if cleaned and len(cleaned) > 10:  # Ignorer les très courts
                    texts.append(cleaned)
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    extract_recursive(value, f"{current_path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_recursive(item, f"{current_path}[{i}]")
        
        extract_recursive(data)
        return "\n".join(texts)
    
    def _load_text_documents(self, data_path: Path) -> List[Document]:
        """Charge les documents texte (TXT, MD)"""
        text_files = []
        for ext in ['.txt', '.md']:
            text_files.extend(data_path.glob(f"*{ext}"))
        
        documents = []
        
        for text_file in text_files:
            try:
                self.logger.info(f"📝 Chargement texte: {text_file.name}")
                
                # Essayer différents encodages
                for encoding in ['utf-8', 'latin-1', 'windows-1252']:
                    try:
                        loader = TextLoader(str(text_file), encoding=encoding)
                        docs = loader.load()
                        
                        for doc in docs:
                            doc.page_content = self.processor.deep_clean_text(doc.page_content)
                            if doc.page_content and len(doc.page_content) >= Config.MIN_DOC_LENGTH:
                                doc.metadata["source_type"] = "text"
                                documents.append(doc)
                        
                        self.logger.info(f"   ✅ {text_file.name} ({encoding})")
                        break
                        
                    except UnicodeDecodeError:
                        continue
                
            except Exception as e:
                self.logger.error(f"   ❌ Erreur texte {text_file.name}: {e}")
        
        return documents

# ==========================
# DOCUMENT PROCESSOR
# ==========================
class DocumentProcessor:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.processor = TextProcessor()
    
    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """Crée des chunks validés"""
        if not documents:
            return []
        
        # Validation des documents
        valid_docs = []
        for doc in documents:
            if (hasattr(doc, 'page_content') and 
                doc.page_content and 
                isinstance(doc.page_content, str) and
                len(doc.page_content.strip()) >= Config.MIN_DOC_LENGTH):
                valid_docs.append(doc)
        
        self.logger.info(f"📄 Documents valides: {len(valid_docs)}")
        
        if not valid_docs:
            return []
        
        # Création des chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        chunks = splitter.split_documents(valid_docs)
        
        # Validation finale
        valid_chunks = []
        for chunk in chunks:
            if (hasattr(chunk, 'page_content') and 
                chunk.page_content and 
                isinstance(chunk.page_content, str) and
                len(chunk.page_content.strip()) >= 10):
                valid_chunks.append(chunk)
        
        # Statistiques par type de source
        source_stats = {}
        for chunk in valid_chunks:
            source_type = chunk.metadata.get("source_type", "unknown")
            source_stats[source_type] = source_stats.get(source_type, 0) + 1
        
        self.logger.info(f"✂️  Chunks créés: {len(valid_chunks)}")
        for source_type, count in source_stats.items():
            self.logger.info(f"   📊 {source_type}: {count} chunks")
        
        return valid_chunks

# ==========================
# VECTOR STORE BUILDER
# ==========================
class VectorStoreBuilder:
    def __init__(self, chroma_path: str, logger: logging.Logger):
        self.chroma_path = Path(chroma_path)
        self.logger = logger
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )
    
    def safe_build_vector_store(self, chunks: List[Document]) -> Optional[Chroma]:
        """Construction sécurisée de la base vectorielle"""
        if not chunks:
            self.logger.error("❌ Aucun chunk à vectoriser")
            return None
        
        # Nettoyage
        if self.chroma_path.exists():
            shutil.rmtree(self.chroma_path)
            self.logger.info("🧹 Base précédente nettoyée")
        
        try:
            self.logger.info("🔨 Construction base vectorielle...")
            
            # Construction par lots pour plus de stabilité
            batch_size = 100
            vector_store = None
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                self.logger.info(f"🔄 Traitement batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
                
                if vector_store is None:
                    # Premier batch
                    vector_store = Chroma.from_documents(
                        documents=batch,
                        embedding=self.embeddings,
                        persist_directory=str(self.chroma_path)
                    )
                else:
                    # Batchs suivants
                    vector_store.add_documents(batch)
            
            if vector_store:
                vector_store.persist()
                self.logger.info(f"✅ Base sauvegardée: {self.chroma_path}")
                self.logger.info(f"📊 Total documents indexés: {len(chunks)}")
            
            return vector_store
            
        except Exception as e:
            self.logger.error(f"❌ Erreur construction base: {e}")
            return None

# ==========================
# PIPELINE PRINCIPAL MULTI-FORMAT
# ==========================
class MultiFormatIngestionPipeline:
    """Pipeline d'ingestion multi-format (PDF, JSON, TXT)"""
    
    def __init__(self, data_path: str = "../data", chroma_path: str = "../chroma_db"):
        self.data_path = Path(data_path)
        self.chroma_path = Path(chroma_path)
        self.logger = setup_logging()
        
        self.loader = MultiFormatLoader(self.logger)
        self.processor = DocumentProcessor(self.logger)
        self.vector_builder = VectorStoreBuilder(chroma_path, self.logger)
    
    def run_ingestion(self) -> bool:
        """Exécute le pipeline complet"""
        self.logger.info("🚀 DÉMARRAGE INGESTION MULTI-FORMAT")
        self.logger.info(f"📁 Dossier source: {self.data_path}")
        
        # Vérifier les fichiers disponibles
        all_files = []
        for ext in Config.SUPPORTED_FORMATS:
            all_files.extend(self.data_path.glob(f"*{ext}"))
        
        if not all_files:
            self.logger.error("❌ Aucun fichier supporté trouvé")
            self.logger.info(f"📋 Formats supportés: {Config.SUPPORTED_FORMATS}")
            return False
        
        self.logger.info(f"📋 Fichiers trouvés: {len(all_files)}")
        for file in all_files:
            self.logger.info(f"   📄 {file.name}")
        
        try:
            # 1. Chargement multi-format
            documents = self.loader.load_all_documents(self.data_path)
            if not documents:
                self.logger.error("❌ Échec chargement documents")
                return False
            
            # 2. Découpage en chunks
            chunks = self.processor.create_chunks(documents)
            if not chunks:
                self.logger.error("❌ Échec création chunks")
                return False
            
            # 3. Construction base vectorielle
            vector_store = self.vector_builder.safe_build_vector_store(chunks)
            
            if vector_store:
                self.logger.info("🎉 INGESTION MULTI-FORMAT RÉUSSIE!")
                return True
            else:
                self.logger.error("💥 ÉCHEC CONSTRUCTION BASE")
                return False
                
        except Exception as e:
            self.logger.error(f"💥 ERREUR CRITIQUE: {e}")
            return False

# ==========================
# EXÉCUTION
# ==========================
if __name__ == "__main__":
    # Installation des dépendances si nécessaire
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("📦 Installation de PyMuPDF...")
        os.system("pip install pymupdf")
    
    pipeline = MultiFormatIngestionPipeline("../data", "../chroma_db")
    success = pipeline.run_ingestion()
    
    if success:
        pipeline.logger.info("✨ PRÊT POUR LES QUESTIONS MULTI-FORMATS!")
    else:
        pipeline.logger.error("💥 VERIFIEZ VOS FICHIERS")
        exit(1)