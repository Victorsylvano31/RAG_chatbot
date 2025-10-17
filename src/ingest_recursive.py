import os
import re
import json
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Any, Dict, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    SUPPORTED_FORMATS = {'.pdf', '.json', '.txt', '.md', '.csv'}
    MAX_WORKERS = 4  # Pour le traitement parallèle
    MAX_FILE_SIZE_MB = 50  # Taille max des fichiers

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
# FILE DISCOVERY - PARCOURS RÉCURSIF
# ==========================
class FileDiscovery:
    """Découverte récursive des fichiers dans l'arborescence"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def discover_files(self, data_path: Path) -> Dict[str, List[Path]]:
        """Découvre récursivement tous les fichiers supportés"""
        files_by_type = {
            'pdf': [],
            'json': [], 
            'text': [],
            'other': []
        }
        
        if not data_path.exists():
            self.logger.error(f"❌ Dossier introuvable: {data_path}")
            return files_by_type
        
        total_files = 0
        total_size = 0
        
        # Parcours récursif de l'arborescence
        for file_path in data_path.rglob('*'):  # rglob pour la récursivité
            if file_path.is_file():
                total_files += 1
                total_size += file_path.stat().st_size
                
                # Classification par type
                if file_path.suffix.lower() == '.pdf':
                    files_by_type['pdf'].append(file_path)
                elif file_path.suffix.lower() == '.json':
                    files_by_type['json'].append(file_path)
                elif file_path.suffix.lower() in ['.txt', '.md', '.csv']:
                    files_by_type['text'].append(file_path)
                else:
                    files_by_type['other'].append(file_path)
        
        # Log des statistiques
        self.logger.info(f"🔍 Découverte fichiers:")
        self.logger.info(f"   📁 Dossiers parcourus: {len(list(data_path.rglob('')))}")
        self.logger.info(f"   📄 Total fichiers: {total_files}")
        self.logger.info(f"   💾 Taille totale: {total_size / (1024*1024):.2f} MB")
        self.logger.info(f"   📊 PDF: {len(files_by_type['pdf'])}")
        self.logger.info(f"   📊 JSON: {len(files_by_type['json'])}")
        self.logger.info(f"   📊 Texte: {len(files_by_type['text'])}")
        self.logger.info(f"   📊 Autres: {len(files_by_type['other'])}")
        
        return files_by_type

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
# MULTI-FORMAT DOCUMENT LOADER AVEC RÉCURSIVITÉ
# ==========================
class RecursiveMultiFormatLoader:
    """Chargeur de documents avec support récursif des dossiers"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.processor = TextProcessor()
        self.file_discovery = FileDiscovery(logger)
    
    def load_all_documents(self, data_path: Path) -> List[Document]:
        """Charge récursivement tous les documents"""
        documents = []
        
        if not data_path.exists():
            self.logger.error(f"❌ Dossier introuvable: {data_path}")
            return documents
        
        # 1. Découverte des fichiers
        files_by_type = self.file_discovery.discover_files(data_path)
        
        # 2. Chargement parallèle par type de fichier
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            # PDFs
            pdf_futures = [executor.submit(self._load_pdf_file, pdf_file) 
                          for pdf_file in files_by_type['pdf']]
            
            # JSONs  
            json_futures = [executor.submit(self._load_json_file, json_file)
                           for json_file in files_by_type['json']]
            
            # Textes
            text_futures = [executor.submit(self._load_text_file, text_file)
                           for text_file in files_by_type['text']]
            
            # Collecte des résultats
            for future in as_completed(pdf_futures + json_futures + text_futures):
                try:
                    docs = future.result()
                    if docs:
                        documents.extend(docs)
                except Exception as e:
                    self.logger.error(f"❌ Erreur chargement fichier: {e}")
        
        self.logger.info(f"📚 Total documents chargés: {len(documents)}")
        return documents
    
    def _load_pdf_file(self, pdf_file: Path) -> List[Document]:
        """Charge un fichier PDF individuel"""
        try:
            # Vérifier la taille du fichier
            file_size_mb = pdf_file.stat().st_size / (1024 * 1024)
            if file_size_mb > Config.MAX_FILE_SIZE_MB:
                self.logger.warning(f"⚠️ PDF trop volumineux ignoré: {pdf_file.name} ({file_size_mb:.1f}MB)")
                return []
            
            # Essayer PyMuPDF d'abord
            try:
                loader = PyMuPDFLoader(str(pdf_file))
            except ImportError:
                loader = PyPDFLoader(str(pdf_file))
            
            docs = loader.load()
            
            # Nettoyage et enrichissement
            for doc in docs:
                doc.page_content = self.processor.deep_clean_text(doc.page_content)
                if doc.page_content and len(doc.page_content) >= Config.MIN_DOC_LENGTH:
                    # Ajouter le chemin relatif pour la traçabilité
                    relative_path = pdf_file.relative_to(pdf_file.parents[1])
                    doc.metadata.update({
                        "source_type": "pdf",
                        "file_path": str(relative_path),
                        "file_name": pdf_file.name,
                        "file_size_mb": file_size_mb
                    })
            
            self.logger.info(f"   ✅ PDF: {pdf_file.name} → {len(docs)} pages")
            return [doc for doc in docs if doc.page_content]
            
        except Exception as e:
            self.logger.error(f"   ❌ Erreur PDF {pdf_file.name}: {e}")
            return []
    
    def _load_json_file(self, json_file: Path) -> List[Document]:
        """Charge un fichier JSON individuel"""
        try:
            # Vérifier la taille du fichier
            file_size_mb = json_file.stat().st_size / (1024 * 1024)
            if file_size_mb > Config.MAX_FILE_SIZE_MB:
                self.logger.warning(f"⚠️ JSON trop volumineux ignoré: {json_file.name} ({file_size_mb:.1f}MB)")
                return []
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            relative_path = json_file.relative_to(json_file.parents[1])
            
            # Traitement selon la structure JSON
            if isinstance(data, list):
                documents.extend(self._process_json_list(data, json_file, relative_path))
            elif isinstance(data, dict):
                documents.extend(self._process_json_dict(data, json_file, relative_path))
            else:
                # JSON avec valeur simple
                content = self.processor.deep_clean_text(str(data))
                if content:
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": json_file.name,
                            "source_type": "json",
                            "file_path": str(relative_path),
                            "content_type": "simple_value"
                        }
                    ))
            
            self.logger.info(f"   ✅ JSON: {json_file.name} → {len(documents)} éléments")
            return documents
            
        except json.JSONDecodeError as e:
            self.logger.error(f"   ❌ JSON invalide {json_file.name}: {e}")
            return []
        except Exception as e:
            self.logger.error(f"   ❌ Erreur JSON {json_file.name}: {e}")
            return []
    
    def _process_json_list(self, data: List, json_file: Path, relative_path: str) -> List[Document]:
        """Traite une liste JSON"""
        documents = []
        
        for i, item in enumerate(data):
            if isinstance(item, str):
                content = self.processor.deep_clean_text(item)
                if content:
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": json_file.name,
                            "source_type": "json",
                            "file_path": str(relative_path),
                            "item_index": i,
                            "content_type": "text_list"
                        }
                    ))
            elif isinstance(item, dict):
                content = self._extract_text_from_dict(item)
                if content:
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": json_file.name,
                            "source_type": "json",
                            "file_path": str(relative_path),
                            "item_index": i,
                            "content_type": "dict_list"
                        }
                    ))
        
        return documents
    
    def _process_json_dict(self, data: Dict, json_file: Path, relative_path: str) -> List[Document]:
        """Traite un dictionnaire JSON"""
        documents = []
        content = self._extract_text_from_dict(data)
        
        if content:
            documents.append(Document(
                page_content=content,
                metadata={
                    "source": json_file.name,
                    "source_type": "json", 
                    "file_path": str(relative_path),
                    "content_type": "single_dict"
                }
            ))
        
        return documents
    
    def _extract_text_from_dict(self, data: Dict) -> str:
        """Extrait récursivement le texte d'un dictionnaire JSON"""
        texts = []
        
        def extract_recursive(obj, current_path=""):
            if isinstance(obj, str):
                cleaned = self.processor.deep_clean_text(obj)
                if cleaned and len(cleaned) > 10:
                    texts.append(cleaned)
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    extract_recursive(value, f"{current_path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_recursive(item, f"{current_path}[{i}]")
        
        extract_recursive(data)
        return "\n".join(texts)
    
    def _load_text_file(self, text_file: Path) -> List[Document]:
        """Charge un fichier texte individuel"""
        try:
            # Vérifier la taille
            file_size_mb = text_file.stat().st_size / (1024 * 1024)
            if file_size_mb > Config.MAX_FILE_SIZE_MB:
                self.logger.warning(f"⚠️ Texte trop volumineux ignoré: {text_file.name} ({file_size_mb:.1f}MB)")
                return []
            
            documents = []
            relative_path = text_file.relative_to(text_file.parents[1])
            
            # Essayer différents encodages
            for encoding in ['utf-8', 'latin-1', 'windows-1252']:
                try:
                    loader = TextLoader(str(text_file), encoding=encoding)
                    docs = loader.load()
                    
                    for doc in docs:
                        doc.page_content = self.processor.deep_clean_text(doc.page_content)
                        if doc.page_content and len(doc.page_content) >= Config.MIN_DOC_LENGTH:
                            doc.metadata.update({
                                "source_type": "text",
                                "file_path": str(relative_path),
                                "file_name": text_file.name,
                                "encoding": encoding
                            })
                            documents.append(doc)
                    
                    self.logger.info(f"   ✅ Texte: {text_file.name} ({encoding}) → {len(docs)} documents")
                    break
                    
                except UnicodeDecodeError:
                    continue
            
            return documents
            
        except Exception as e:
            self.logger.error(f"   ❌ Erreur texte {text_file.name}: {e}")
            return []

# ==========================
# DOCUMENT PROCESSOR
# ==========================
class DocumentProcessor:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
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
        
        # Statistiques détaillées
        self._log_detailed_statistics(valid_chunks)
        
        return valid_chunks
    
    def _log_detailed_statistics(self, chunks: List[Document]):
        """Log des statistiques détaillées par type et dossier"""
        source_stats = {}
        folder_stats = {}
        
        for chunk in valid_chunks:
            source_type = chunk.metadata.get("source_type", "unknown")
            file_path = chunk.metadata.get("file_path", "unknown")
            folder = str(Path(file_path).parent) if file_path != "unknown" else "root"
            
            # Stats par type
            source_stats[source_type] = source_stats.get(source_type, 0) + 1
            
            # Stats par dossier
            folder_stats[folder] = folder_stats.get(folder, 0) + 1
        
        self.logger.info(f"✂️  Chunks créés: {len(valid_chunks)}")
        
        # Stats par type
        self.logger.info("   📊 Par type:")
        for source_type, count in source_stats.items():
            self.logger.info(f"      {source_type}: {count}")
        
        # Top dossiers
        self.logger.info("   📁 Top dossiers:")
        top_folders = sorted(folder_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        for folder, count in top_folders:
            self.logger.info(f"      {folder}: {count}")

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
                batch_num = i // batch_size + 1
                total_batches = (len(chunks) - 1) // batch_size + 1
                
                self.logger.info(f"🔄 Traitement batch {batch_num}/{total_batches} ({len(batch)} documents)")
                
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
# PIPELINE PRINCIPAL RÉCURSIF
# ==========================
class RecursiveIngestionPipeline:
    """Pipeline d'ingestion avec support récursif des dossiers"""
    
    def __init__(self, data_path: str = "../data", chroma_path: str = "../chroma_db"):
        self.data_path = Path(data_path)
        self.chroma_path = Path(chroma_path)
        self.logger = setup_logging()
        
        self.loader = RecursiveMultiFormatLoader(self.logger)
        self.processor = DocumentProcessor(self.logger)
        self.vector_builder = VectorStoreBuilder(chroma_path, self.logger)
    
    def run_ingestion(self) -> bool:
        """Exécute le pipeline complet"""
        self.logger.info("🚀 DÉMARRAGE INGESTION RÉCURSIVE")
        self.logger.info(f"📁 Dossier racine: {self.data_path}")
        self.logger.info(f"🎯 Base vectorielle: {self.chroma_path}")
        
        try:
            # 1. Chargement récursif multi-format
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
                self.logger.info("🎉 INGESTION RÉCURSIVE RÉUSSIE!")
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
    
    pipeline = RecursiveIngestionPipeline("../data", "../chroma_db")
    success = pipeline.run_ingestion()
    
    if success:
        pipeline.logger.info("✨ SYSTÈME PRÊT POUR L'ARBORESCENCE COMPLÈTE!")
    else:
        pipeline.logger.error("💥 VERIFIEZ VOTRE ARBORESCENCE DE DOSSIERS")
        exit(1)