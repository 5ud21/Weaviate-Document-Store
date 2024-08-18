"""
ingest.py

This script is responsible for ingesting documents into a Weaviate document store. It performs the following steps:
1. Converts PDF documents into text.
2. Preprocesses the text documents.
3. Writes the preprocessed documents into the Weaviate document store.
4. Updates the document embeddings using a specified embedding model.

Modules:
    - haystack.nodes: Various nodes for processing and retrieving documents.
    - haystack.document_stores: Document stores for storing and retrieving documents.
    - haystack: Core module for pipelines.
    - haystack.preview.components.file_converters.pypdf: PDF to document converter.
    - haystack.preview.dataclasses: Data classes for documents.

Functions:
    - main: Main function to execute the document ingestion process.
"""

from haystack.nodes import EmbeddingRetriever, MarkdownConverter, PreProcessor, AnswerParser, PromptModel, PromptNode, PromptTemplate
from haystack.document_stores import WeaviateDocumentStore
from haystack.preview.components.file_converters.pypdf import PyPDFToDocument
from haystack import Pipeline

print("Import Successfully")

# Path to the PDF document
path_doc =["data/atc manual - split.pdf"]

# Initialize the Weaviate document store
document_store = WeaviateDocumentStore(host='http://localhost',
                                       port=8080,
                                       embedding_dim=768)

print("Document Store: ", document_store)
print("#####################")

# Convert PDF to document
converter = PyPDFToDocument()
print("Converter: ", converter)
print("#####################")
output = converter.run(paths=path_doc)
docs = output["documents"]
print("Docs: ", docs)
print("#####################")

# Prepare documents for ingestion
final_doc = []
for doc in docs:
    print(doc.text)
    new_doc = {
        'content': doc.text,
        'meta': doc.metadata
    }
    final_doc.append(new_doc)
    print("#####################")

# Preprocess documents
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=False,
    clean_header_footer=True,
    split_by="word",
    split_length=500,
    split_respect_sentence_boundary=True,
)
print("Preprocessor: ", preprocessor)
print("#####################")

preprocessed_docs = preprocessor.process(final_doc)
print("Preprocessed Docs: ", preprocessed_docs)
print("#####################")

#Write documents to the document store
document_store.write_documents(preprocessed_docs)

#Initialize the retriever
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

print("Retriever: ", retriever)

#Update document embeddings
document_store.update_embeddings(retriever)

print("Embeddings Done.")





