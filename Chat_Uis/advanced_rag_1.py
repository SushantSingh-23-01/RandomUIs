import os
import re
import time
import ollama
import chromadb
from chromadb.config import Settings
import gradio as gr
from PyPDF2 import PdfReader
import uuid
from dataclasses import dataclass
from typing import List, Optional
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# GLOBAL VARIABLES
client = chromadb.Client(settings = Settings(anonymized_telemetry=False))
chroma_collection = client.get_or_create_collection('docs')
parent_doc_store = {}

@dataclass
class SharedState:
    chats_model = r'hf.co/mlabonne/gemma-3-4b-it-abliterated-GGUF:Q4_K_M'
    emb_model:str = r'granite-embedding:latest'
    system_prompt:str = r'Act as a helpful assistant'
    temperature:float = 0.9
    top_k:int = 50
    top_p:float = 0.9

class RecursiveTextSplitter:
    """
    Implements a recursive text splitter from scratch.
    
    The splitter attempts to break text down by a list of separators,
    recursing with the next separator if a chunk is still too large.
    """
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: Optional[List[str]]= None
    ):
        """
        Initializes the text splitter.

        Args:
            chunk_size (int): The maximum size of each chunk (in characters).
            chunk_overlap (int): The number of characters to overlap between chunks.
            separators (List[str]): A list of strings to split the text by,
                                    in order of priority.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Sensible default separators for general text
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """
        The main public method to split text.
        
        Args:
            text (str): The text to be split.
            
        Returns:
            List[str]: A list of text chunks.
        """
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """The core recursive splitting logic."""
        final_chunks = []
        
        # 1. Determine the best separator to use
        separator = separators[0]
        # Find the first separator that actually exists in the text
        for s in separators:
            if s in text:
                separator = s
                break
        
        # 2. Handle the base case: if the text is small enough, return it as a chunk
        if len(text) <= self.chunk_size:
            return [text]
            
        # 3. Split the text by the best separator
        splits = text.split(separator)
        
        # 4. Recursively process and merge the splits
        current_chunk = ""
        for i, split in enumerate(splits):
            # If the split itself is too large, recurse on it with the next separators
            if len(split) > self.chunk_size:
                # If there are more separators, recurse
                if len(separators) > 1:
                    # Extend final_chunks with the results of the recursive call
                    final_chunks.extend(self._recursive_split(split, separators[1:]))
                else: # No more separators, so we have to add the oversized chunk
                    final_chunks.append(split)
                continue

            # Check if adding the next split exceeds the chunk size
            # We add the separator back in, except for the last split
            separator_to_add = separator if i < len(splits) - 1 else ""
            if len(current_chunk) + len(split) + len(separator_to_add) > self.chunk_size:
                final_chunks.append(current_chunk)
                # Start the new chunk with an overlap from the end of the last one
                current_chunk = current_chunk[-self.chunk_overlap:] + split + separator_to_add
            else:
                current_chunk += split + separator_to_add
        
        # Add the last remaining chunk
        if current_chunk:
            final_chunks.append(current_chunk)
            
        return final_chunks

def clean_pdf_text(text):
    # 1. Handle hyphenated words split across lines (less likely page-by-page, but good to keep)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # 2. Remove multiple newlines and replace with single space
    text = re.sub(r'\n+', ' ', text)      
    # 3. Remove page numbers, headers/footers (customize heavily for your PDF
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)     
    # 4. Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)  
    # 5. Remove Email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    return text

def read_pdfs(files):
    total_text = ''
    for f in files:
        reader = PdfReader(f)
        for pages in reader.pages:
            text = pages.extract_text()
            text = clean_pdf_text(text)
            total_text += text
    return total_text

class ParentChildRetriever:
    def ingest_parent_document(self, text, chunk_size, chunk_overlap):
        splitter = RecursiveTextSplitter(chunk_size, chunk_overlap)
        parent_chunks = splitter.split_text(text)
        
        parent_doc_store = {}
        for parent_chunk in parent_chunks:
            parent_id = str(uuid.uuid4())
            parent_doc_store[parent_id] = parent_chunk
        return parent_doc_store
        
    def ingest_child_document(self, parent_doc_store, chunk_size, chunk_overlap, shared_state):
        global chroma_collection
        total_chunks_count = 0
        for parent_id, parent_chunk in parent_doc_store.items():
            splitter = RecursiveTextSplitter(chunk_size, chunk_overlap)
            child_chunks =splitter.split_text(parent_chunk)
            for i, child_chunk in enumerate(child_chunks):
                chunk_id = f'{parent_id}-{i}'
                embeddings = ollama.embed(shared_state.emb_model, child_chunk)['embeddings']
                chroma_collection.add(
                    ids = chunk_id,
                    embeddings = embeddings,
                    metadatas={"parent_id": parent_id, "chunk_index": i},
                    documents=child_chunk
                )
                total_chunks_count += 1
        return total_chunks_count

    def generate_pdf_embeddings(self, files,  shared_state, chunk_size, chunk_overlap):
        global parent_doc_store
        start = time.time()
        text = read_pdfs(files)
        parent_doc_store = self.ingest_parent_document(text, chunk_size*2, chunk_overlap*2)
        total_chunk_counts = self.ingest_child_document(parent_doc_store, chunk_size, chunk_overlap, shared_state)
        print(f'Generated Document Stores: {time.time()- start:.2f}s')
        return gr.Textbox(f'Created PDF Embeddings.\nTotal Chunk Count: {total_chunk_counts}')

    @staticmethod
    def retrieve_parent_documents(query_text, shared_state, n_results=3):
        global parent_doc_store
        query_embeddings = ollama.embed(shared_state.emb_model, query_text)['embeddings']
        results = chroma_collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                include=['metadatas']
            )
        retrieved_parent_ids = set()
        if results['metadatas']:
            for metadata_list in results['metadatas']:
                for metadata in metadata_list:
                    if 'parent_id' in metadata:
                        retrieved_parent_ids.add(metadata['parent_id'])
        retrieved_documents = []
        for parent_id in retrieved_parent_ids:
            if parent_id in parent_doc_store:
                retrieved_documents.append(parent_doc_store[parent_id])
        return retrieved_documents
    
    @staticmethod
    def clear_rag_embeddings():
        global chroma_collection
        all_ids = chroma_collection.get(limit=chroma_collection.count())['ids']
        if all_ids:
            chroma_collection.delete(ids=all_ids)
            return gr.Textbox(f"RAG embeddings cleared! Total documents: {chroma_collection.count()}")
        else:
            return gr.Textbox(f'Chroma collection is already empty.')

def rephrase_query(original_query: str, shared_state:SharedState) -> list[str]:        
    # This is the prompt that instructs the model on how to behave.
    # It's engineered to produce a numbered list of alternative questions.
    prompt = f"""
    As a helpful AI assistant, your task is to rephrase the following user query.
    Generate three diverse, rephrased versions of the query. The rephrased queries should be different from each other and from the original query, covering different angles or phrasings that might better match documents in a database.
    The goal is to provide a variety of questions that could retrieve more relevant context.
    Original Query: {original_query}
    Please provide the three rephrased queries as a numbered list.
    """
    start = time.time()
    response = ollama.chat(
        model=shared_state.chats_model,
        messages=[{'role': 'user', 'content': prompt}],
    )['message']['content']


    rephrased_text = clean_llm_output_text(response)
    rephrased_queries = re.findall( r'\d+\.\s+["“]([^"”]+)["”]', rephrased_text)
    

    if not rephrased_queries:
        print("Warning: Could not parse rephrased queries from the model's response.")
        return [original_query]

    # # The final list of queries includes the original one plus the generated ones.
    all_queries = [original_query] + rephrased_queries
    print(f'Rephrased Queries: {time.time() - start:.2f}s')
    return all_queries

def rerank_documents(query: str, documents: list, shared_state:SharedState) -> list:
    start = time.time()
    relevant_docs = []
    for doc in documents:
        prompt = (
            f"User query: '{query}'\n\n"
            f"Document: '{doc}'\n\n"
            "Is the document relevant to the user query? (Yes/No)"
        )
        response = ollama.chat(
            model=shared_state.chats_model,
            messages=[{'role': 'user', 'content': prompt}]
        )['message']['content'].strip().lower()

        if "yes" in response:
            relevant_docs.append(doc)
        else:
            pass
    print(f'Reranking Documents: {time.time() - start:.2f}s')
    return relevant_docs


def light_keyword_filter(query:str, documents:list)->list:
    sw = set(stopwords.words('english'))
    query_words = set(word for word in  word_tokenize(query.lower()) if word not in sw)
    scored_docs = []
    for doc in documents:
        doc_words = set(word for word in  word_tokenize(doc.lower()) if word not in sw)
        overlap = len(query_words.intersection(doc_words))
        scored_docs.append({'document': doc, 'overlap': overlap})
    scored_docs.sort(key=lambda x: x['overlap'], reverse=True)
    return [item['document'] for item in scored_docs if item['overlap'] > 0]
    
          
def ollama_response(message, history, shared_state):
    global chroma_collection
    start = time.time()
    if chroma_collection.count() > 0:
        multiple_queries = rephrase_query(message, shared_state)
        results = set()
        for query in multiple_queries:
            result_doc = ParentChildRetriever.retrieve_parent_documents(query, shared_state, n_results=5)
            result_doc = light_keyword_filter(message, result_doc)
            results.add(' '.join(i for i in result_doc))
            
        text = ' '.join(i for i in results)
        system_prompt = f'''
        Based only on the following context, answer the question.
        Context
        ---
        {text}
        '''
    else:
        system_prompt = shared_state.system_prompt
        
    ollama_input_message = [{'role':'system', 'content': system_prompt}] + history + [{'role':'user', 'content':message}]
    history.append({'role':'user', 'content':message})
    
    full_response = ''  
    for chunk in ollama.chat(shared_state.chats_model, ollama_input_message, stream=True):
        full_response += chunk['message']['content']
        yield (
                '',
                history + [{"role": "assistant", "content": full_response}],
                history + [{"role": "assistant", "content": full_response}],
            )
    history.append({"role": "assistant", "content": full_response})
    print(f'Chat LLM response: {time.time() - start:.2f}s')
    yield '', history, history

def regen_response(history, shared_state):
    if len(history) >= 2 and history[-1]['role'] == 'assistant':
        history.pop()
        message = history.pop()['content']
        yield from ollama_response(message, history, shared_state)
    yield '', history, history

def clean_llm_output_text(text):
    # 1. Normalize various whitespace characters to a single space
    cleaned_text = re.sub(r'\s+', ' ', text)
    # 2. Convert smart quotes to straight quotes
    cleaned_text = re.sub(r'[“”]', '"', cleaned_text)
    cleaned_text = re.sub(r'[‘’]', "'", cleaned_text)
    
    # 4. Strip common Markdown formatting
    # Remove bold (**text** or __text__)
    cleaned_text = re.sub(r'\*\*([^*]+?)\*\*', r'\1', cleaned_text)
    cleaned_text = re.sub(r'__([^_]+?)__', r'\1', cleaned_text)
    # Remove italics (*text* or _text_)
    cleaned_text = re.sub(r'\*([^*]+?)\*', r'\1', cleaned_text)
    cleaned_text = re.sub(r'_([^_]+?)_', r'\1', cleaned_text)
    # Remove headers (# Header, ## Subheader, etc.)
    cleaned_text = re.sub(r'#+\s*', '', cleaned_text)
    # Remove bullet points/list markers that are not part of the query numbering
    # (This is tricky if queries themselves might contain bullets, but for numbered lists, it's usually fine)
    cleaned_text = re.sub(r'^[*-]\s+', '', cleaned_text, flags=re.MULTILINE)


    # 5. Consolidate multiple newlines (after other replacements might create them)
    # This step is particularly important for later splitting by lines.
    cleaned_text = re.sub(r'\n{2,}', '\n', cleaned_text)

    # 6. Remove leading/trailing whitespace from each line and the whole text
    # This addresses potential extra spaces from previous replacements
    cleaned_text = '\n'.join([line.strip() for line in cleaned_text.split('\n')])
    cleaned_text = cleaned_text.strip() # Final trim for the whole text
    return cleaned_text
    
class Chatui:
    def __init__(self, shared_state) -> None:
        self.shared_state = shared_state
        self.history = gr.State([])
        self.rag_manager = ParentChildRetriever()
        
        self._interface()
        self._event()
    
    def _interface(self):
        with gr.Sidebar():
            gr.Markdown("## Navigation")
            self.chat_tab_btn = gr.Button("Chat")
            self.docs_tab_btn = gr.Button('Documents')
            self.settings_tab_btn = gr.Button("Settings")
            
        # Chat Tab
        with gr.Column(visible=True) as self.chat_page:
            gr.Markdown('## Chat interface')
            
            self.chatbot = gr.Chatbot(
                show_label=False, 
                type="messages",
                height=500,
                )
            
            with gr.Row():
                self.clear_chat_btn = gr.Button('Clear', interactive=True,size='md')
                self.regen_message_btn = gr.Button('Regenerate', interactive=True, size='md')
            self.msg = gr.Textbox(lines=1,scale=3, interactive=True, 
                                        submit_btn=True, stop_btn=True)
            
        # Documents Page
        with gr.Column(visible=False) as self.docs_page: 
            gr.Markdown('## Docuemnts and RAG')
            self.pdf_path_in = gr.File(
                file_count='multiple',
                file_types=['.pdf'],
                label='PDF file',
            )  
            with gr.Row():
                self.chunk_size_in = gr.Slider(minimum=64, maximum=1024, value=256, 
                                            step=1, label='Chunk Size', interactive=True)
                self.chunk_overlap_in = gr.Slider(minimum=0, maximum=128, value=32, 
                                            step=1, label='Chunk Overlap', interactive=True)
            with gr.Row():
                    self.process_pdf_btn = gr.Button('Process PDF', size='md')
                    self.clear_embds_btn = gr.Button('Delete Embeddings', size='md')
            self.pdf_status = gr.Textbox(label='pdf status', interactive=False, lines=3)
            
        # Settings Page
        with gr.Column(visible=False) as self.settings_page:
            gr.Markdown("## Settings")
    
    def _event(self):
        # sidebar events
        self.chat_tab_btn.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)),
            outputs=[self.chat_page, self.docs_page, self.settings_page]
        )
        self.docs_tab_btn.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)),
            outputs=[self.chat_page, self.docs_page, self.settings_page]
        )
        self.settings_tab_btn.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)),
        )
        # chat tab events
        self.msg.submit(
            fn=ollama_response,
            inputs=[self.msg, self.history, self.shared_state],
            outputs= [self.msg, self.history, self.chatbot],
        )
        self.clear_chat_btn.click(
            lambda: ([], []),
            outputs=[self.history, self.chatbot]
        )
        self.regen_message_btn.click(
            regen_response,
            [self.history, self.shared_state],
            [self.msg, self.history, self.chatbot]
        )
        self.process_pdf_btn.click(
            fn=self.rag_manager.generate_pdf_embeddings,
            inputs=[self.pdf_path_in, self.shared_state, self.chunk_size_in, self.chunk_overlap_in],
            outputs=[self.pdf_status]
        )
        self.clear_embds_btn.click(
            self.rag_manager.clear_rag_embeddings,
            outputs=[self.pdf_status]
        )

with gr.Blocks() as demo:
    app_shared_state = gr.State(SharedState())
    Chatui(app_shared_state)

demo.launch() 
