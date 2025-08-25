import ollama
import re
import os
from datetime import datetime
from ddgs import DDGS
from PyPDF2 import PdfReader
import chromadb
from chromadb.config import Settings
import hashlib
import gradio as gr
import json
from nltk.tokenize import word_tokenize, sent_tokenize

class SharedState:
    def __init__(self) -> None:
        self.chat_model = r'gemma3:4b'
        self.emb_model = r'granite-embedding:latest'
        self.num_ctx = 8192
        self.temperature = 0.9
        self.top_k = 40
        self.top_p = 0.7
        
        self.rag_flag = False
        self.ddgs_flag = False
        
        self.num_tokens = 64
        self.token_overlap = 8
        self.n_results = 5
        
        self.chats_dir = r'chats'
        self.system_prompt = r'You are helpful assistant.'
        

    def _toggle_rag_flag(self, rag_flag_in):
        self.rag_flag = rag_flag_in
        return self

    def _toggle_ddgs_flag(self, ddgs_flag_in):
        self.ddgs_flag = ddgs_flag_in
        if self.ddgs_flag is True:
            print(f'\nDuckduckgo search Enabled')
        return self
    
    def _update_model_settings(
        self, 
        chat_model,
        system_prompt,
        emb_model,
        temperature,
        top_k, 
        top_p,
        num_ctx
    ):
        self.chat_model = chat_model
        self.system_prompt = system_prompt
        self.emb_model = emb_model
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.num_ctx = num_ctx
        
        status_dict = {
            'Chat Model': chat_model,
            'Embedding Model': emb_model,
            'System Prompt': system_prompt,
            'System Prompt Length': len(system_prompt),
            'Temperature': temperature,
            'Top-K': top_k,
            'Top-P': top_p,
            'Context Length': num_ctx
        }
        
        max_key_length = max(len(k) for k in status_dict.keys())
        status_output = 'Updated Settings:\n' + '-'*100 + '\n'
        for k, v in status_dict.items():
            status_output += f'{k:<{max_key_length+10}} {v}\n'

        return self, f'```\n{status_output}\n```'

def sentence_aware_splitter(text, num_tokens, token_overlap):
    if num_tokens <= token_overlap:
        raise ValueError("num_tokens must be greater than token_overlap")

    # Split the text into sentences
    sentences = sent_tokenize(text)
    if not sentences:
        return []

    # Create a list of tuples, where each tuple contains the sentence and its word count
    sentence_data = [(s, len(word_tokenize(s))) for s in sentences]

    chunks = []
    current_sentence_idx = 0
    while current_sentence_idx < len(sentence_data):
        chunk_sentences = []
        chunk_token_count = 0
        end_sentence_idx = current_sentence_idx

        # --- Build a chunk by adding whole sentences ---
        # Greedily add sentences to the chunk until the token limit is reached.
        while end_sentence_idx < len(sentence_data):
            sentence, count = sentence_data[end_sentence_idx]
            
            # If a single sentence is longer than num_tokens, it becomes its own chunk.
            if not chunk_sentences and count > num_tokens:
                chunk_sentences.append(sentence)
                chunk_token_count += count
                end_sentence_idx += 1
                break

            # If adding the next sentence would exceed the token limit, stop.
            if chunk_token_count + count > num_tokens:
                break
            
            # Otherwise, add the sentence to the current chunk.
            chunk_sentences.append(sentence)
            chunk_token_count += count
            end_sentence_idx += 1
        
        # Add the completed chunk to the list of chunks.
        chunks.append(" ".join(chunk_sentences))

        # If we've processed all sentences, we're done.
        if end_sentence_idx >= len(sentence_data):
            break

        # --- Determine the start of the next chunk for overlap ---
        # To create an overlap, we step back from the end of the current chunk.
        overlap_token_count = 0
        next_start_idx = end_sentence_idx - 1
        
        # Keep moving the start index back until we have enough tokens for the overlap.
        while next_start_idx > current_sentence_idx and overlap_token_count < token_overlap:
            # We subtract 1 because next_start_idx is an index
            overlap_token_count += sentence_data[next_start_idx][1]
            next_start_idx -= 1
        
        # The next chunk starts at the calculated index.
        # We use max() to ensure the process always moves forward.
        current_sentence_idx = max(next_start_idx + 1, current_sentence_idx + 1)
            
    return chunks
    
class FileManager:
    def save_chat(self, history: list, filename: str, shared_state):
        if not history:
            return gr.Dropdown(), gr.Textbox()
        
        basename = filename.strip()
        if not basename:
            now = datetime.now()
            time_str = now.strftime('%Y-%m-%d-%H-%M-%S')
            basename = f'Chat_{time_str}'
        
        if not basename.endswith('json'):
            basename += '.json'
        
        filepath = os.path.join(shared_state.chats_dir, basename)
        with open(filepath, 'w', encoding='utf-8') as f:
            content = [{'role':'system', 'content':shared_state.system_prompt}] + history
            json.dump(content, f, indent=4)
        print(f'\nChat Successfully saved at {filepath}')
        updated_chats = [basename] + [f for f in os.listdir(shared_state.chats_dir) if f.endswith('json')]
        return gr.Dropdown(updated_chats, value=basename), gr.Textbox(basename)

    def load_chat(self, filename:str, shared_state):
        history = []
        if not filename:
            return [], [], shared_state, gr.Textbox()
        filepath = os.path.join(shared_state.chats_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
                history = json.load(f)
        print(f'\nLoaded {filename} successfully.')
        if history[0]['role'] == ['system']:
            shared_state.system_prompt = history[0]['content']
            history = history[1:]
        return history, history, shared_state, gr.Textbox(filename)
        
    def delete_chat(self, filename:str, shared_state):
        filepath = os.path.join(shared_state.chats_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        print(f'\nDeleted Chat {filename}. \u2705')
        updated_chats = [f for f in os.listdir(shared_state.chats_dir) if f.endswith('json')]
        return [], [], gr.Dropdown(updated_chats, value=updated_chats[0]), gr.Textbox('')

class ModelManager:
    def _get_available_models(self):
        model_names = []
        models_info = ollama.list()
        for i in models_info['models']:
            model_names.append(i['model'])
        return model_names 

class WebAgent:
    def _rephrase_query(self, query, chat_history, shared_state):
        now = datetime.now().strftime('%Y-%m-%d')
        
        query_reprahser_system_prompt = (f'''
        Paraphrase the user query  cleverly three unique times in a way that it can be used to get best *upto date* results from a search engine.
        You can use the current date: {now} to imporve the query. 
        You should **ONLY** provide suggested query in JSON format as provided below.
        ''' +
        '''
        ```json
        {
            "query": [
                "query1", 
                "query2", 
                "query3"
                ]
        }
        ```
        ''')
        ollama_messages = (
        [{'role': 'system', 'content': query_reprahser_system_prompt}] +
        chat_history +
        [{'role': 'user', 'content': query}]
        )
        response = ollama.chat(shared_state.chat_model, ollama_messages, options={'temperature':1.0})['message']['content']
        try:
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.strip()
            data = json.loads(json_str)
            queries = data.get("query", [])
            return queries
            
        except json.JSONDecodeError as e:
            print(f'Error: Failed to decode JSON from LLM response. Error: {e}')
            return [query]
    
    def _get_domains_suggestions(self, query, shared_state):
        sites_suggester_system_prompt = '''
        You are a relevant site sugesting agent. Your role is to provide a list of three trustworthy domains which may have relevant data to user query.
        You should **ONLY** provide suggested domains list in JSON format as provided below.
        ```json
        {
            "suggestions": ["wwww.domain1.com", "www.domain2.com", "www.domain3.com"]
        }
        ```
        '''
        ollama_messages = [
            {'role': 'system', 'content': sites_suggester_system_prompt},
            {'role': 'user', 'content': query}
        ]
        try:
            response = ollama.chat(shared_state.chat_model, ollama_messages)['message']['content']
            json_match = re.search(r'```json\s*(\{.*\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.strip()
            data = json.loads(json_str)
            suggestions = data.get("suggestions", [])
            return suggestions
            
        except json.JSONDecodeError as e:
            print(f'Error: Failed to decode JSON from LLM response. Error: {e}')
            return []
        except KeyError as e:
            print(f"Error: 'suggestions' key not found in the JSON response. Error: {e}")
            return []
    
    def _search_web_chain(self, user_message, chat_history, shared_state):
        # 1. Generate the intial rephrased query
        rephrased_queries = self._rephrase_query(user_message, chat_history, shared_state)
        
        # 2. Get trusted sites 
        #sites_suggestion = self._get_domains_suggestions(rephrased_queries[0], shared_state)

        metadata = {'context': '', 'sources': set(), 'rephrased_query': rephrased_queries}
        
        #3. Perform a general search first
        
        for query in rephrased_queries:
            web_search_data = DDGS().text(query=query, max_results = 3)
            for search in web_search_data:
                metadata['context'] += '\n- ' + f'({search['href']})\t' + search['body']
                metadata['sources'].add(search['href'])
        
        # 4. Perform targeted searches for each suggested domain
        # for site in sites_suggestion:
        #     if site:
        #         targeted_query = f'{rephrased_query} {site}'
        #         metadata['rephrased_query'].append(targeted_query)
        #         web_searches_targeted = DDGS().text(query=targeted_query, max_results = 3)
        #         for search in web_searches_targeted:
        #             metadata['context'] += '\n- ' + f'({search['href']})\t' + search['body']
        #             metadata['sources'].add(search['href'])
        return metadata           
    
              
class DocumentAgent:
    def _clean_pdf_text(self, text):
        text = re.sub(r'[“”]', '"', text)
        # Convert curly single quotes to straight single quotes
        text = re.sub(r'[‘’]', "'", text)
        text = re.sub(r'\n+', ' ', text)
        return text
    
    def _read_pdf(self, pdfs_dir):
        total_text = ''
        for pdf_file in pdfs_dir:
            reader = PdfReader(pdf_file)
            for pages in reader.pages:
                text = pages.extract_text()
                text = self._clean_pdf_text(text)
                total_text += text
        return total_text 

class RAGPipe:
    def __init__(self) -> None:
        self.chunks_stoarge = {}
        client = chromadb.Client(settings = Settings(anonymized_telemetry=False))
        self.chroma_collection = client.get_or_create_collection('docs')
        self.initial_vectorstore_count = self.chroma_collection.count()
        self.web_agent = WebAgent()
        self.pdf_agent = DocumentAgent()
    
    def _ingest_chunks(self, text, shared_state):
        chunks = sentence_aware_splitter(
            text, 
            shared_state.num_tokens, 
            shared_state.token_overlap
            )
        for chunk in chunks:
            chunk_id = hashlib.sha256(chunk.encode()).hexdigest()
            self.chunks_stoarge[chunk_id] = chunk
            
            embeddings = ollama.embed(model = shared_state.emb_model, input = chunk)['embeddings']
            self.chroma_collection.add(
                ids=chunk_id,
                embeddings=embeddings,
                documents=chunk
            )
        
    def _ingest_documents(self, shared_state, filespath):
        if filespath:
            text = self.pdf_agent._read_pdf(filespath)
            self._ingest_chunks(text, shared_state)
            
            return (
                'Successfully ingested chunks into vector store.'
                f'Vector store count: {self.chroma_collection.count()}'
            )
        else:
            raise FileExistsError("PDF's directory does not exsist.")
    
    def _retrieve_query_docs(self, query, shared_state):
        query_embeddings = ollama.embed(shared_state.emb_model, query)['embeddings']
        query_retreival = self.chroma_collection.query(
            query_embeddings=query_embeddings,
            n_results=shared_state.n_results,
            include=['documents']
        )
        relevant_chunks = []
        if query_retreival['documents']:
            for docs in query_retreival['documents']:
                for doc in docs:
                    relevant_chunks.append(doc)
        return '\n'.join(relevant_chunks)
    
    def _parse_thinking(self, model_repsonse):
        thinking_tag_start = '<think>'
        thinking_tag_end = '</think>'
        # Case 1: The thinking block has started and finished.
        if thinking_tag_start in model_repsonse and thinking_tag_end in model_repsonse:
            # Use regex to definitively separate the completed thinking block from the main message.
            match = re.search(r'<think>(.*?)</think>(.*)', model_repsonse, re.DOTALL)
            if match:
                return match.group(1).strip(), match.group(2).strip()
            else:
                # Fallback if regex fails, though unlikely with both tags present.
                return "", model_repsonse

        # Case 2: The thinking block has started but not yet finished.
        elif thinking_tag_start in model_repsonse:
            # Split the content at the opening <think> tag.
            parts = model_repsonse.split(thinking_tag_start, 1)
            main_message_part = parts[0].strip()
            thinking_message_part = parts[1] # Don't strip here to preserve streaming space.
            return thinking_message_part, main_message_part

        # Case 3: No <think> tag has appeared yet.
        else:
            # Everything is part of the main message.
            return "", model_repsonse.strip()
    
    def _filter_metadata(self, chat_history):
        ollama_history = []
        for message in chat_history:
            if not message.get('metadata'):
                ollama_history.append({'role': message['role'], 'content': message['content']})
        return ollama_history
    
    def _ollama_response(self, user_message, chat_history, shared_state):
        context = ''
        metadata = {}
        if shared_state.rag_flag is True:
            context = self._retrieve_query_docs(user_message, shared_state)
            system_prompt = f'You are a helpful assistant who answers question based **only** on provided context.'
        elif shared_state.ddgs_flag is True:
            metadata = self.web_agent._search_web_chain(user_message, chat_history, shared_state)
            context = metadata['context'] 
            system_prompt = f'''
            You will be provided with data retreived from web searches. It might contain old and updated conflicting data.
            Carefully analyze the context provided and then answer the user query with the right answer.
            '''
        else:
            system_prompt = shared_state.system_prompt

        ollama_history = self._filter_metadata(chat_history)
        ollama_messages = (
            [{'role': 'system', 'content': system_prompt}] + 
            ollama_history +
            [{'role': 'user', 'content': f'USER: {user_message}\n\nCONTEXT: {context}'}]
        )
        
        # add latest user message to chat history
        chat_history.append({'role': 'user', 'content': user_message})

        # add place holders for streaming
        chat_history.append({'role': 'assistant', 'content': '', 'metadata': {'title': 'thinking'}})
        chat_history.append({'role': 'assistant', 'content': ''})
        
        full_response = ''
        response_streamer = ollama.chat(
            shared_state.chat_model, 
            ollama_messages,
            stream=True,
            options = {
                'temperature': shared_state.temperature, 
                'top_k': shared_state.top_k,
                'top_p': shared_state.top_p,
                'num_ctx': shared_state.num_ctx
                }
            )

        for chunk in response_streamer:
            full_response += chunk['message']['content']
            
            # parse out thinking and main message part
            thinking_message, main_message = self._parse_thinking(full_response)
            
            # Update the placeholder messages with the new content
            chat_history[-2]['content'] = "🤔\n" + thinking_message
            chat_history[-1]['content'] = main_message
            
            yield '', chat_history, chat_history
        
        # --- Finalization Step ---
        final_thinking, _ = self._parse_thinking(full_response) 
        # If there was no thinking content, remove its placeholder
        if not final_thinking.strip():
            chat_history.pop(-2)

        if context:
            if metadata:
                context = (
                    f'REPHRASED QUERY: {metadata['rephrased_query']}'
                    f'\nCONTEXT:\n{context}'
                    f'\nSOURCE: {metadata['sources']}'
                )
            chat_history.append({'role': 'assistant', 'content': context, 'metadata': {'title': 'metadata'}})
        yield '', chat_history, chat_history

    def _regen_response(self, chat_history, shared_state):
        message_to_regenerate = None
        
        while len(chat_history) > 0:
            last_entry = chat_history.pop()
            print(last_entry)
            if last_entry.get('role') == 'user':
                message_to_regenerate = last_entry['content']
                break
            
        if message_to_regenerate:
            yield from self._ollama_response(message_to_regenerate, chat_history, shared_state)
        else:
            yield '', chat_history, chat_history
            
class ChatUi:
    def __init__(self, shared_state: gr.State) -> None:
        self.shared_state = shared_state
        self.chat_history = gr.State([])
        
        self.rag_pipe = RAGPipe()
        self.files_manager = FileManager()
        self.model_manager = ModelManager()
        self._interface()
        self._events()
    
    def _interface(self):
        # sidebar defintion
        with gr.Sidebar():
            gr.Markdown('Navigation')
            self.chat_tab_btn = gr.Button('Chat')
            self.docs_tab_btn = gr.Button('Documents')
            self.settings_tab_btn = gr.Button('Settings')
            
        # chat page
        with  gr.Column(visible=True) as self.chat_page:
            gr.Markdown('## Chat')
            
            with gr.Row():
                self.load_chat_dropdown = gr.Dropdown(
                    choices=[f for f in os.listdir(self.shared_state.value.chats_dir) if f.endswith('json')],
                    label='File to load',
                    interactive=True,
                )
                self.chat_save_name_in = gr.Textbox(label='New Chat Save name', interactive=True)

            with gr.Row():
                self.save_chat_btn = gr.Button('Save Chat File', size='md')
                self.load_chat_btn = gr.Button('Load Chat File', size='md')
            self.chatbot = gr.Chatbot(
                show_label=False,
                type='messages',
                height=500
            )
            with gr.Row():
                self.clear_msg_btn = gr.Button(value='Clear Complete Chat', interactive=True)
                self.web_search_flag_in = gr.Checkbox(value=False, label='Duckduckgo Search')
                self.rag_flag_in = gr.Checkbox(value=False, label='RAG')
                self.regen_msg_btn = gr.Button(value='Regenerate last message', interactive=True)
            
                
            self.user_message_in = gr.Textbox(
                lines=1, 
                scale=3,
                interactive=True,
                submit_btn=True,
                stop_btn=True
            )
            
        # documents page
        with gr.Column(visible=False) as self.docs_page:
            gr.Markdown('## Documents')
            self.pdf_file_in = gr.File(
                file_count='multiple',
                file_types=['.pdf'],
                label='PDF File'
            )
            self.process_pdf_btn = gr.Button(value="Process PDF Files", interactive=True)
            self.doc_page_display = gr.Textbox(label = 'Documents Status', interactive=False)
        
        # settings page
        with gr.Column(visible=False) as self.settings_page:
            gr.Markdown('## Settings')
            with gr.Accordion(label='Model Selection', open=False):
                gr.Markdown('**Warning: Carefully Select the Chat model and embedding Models.**')
                available_models = self.model_manager._get_available_models()
                self.chat_model_in = gr.Dropdown(
                    choices=available_models,
                    label='Carefully Chose Chat Model.',
                    interactive=True,
                    value=available_models[0]
                )
                self.emb_model_in = gr.Dropdown(
                    choices=available_models,
                    label='Carefully Chose Embedding Model.',
                    interactive=True,
                    value=available_models[0]
                )
                self.system_prompt_in = gr.Textbox(
                    value=self.shared_state.value.system_prompt,
                    label='System Prompt',
                    lines=5,
                    interactive=True
                )
            with gr.Accordion(label='Model Inference Parameters', open=False):
                with gr.Row():
                    self.temeprature_in = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.9,label='Temperature', interactive=True)
                    self.top_k_in = gr.Slider(minimum=1, maximum=50, step=1, value=40, label='Top-K', interactive=True)
                    self.top_p_in = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.7, label='Top-P', interactive=True)
                    self.n_ctx_in = gr.Slider(minimum=10, maximum=18, step=1, value=12, label='2^(n) Context Length', interactive=True)
            
            self.update_model_settings_btn = gr.Button(value='Update Settings', interactive=True)
            gr.Markdown('Settings Status')
            self.settings_page_status = gr.Markdown(value='``````', label='Settings Status', visible=True)
    
    
    def _events(self):
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
            outputs=[self.chat_page, self.docs_page, self.settings_page]
        )
        # chat page events
        self.user_message_in.submit(
            self.rag_pipe._ollama_response,
            [self.user_message_in, self.chat_history, self.shared_state],
            [self.user_message_in, self.chat_history, self.chatbot]
        )
        self.regen_msg_btn.click(
            self.rag_pipe._regen_response,
            [self.chat_history, self.shared_state], 
            [self.user_message_in, self.chat_history, self.chatbot]
        )
        self.web_search_flag_in.change(
            self.shared_state.value._toggle_ddgs_flag,
            [self.web_search_flag_in],
            [self.shared_state]
        )
        
        self.load_chat_btn.click(
            self.files_manager.load_chat,
            [self.load_chat_dropdown, self.shared_state],
            [self.chat_history, self.chatbot, self.shared_state, self.chat_save_name_in]
        )
        
        self.save_chat_btn.click(
            self.files_manager.save_chat,
            [self.chat_history, self.chat_save_name_in, self.shared_state],
            [self.load_chat_dropdown, self.chat_save_name_in]
        )

        self.clear_msg_btn.click(
            lambda: ([], []),
            outputs=[self.chatbot, self.chat_history],
        )
        
        # document page events
        self.process_pdf_btn.click(
            self.rag_pipe._ingest_documents,
            [self.shared_state, self.pdf_file_in],
            [self.doc_page_display]
        )
        
        self.update_model_settings_btn.click(
            self.shared_state.value._update_model_settings,
            [self.chat_model_in, self.system_prompt_in, self.emb_model_in, 
             self.temeprature_in, self.top_k_in, self.top_p_in, self.n_ctx_in],
            [self.shared_state, self.settings_page_status]
        )
                  
with gr.Blocks() as demo:
    app_shared_state = gr.State(SharedState())
    ChatUi(app_shared_state)

demo.launch() 
