import gradio as gr
import ollama
import os
from datetime import datetime
import json
from dataclasses import dataclass
from PyPDF2 import PdfReader
import chromadb
from chromadb.config import Settings
import re
import subprocess

client = chromadb.Client(Settings(anonymized_telemetry=False))
chroma_collection = client.get_or_create_collection('docs')

@dataclass
class SharedState:
    cwd:str = os.path.dirname(os.path.realpath(__file__))
    chats_dir:str = os.path.join(cwd, 'chats')
    
    chat_model:str = r'hf.co/mlabonne/gemma-3-4b-it-abliterated-GGUF:Q4_K_M'
    emb_model:str = r'granite-embedding:latest'
    system_prompt:str = r'Act as a helpful assistant'
    temperature:float = 0.9
    top_k:int = 50
    top_p:float = 0.9

    enable_rag:bool = False

def save_chat(history, filename, shared_state):
    if not history:
        return gr.Textbox(), gr.Dropdown()
    
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
    
    updated_chats = [basename] + [f for f in os.listdir(shared_state.chats_dir) if f.endswith('json')]
    return gr.Dropdown(updated_chats, value=basename), gr.Textbox(basename)

def load_chat(filename, shared_state):
    current_state = shared_state
    history = []
    if not filename:
        return [], [], gr.Textbox('')
    filepath = os.path.join(current_state.chats_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            history = json.load(f)
        except json.JSONDecodeError:
            print(f'Empty/Corrupted Json file')

    if history[0]['role'] == ['system']:
        current_state.system_prompt = history[0]['content']
        history = history[1:]
    return history, history, current_state, gr.Textbox(filename)
    
def delete_chat(filename, shared_state):
    filepath = os.path.join(shared_state.chats_dir, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    updated_chats = [f for f in os.listdir(shared_state.chats_dir) if f.endswith('json')]
    return [], [], gr.Dropdown(updated_chats, value=updated_chats[0]), gr.Textbox('')
    
def simple_text_splitter(text, chunk_size=256, chunk_overlap=64):       
    chunks = []
    if chunk_size > chunk_overlap:
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunks.append(text[i: i + chunk_size])
    return chunks

def simple_paragraph_splitter(text, max_chunk_chars, overlap_chars):
    paragraphs = re.split(r'\n\n+', text) # split of one or more newlines
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If adding the next paragraph exceeds max_chunk_chars, close current chunk and start new
        # +2 for potential newline
        if len(current_chunk) + len(para) + 2 > max_chunk_chars and current_chunk:
            chunks.append(current_chunk)
            
            # Create overlap: take last 'overlap_chars' from current_chunk
            if len(current_chunk) >= overlap_chars:
                current_chunk = current_chunk[-overlap_chars:].strip()
            else: 
                current_chunk = ""
            
        if current_chunk:
            current_chunk += "\n\n" + para
        else:
            current_chunk = para

    if current_chunk: # Add the last chunk if any remaining
        chunks.append(current_chunk)
    
    return chunks

def embed_pdfs(files, model_config, chunk_size, chunk_overlap, split_tech):
    if files is None:
        return gr.Textbox('upload pdf file')
    char_count = 0
    chunk_count = 0
    for f in files:
        reader = PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text().strip()
            char_count += len(text)
            if split_tech == 'simple':
                chunks = simple_text_splitter(text, chunk_size, chunk_overlap)
            else:
                chunks = simple_paragraph_splitter(text, chunk_size, chunk_overlap)
            
            for j, chunk in enumerate(chunks):
                embeddings = ollama.embed(model_config.emb_model, chunk)['embeddings']
                unique_ids = f'{os.path.basename(f)}_page{i}_chunk_{j}'
                chroma_collection.add(
                    ids=unique_ids,
                    embeddings=embeddings,
                    documents=chunk
                )
                chunk_count += 1
                
    return gr.Textbox(f'Status: Embedded PDFs in ChromaDB.'
            f'\nTotal Length of Text: {char_count}'
            f'\nTotal chunks count: {chunk_count}')

def ollama_response(message, history, shared_state):
    system_prompt = shared_state.system_prompt 
    if shared_state.enable_rag is True and chroma_collection.count() > 0:
        query_emb = ollama.embed(shared_state.emb_model, message)['embeddings']
        results = chroma_collection.query(
            query_embeddings=query_emb,
            n_results=3,
            include=['documents']
        )
        context = 'Context:\n'
        if results['documents'] is not None:
            for doc in results['documents']:
                context += ' '.join(doc)
        
        system_prompt +='''Based only on the following context, answer the question.\nContext:\n---\n'''
        
        system_prompt += context
        user_message = f'Query: {message}'
        messages = ([{'role':'system', 'content': system_prompt}] + 
                    history + [{'role':'user', 'content':user_message}])
    else:
        messages = ([{'role':'system', 'content': system_prompt}] + history +
                     [{'role':'user', 'content':message}])
    
    history.append({'role':'user', 'content':message}) 
    full_response = ''  
    for chunk in ollama.chat(shared_state.chat_model, messages, stream=True):
        full_response += chunk['message']['content']
        yield ("", 
               history + [{"role": "assistant", "content": full_response}],
               history + [{"role": "assistant", "content": full_response}],
               )
    
    history.append({"role": "assistant", "content": full_response})
    return '', history, history
            
def regen_response(history, shared_state):
    if len(history) >= 2 and history[-1]['role'] == 'assistant':
        history.pop()
        message = history.pop()['content']
        response_generator = ollama_response(message, history, shared_state)
        for current_in, current_hist, current_chat_display in response_generator:
            yield current_in, current_hist, current_chat_display
    return '', history, history

def toggle_rag_enable(shared_state, toggle):
    shared_state.enable_rag = toggle
    return shared_state

def update_settings(shared_state, chat_model, emb_model, system_prompt, temperature, top_k, top_p):
    shared_state.chat_model = chat_model
    shared_state.emb_model = emb_model
    shared_state.system_promp = system_prompt
    shared_state.temperature = temperature
    shared_state.top_k = top_k
    shared_state.top_p = top_p
    return shared_state, gr.Textbox('Settings updated')
    
def clear_rag_embeddings():
    if chroma_collection.count() > 0:
        chroma_collection.delete(where={})
        return gr.Textbox(f"RAG embeddings cleared! Total documents: {chroma_collection.count()}")
    else:
        return gr.Textbox(f'Chroma collection is empty')

def available_ollama_models():
    model_names = []
    models_info = ollama.list()
    for i in models_info['models']:
        model_names.append(i['model'])
    return model_names

def pull_ollama_model(name):
    command = ['ollama', 'pull', name]
    try:
        process = subprocess.run(command, check=True)
        msg = f'Successfully Pulled {name}'
    except subprocess.CalledProcessError as e:
        print('Error:', e)
        msg = f'Failed To load Model.'
    except FileNotFoundError:
        msg = f'Ollama not found.'
    return (gr.Dropdown(available_ollama_models()), gr.Dropdown(available_ollama_models()), 
             gr.Dropdown(available_ollama_models()), gr.Textbox(msg))

def delete_ollama_model(name):
    command = ['ollama', 'rm', name]
    try:
        process = subprocess.run(command, check=True)
        msg = f'Successfully Removed {name}'
    except subprocess.CalledProcessError as e:
        print('Error:', e)
        msg = f'Failed To load Model.'
    except FileNotFoundError:
        msg = f'Ollama not found.'
    return (gr.Dropdown(available_ollama_models()), gr.Dropdown(available_ollama_models()), 
             gr.Dropdown(available_ollama_models()), gr.Textbox(msg))
    
def save_profile(shared_state, profile_name):
    if profile_name:
        data = {
            'model_name': shared_state.chat_model,
            'system_prompt': shared_state.system_prompt,
            'temperature': shared_state.temperature,
            'top_k': shared_state.top_k,
            'top_p': shared_state.top_p
        }
        filepath = os.path.join(shared_state.cwd, 'profiles', profile_name)
        filepath = filepath + '.json' if not filepath.endswith('json') else filepath
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        msg = f'Saved profile :{profile_name}'
    else:
        msg = 'Provide File Name'
    update_profile_list = [f for f in os.listdir(os.path.join(shared_state.cwd, 'profiles')) if f.endswith('json')]
    return gr.Dropdown(update_profile_list), gr.Textbox(msg)

def load_profile(shared_state, profile_name):
    filepath = os.path.join(shared_state.cwd, 'profiles', profile_name)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if data:
        shared_state.chat_model = data['chat_model']
        shared_state.system_prompt = data['system_prompt']
        shared_state.temperature = data['temperature']
        shared_state.top_k = data['top_k']
        shared_state.top_p = data['top_p']
        msg = 'Loaded Profile'
    else:
        msg = 'Profile Corrupted/Empty'
    return shared_state, gr.Textbox(msg)

class ChatUI:
    def __init__(self, shared_state):
        self.shared_state = shared_state
        self.history = gr.State([])
        
        self._interface()
        self._events()   
    
    def _interface(self):
        with gr.Sidebar():
            gr.Markdown("## Navigation")
            self.chat_btn = gr.Button("Chat")
            self.docs_btn = gr.Button('Documents')
            self.settings_btn = gr.Button("Settings")
            
        # Chat Page
        with gr.Column(visible=True) as self.chat_page:
            gr.Markdown("### Chat Interface")
            with gr.Row():
                self.current_chat_file = gr.Dropdown(
                    choices=[f for f in os.listdir('chats') if f.endswith('json')],
                    label='Chat Files',
                    interactive=True
                )
                self.chat_save_name = gr.Textbox(label='Chat Save Name', interactive=True)
            
            with gr.Row():
                self.load_chat_btn = gr.Button(value='Load Chat', size='md')
                self.save_chat_btn = gr.Button(value='Save Chat', size='md')
                self.delete_chat_btn = gr.Button(value='Delete Chat', size='md')
                
            self.chatbot = gr.Chatbot(
                show_label=False, 
                type="messages",
                height=500,
                )
            
            with gr.Row():
                self.clear_btn = gr.Button('Clear', interactive=True,size='md')
                self.regen_btn = gr.Button('Regenerate', interactive=True, size='md')
                self.enable_rag = gr.Checkbox(False, label='Enable RAG')
            self.msg = gr.Textbox(lines=1,scale=3, interactive=True, 
                                    submit_btn=True, stop_btn=True)
            
        # Documents Page
        with gr.Column(visible=False) as self.docs_page: 
                self.pdf_path = gr.File(
                    file_count='multiple',
                    file_types=['.pdf'],
                    label='PDF file',
                )  
                with gr.Row():
                    self.chunk_size = gr.Slider(minimum=128, maximum=1024, value=512, 
                                                step=1, label='Chunk Size', interactive=True)
                    self.chunk_overlap = gr.Slider(minimum=0, maximum=128, value=64, 
                                                step=1, label='Chunk Overlap', interactive=True)
                    self.split_tech = gr.Dropdown(
                        choices=['simple', 'paragraph'],
                        value='paragraph',
                        label='Splitting Technique',
                        interactive=True
                    )
                with gr.Row():
                    self.process_pdf_btn = gr.Button('Process PDF', interactive=True)
                    self.clear_embds_btn = gr.Button('Delete Embeddings', interactive=True)
                self.pdf_status = gr.Textbox(label='pdf status', interactive=False, lines=3)
                
        # Settings Page
        with gr.Column(visible=False) as self.settings_page:
            gr.Markdown("### Settings")
            with gr.Accordion('Checkpoints Selection', open=False):
                self.chat_models_dropdown = gr.Dropdown(
                    available_ollama_models(),
                    label='Choose Chat Model',
                    interactive=True,
                )
                self.emb_models_dropdown = gr.Dropdown(
                    available_ollama_models(),
                    label='Choose Embedding Model',
                    interactive=True,
                )
            
            with gr.Accordion('Model Settings', open=False):
                with gr.Row():
                    self.profile_dropdown = gr.Dropdown(
                        choices=[f for f in os.listdir('profiles') if f.endswith('json')],
                        label='Available Profiles',
                        interactive=True
                    )
                    self.load_profile_btn = gr.Button('Load Profile')
                self.system_prompt_in = gr.Textbox(
                    value='Act as a helpful assistant',
                    label='Chat Model System Prompt',
                    interactive=True, 
                )
                
                with gr.Row():
                    self.temp_in = gr.Slider(
                        minimum=0,
                        maximum=1.0,
                        step=0.1,
                        value=0.9,
                        label='Temperature',
                        interactive=True
                    )
                    self.top_k_in = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=40,
                        label='Top-K',
                        interactive=True
                    )
                    self.top_p_in = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=0.9,
                        label='Top-P',
                        interactive=True
                    )
                with gr.Row():
                    self.new_profile_in = gr.Textbox(label='New Profile Name', interactive=True)
                    self.save_profile_btn = gr.Button('save Profile')
                    
            with gr.Accordion('Model Downloading and Removing', open=False):
                with gr.Row():
                    self.download_model_in = gr.Textbox(
                        label='Checkpoint Name/ HF Directory',
                        placeholder='hf.co/ggml-org/SmolLM3-3B-GGUF:Q4_K_M',
                        interactive=True, 
                    )
                    self.download_model_btn = gr.Button('Download Model', interactive=True)
                with gr.Row():
                    self.delete_model_in = gr.Dropdown(
                        choices=available_ollama_models(),
                        label='Name of checkpoint to delete',
                        interactive=True
                    )
                    self.delete_model_btn = gr.Button('Delete model', interactive=True)
            self.update_settings_btn = gr.Button('Update Model Settings', size='md',interactive=True)
            self.settings_status = gr.Textbox(label='Status', interactive=False)

                
    def _events(self):
        self.chat_btn.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)),
            outputs=[self.chat_page, self.docs_page, self.settings_page]
        )
        self.docs_btn.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)),
            outputs=[self.chat_page, self.docs_page, self.settings_page]
        )
        self.settings_btn.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)),
            outputs=[self.chat_page, self.docs_page, self.settings_page]
        )
        
        self.process_pdf_btn.click(
            fn=embed_pdfs,
            inputs=[self.pdf_path, self.shared_state, self.chunk_size, 
                    self.chunk_overlap, self.split_tech],
            outputs=[self.pdf_status]
        )
        self.msg.submit(
            fn=ollama_response,
            inputs=[self.msg, self.history, self.shared_state],
            outputs= [self.msg, self.history, self.chatbot],
        )
        self.save_chat_btn.click(
            save_chat,
            [self.history, self.chat_save_name, self.shared_state],
            [self.current_chat_file, self.chat_save_name]
        )
        self.load_chat_btn.click(
            load_chat,
            [self.current_chat_file, self.shared_state],
            [self.history, self.chatbot, self.shared_state, self.chat_save_name]
        )
        self.delete_chat_btn.click(
            delete_chat,
            [self.current_chat_file, self.shared_state],
            [self.history, self.chatbot, self.current_chat_file, self.chat_save_name]
        )
        self.clear_btn.click(
            lambda: ([], []),
            outputs=[self.history, self.chatbot]
        )
        self.regen_btn.click(
            regen_response,
            [self.history, self.shared_state],
            [self.msg, self.history, self.chatbot]
        )

        self.enable_rag.change(
            toggle_rag_enable,
            [self.shared_state, self.enable_rag],
            [self.shared_state]
        )
        self.update_settings_btn.click(
            update_settings,
            inputs=[self.shared_state, self.chat_models_dropdown, self.emb_models_dropdown,
                    self.system_prompt_in, self.temp_in, self.top_k_in, self.top_p_in],
            outputs=[self.shared_state, self.settings_status]
        )
        self.clear_embds_btn.click(
            clear_rag_embeddings,
            outputs=[self.pdf_status]
        )
        self.download_model_btn.click(
            pull_ollama_model,
            [self.download_model_in],
            [self.chat_models_dropdown, self.emb_models_dropdown, 
             self.delete_model_in, self.settings_status]
        )
        self.delete_model_btn.click(
            delete_ollama_model,
            [self.delete_model_in],
            [self.chat_models_dropdown, self.emb_models_dropdown, 
             self.delete_model_in, self.settings_status]
        )
        self.load_profile_btn.click(
            load_profile,
            [self.shared_state, self.profile_dropdown],
            [self.shared_state, self.settings_status]
        )
        self.save_profile_btn.click(
            save_profile,
            [self.shared_state, self.new_profile_in],
            [self.profile_dropdown, self.settings_status]
        )

with gr.Blocks() as demo:
    app_shared_state = gr.State(SharedState())
    ChatUI(app_shared_state)

demo.launch()
