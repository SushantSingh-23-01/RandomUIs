import gradio as gr
import ollama
import os
from datetime import datetime
import json
from dataclasses import dataclass
from PyPDF2 import PdfReader
import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(anonymized_telemetry=False))
chroma_collection = client.get_or_create_collection('docs')

@dataclass
class model_config:
    available_models = [i['model'] for i in ollama.list()['models']]
    chat_model: str = available_models[0] if available_models else ''
    emb_model: str = available_models[0] if available_models else ''
    system_prompt:str = r'Act as a helpful assistant'
    
    chats_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'chats')

def save_chat(history, filename, model_config):
    if not history:
        return gr.Textbox(), gr.Dropdown()
    
    basename = filename.strip()
    if not basename:
        now = datetime.now()
        time_str = now.strftime('%Y-%m-%d-%H-%M-%S')
        basename = f'Chat_{time_str}'
    
    if not basename.endswith('json'):
        basename += '.json'
    
    filepath = os.path.join(model_config.chats_dir, basename)
    with open(filepath, 'w', encoding='utf-8') as f:
        content = [{'role':'system', 'content':model_config.system_prompt}] + history
        json.dump(content, f, indent=4)
    
    updated_chats = [basename] + [f for f in os.listdir(model_config.chats_dir )if f.endswith('json')]
    return gr.Dropdown(updated_chats, value=basename), gr.Textbox(basename)

def load_chat(filename, model_config):
    history = []
    if not filename:
        return [], [], gr.Textbox('')
    
    filepath = os.path.join(model_config.chats_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            history = json.load(f)
        except json.JSONDecodeError:
            print(f'Empty/Corrupted Json file')

    if history[0]['role'] == ['system']:
        model_config.system_prompt = history[0]['content']
        history = history[1:]
    return history, history, model_config, gr.Textbox(filename)
    
def delete_chat(filename, chats_dir):
    filepath = os.path.join(chats_dir, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    updated_chats = [f for f in os.listdir(chats_dir )if f.endswith('json')]
    return [], [], gr.Dropdown(updated_chats, value=updated_chats[0]), gr.Textbox('')
    
def simple_text_splitter(text, chunk_size=256, chunk_overlap=64):       
    chunks = []
    if chunk_size > chunk_overlap:
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunks.append(text[i: i + chunk_size])
    return chunks

def read_pdf(filepath, model_config, chunk_size, chunk_overlap):
    if filepath is None:
        return gr.Textbox('upload pdf file')
    
    reader = PdfReader(filepath)
    for i, page in enumerate(reader.pages):
        text = page.extract_text().strip()
        chunks = simple_text_splitter(text, chunk_size, chunk_overlap)
        for j, chunk in enumerate(chunks):
            embeddings = ollama.embed(model_config.emb_model, chunk)['embeddings']
            chroma_collection.add(
                ids=f'page{i}_chunk{j}',
                embeddings=embeddings,
                documents=chunk
            )
    return 'status: embedded pdf'

def ollama_response(message, history, model_config):
    system_prompt = model_config.system_prompt 
    if chroma_collection.count() > 0:
        query_emb = ollama.embed(model_config.emb_model, message)['embeddings']
        results = chroma_collection.query(
            query_embeddings=query_emb,
            n_results=3,
            include=['documents']
        )
        context = 'Context:\n'
        if results['documents'] is not None:
            for doc in results['documents']:
                context += ' '.join(doc)
        
        system_prompt +='\nContext:\n' + context
        message = f'Query: {message}'
        
    history.append({'role':'user', 'content':message})
    messages = [{'role':'system', 'content': system_prompt}] + history
    response = ollama.chat(model_config.chat_model, messages)['message']['content']  
    history.append({"role": "assistant", "content": response})
    return '', history, history
            
def regen_response(history, model_config):
    if len(history) >= 2 and history[-1]['role'] == 'assistant':
        history.pop()
        message = history.pop()['content']
        _, history, history = ollama_response(message, history, model_config)
    
    return '', history, history

class ChatUI:
    def __init__(self, model_config):
        self.model_config = model_config
        self.history = gr.State([])
        
        self._interface()
        self._events()   
    
    def _interface(self):
        with gr.Accordion('Chat Files', open=False):
            self.current_chat_file = gr.Dropdown(
                choices=[f for f in os.listdir(self.model_config.chats_dir) if f.endswith('json')],
                label='Chat Files',
                interactive=True
            )
            self.chat_save_name = gr.Textbox(label='Chat Save Name', interactive=True)
            with gr.Row():
                self.load_chat_btn = gr.Button(value='Load Chat', size='md')
                self.save_chat_btn = gr.Button(value='Save Chat', size='md')
                self.delete_chat_btn = gr.Button(value='Delete Chat', size='md')
                
        with gr.Accordion('Extra files', open=False):
            with gr.Row():
                self.pdf_path = gr.File(
                    file_count='single',
                    file_types=['.pdf'],
                    label='PDF file',
                )
                self.process_pdf_btn = gr.Button('Process PDF', interactive=True)
            with gr.Row():
                self.chunk_size = gr.Slider(minimum=128, maximum=1024, value=512, 
                                            step=1, label='Chunk Size', interactive=True)
                self.chunk_overlap = gr.Slider(minimum=0, maximum=128, value=64, 
                                            step=1, label='Chunk Overlap', interactive=True)
                self.pdf_status = gr.Textbox(label='pdf status', interactive=False)
            
        self.chatbot = gr.Chatbot(
            show_label=False, 
            type="messages",
            height=500,
            )
        with gr.Row():
            self.msg = gr.Textbox(lines=1,scale=3, interactive=True, 
                                  submit_btn=True, stop_btn=True)
            # with gr.Column():
            #     self.clear = gr.ClearButton([self.msg, self.chatbot, self.history])
            #     self.undo_btn = gr.Button(value='undo')
            
    def _events(self):
        self.process_pdf_btn.click(
            fn=read_pdf,
            inputs=[self.pdf_path, self.model_config, 
                    self.chunk_size, self.chunk_overlap],
            outputs=[self.pdf_status]
        )
        self.msg.submit(
            fn=ollama_response,
            inputs=[self.msg, self.history, self.model_config],
            outputs= [self.msg, self.history, self.chatbot],
        )
        self.save_chat_btn.click(
            save_chat,
            [self.history, self.chat_save_name, self.model_config],
            [self.current_chat_file, self.chat_save_name]
        )
        self.load_chat_btn.click(
            load_chat,
            [self.current_chat_file, self.model_config],
            [self.history, self.chatbot, self.current_chat_file]
        )

with gr.Blocks() as demo:
    app_model_config = gr.State(model_config())
    ChatUI(app_model_config)

demo.launch()
