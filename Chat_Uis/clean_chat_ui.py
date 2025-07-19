import gradio as gr
import json
import ollama
from datetime import datetime
import os
from dataclasses import dataclass
from typing import Any, List

@dataclass
class SharedState:
    available_models = [i['model'] for i in ollama.list()['models']]
    model_name: str = available_models[0] if available_models else ''
    system_prompt: str = 'Act as an assistant.'
    chats_folder: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'chats')
    
class ChatTab:
    def __init__(self, shared_state):
        with gr.Row():
            self.chat_files_list = gr.Dropdown(
                choices=[f for f in os.listdir(shared_state.value.chats_folder) if f.endswith('json')],
                label='Chat Files',
                interactive=True
                )
            self.chat_file_name = gr.Textbox(label='Name of new chat', interactive=True)
        
        with gr.Row():
            self.load_chat_btn = gr.Button(value='Load Chat', size='md')
            self.save_chat_btn = gr.Button(value='Save Chat', size='md')
            self.delete_chat_btn = gr.Button(value='Delete Chat', size='md')
            
        self.chatbot = gr.Chatbot(type="messages", min_height=500)
        
        with gr.Row():
            self.regen_btn = gr.Button(value='Regenerate', size='md')
            self.undo_btn = gr.Button(value='Undo', size='md')
            self.clear_btn = gr.Button(value='Clear Chat', size='md')
            
        self.msg = gr.Textbox(lines=1, scale=3, interactive=True, submit_btn=True)
    
    def get_ollama_response(self, user_message, chat_history_state, shared_state):
        try:
            chat_history_state.append({"role": "user", "content": user_message}) 
            messages = [{'role':'system','content':shared_state.system_prompt}] + chat_history_state
            response = ollama.chat(model=shared_state.model_name, messages=messages)['message']['content']  
            chat_history_state.append({"role": "assistant", "content": response})
        except ollama.ResponseError as e:
            print(f'Ollama Error: {e}')
        return '', chat_history_state, chat_history_state
    
    def regenerate_message(self, chat_history_state, shared_state):
        if len(chat_history_state) >= 2 and chat_history_state[-1]['role'] == 'assistant':
            chat_history_state.pop()
            last_user_message = chat_history_state.pop()['content']
            _, chat_history_state, chat_history_state = self.get_ollama_response(
            last_user_message,chat_history_state, shared_state
            )
        return '', chat_history_state, chat_history_state
    
    def undo_user_message(self, chat_history_state):
        if (
            len(chat_history_state) >= 2 and  
            chat_history_state[-1]['role'] == 'assistant' and
            chat_history_state[-2]['role'] == 'user'
            ):
            for _ in range(2):
                chat_history_state.pop()
        return chat_history_state, chat_history_state

def save_chat_file(chat_history_state, shared_state, chat_file_name):
    if not chat_history_state:
        return gr.Textbox(), gr.Dropdown()   
    base_file_name = chat_file_name.strip()
    
    if not base_file_name:
        now = datetime.now()
        time_str = now.strftime('%Y-%m-%d-%H-%M-%S')
        base_file_name = f'Chat_{time_str}'
        
    if not base_file_name.endswith('json'):
        base_file_name += '.json'
        
    file_path = os.path.join(shared_state.chats_folder, base_file_name)
    with open(file_path, "w", encoding='utf-8') as f:
        messages = [{'role':'system','content':shared_state.system_prompt}] + chat_history_state
        json.dump(messages, f, indent=4) 
        
    updated_chat_files_list = [base_file_name] + [f for f in os.listdir(shared_state.chats_folder) if f.endswith('json')]
    
    return gr.Dropdown(updated_chat_files_list,value=base_file_name), gr.Textbox(base_file_name)

def load_chat_file(shared_state, chat_file_name):
    chat_history_state = []
    if not chat_file_name:
        return [], [], gr.Textbox(value="")
    
    file_path = os.path.join(shared_state.chats_folder, chat_file_name)
    with open(file_path, "r", encoding='utf-8') as f:
        try:
            chat_history_state = json.load(f)
        except json.JSONDecodeError:
            print(f'Chat File Empty/Corrupted!')
    
    if chat_history_state[0]['role'] == 'system':
        shared_state.system_prompt = chat_history_state[0]['content']
        chat_history_state = chat_history_state[1:]
    return chat_history_state, chat_history_state, shared_state, gr.Textbox(chat_file_name)

def delete_chat_file(chat_history_state, shared_state, chat_file_name):
    file_path = os.path.join(shared_state.chats_folder, chat_file_name)
    
    if os.path.exists(file_path):
        os.remove(file_path)
    if chat_history_state:
        chat_history_state = []
    updated_chats_list = [f for f in os.listdir(shared_state.chats_folder) if f.endswith('json')]
    return (
        chat_history_state, chat_history_state,
        gr.Dropdown(choices=updated_chats_list, value=updated_chats_list[0]),
        gr.Textbox('')
    )
        
class SettingsTab:
    def __init__(self, shared_state):
        self.chat_model_name_settings = gr.Dropdown(
            choices=[i['model'] for i in ollama.list()['models']],
            value=shared_state.value.model_name, 
            label='Select Ollama Model', 
            info='WARNING: Do not select an embedding Model here.',
            interactive=True 
        )
        self.system_prompt_settings = gr.Textbox(
                value=shared_state.value.system_prompt,
                label='LLM System Prompt',
                interactive=True, 
            )
        self.chats_folder_settings = gr.Textbox(
            value=shared_state.value.chats_folder,
            label='Chats Folder', 
            interactive=True,
        )
        self.update_settings_btn = gr.Button(value='Update Settings')
    
    def update_settings(
        self, 
        shared_state, 
        model_name,
        system_prompt,
        chats_folder
        ):
        current_state = shared_state
        current_state.model_name = model_name
        current_state.system_prompt = system_prompt
        shared_state.chats_folder = chats_folder
        print(f'Updated Settings: {current_state}')
        return current_state

class RagTab:
    def __init__(self) -> None:
        pass

class App:
    def __init__(self, shared_state):
        self.shared_state = shared_state
        self.chat_history_state = gr.State([])
        self.interface()

        
        self.events()
    
    def interface(self):
        with gr.Tab('Chat'):
            self.chat_tab = ChatTab(self.shared_state)
        with gr.Tab('Settings'):
            self.settings_tab = SettingsTab(self.shared_state)
    
    def events(self):
        self.chat_tab.msg.submit(
            fn=self.chat_tab.get_ollama_response,
            inputs=[self.chat_tab.msg, self.chat_history_state, self.shared_state],
            outputs=[self.chat_tab.msg, self.chat_history_state, self.chat_tab.chatbot]
        )
        self.chat_tab.load_chat_btn.click(
            fn=load_chat_file, 
            inputs=[self.shared_state, self.chat_tab.chat_files_list], 
            outputs=[self.chat_history_state, self.chat_tab.chatbot, self.shared_state, self.chat_tab.chat_file_name]
        )
        self.chat_tab.save_chat_btn.click(
            fn=save_chat_file,
            inputs=[self.chat_history_state, self.shared_state, self.chat_tab.chat_file_name],
            outputs=[self.chat_tab.chat_files_list, self.chat_tab.chat_file_name]
        )
        self.chat_tab.delete_chat_btn.click(
            fn=delete_chat_file,
            inputs=[self.chat_history_state, self.shared_state, self.chat_tab.chat_file_name],
            outputs=[self.chat_history_state, self.chat_tab.chatbot,
                     self.chat_tab.chat_files_list, self.chat_tab.chat_file_name]
        )
        self.chat_tab.regen_btn.click(
            fn=self.chat_tab.regenerate_message,
            inputs=[self.chat_history_state, self.shared_state],
            outputs=[self.chat_tab.msg, self.chat_history_state, self.chat_tab.chatbot]
        )
        self.chat_tab.undo_btn.click(
            fn=self.chat_tab.undo_user_message,
            inputs=[self.chat_history_state],
            outputs=[self.chat_history_state, self.chat_tab.chatbot]
        )
        self.chat_tab.clear_btn.click(
            fn=lambda : ([], []),
            outputs=[self.chat_history_state, self.chat_tab.chatbot]
        )
        
        self.settings_tab.update_settings_btn.click(
            fn=self.settings_tab.update_settings,
            inputs=[self.shared_state, self.settings_tab.chat_model_name_settings, 
                    self.settings_tab.system_prompt_settings, self.settings_tab.chats_folder_settings],
            outputs=[self.shared_state]
        )

with gr.Blocks() as demo:
    app_shared_state = gr.State(SharedState())
    App(app_shared_state)
    
demo.launch()   
