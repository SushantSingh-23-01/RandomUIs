import gradio as gr
import random
import json
import ollama
from datetime import datetime
import os

class ChatUI:
    def __init__(self):
        # Default Parameters
        self.default_chats_directory = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'chats'
        )
        os.makedirs(self.default_chats_directory, exist_ok=True)
        self.default_system_prompt = 'Act as an assistant.'
        
        # State Parameters
        self.system_prompt_state = gr.State('Act as an assistant')
        self.available_models = [i['model'] for i in ollama.list()['models']]
        default_model = self.available_models[0] if self.available_models else ''
        self.model_name_state = gr.State(default_model)
        self.chat_history_state = gr.State([])
        
        self._interface()
        self._events()   
    
    def response(self, message, chat_history, system_prompt, model_name):
        chat_history.append({"role": "user", "content": message}) 
        messages = [{'role':'system','content':system_prompt}] + chat_history
        bot_message = ollama.chat(model = model_name, messages=messages)['message']['content']    
        chat_history.append({"role": "assistant", "content": bot_message})
        return '', chat_history, chat_history
    
    def _undo_last_turn(self, chat_history):
        if len(chat_history) >= 2 and \
            chat_history[-2]['role'] == 'user' and \
            chat_history[-1]['role'] == 'assistant':
            for _ in range(2):
                chat_history.pop()
        return chat_history, chat_history
    
    def _regenerate_last_message(self, chat_history, system_prompt, model_name):
        if len(chat_history) >= 2 and \
            chat_history[-2]['role'] == 'user' and \
            chat_history[-1]['role'] == 'assistant':
            chat_history.pop()
            last_user_message = chat_history.pop()['content']
            _, chat_history, chat_history= self.response(last_user_message, chat_history, system_prompt, model_name)
        return '', chat_history, chat_history
    
    def _save_chat(self, chat_history, chat_files_dir, chat_name_input):
        if not chat_history:
            return gr.Textbox(), gr.Dropdown()
        filename_base = chat_name_input.strip()
        if not filename_base:
            now = datetime.now()
            time_str = now.strftime('%Y-%m-%d-%H-%M-%S')
            filename_base = f'Chat_{time_str}'
        
        if not filename_base.endswith('.json'):
            filename_base += '.json'
            
        filepath = os.path.join(chat_files_dir, filename_base)

        with open(filepath, "w", encoding='utf-8') as f:
            json.dump(chat_history, f, indent=4) 
        
        print(f'\nSaved Chat at: {filepath}')
        new_chat_files_list = [filename_base] + [f for f in os.listdir(chat_files_dir) if f.endswith('json')]
        return gr.Textbox(value=filename_base), gr.Dropdown(choices=new_chat_files_list, value=filename_base)
            
    def _load_chat(self, chat_files_dir, filename):
        chat_history = []
        filepath = os.path.join(chat_files_dir, filename)

        with open(filepath, "r", encoding='utf-8') as f:
            try:
                chat_history = json.load(f)
            except json.JSONDecodeError:
                print(f'Chat File Empty/Corrupted!')
        return chat_history, chat_history, gr.Textbox(value=filename)

    def _update_settings(self, system_prompt):
        print(f'System Prompt Set to: {system_prompt}')
        return system_prompt
    
    def _update_model(self, model_name):
        print(f'Model Set to: {model_name}')
        return model_name     
    
    def parse_json_files(self, folder_dir):
        filenames = []
        for f in os.listdir(folder_dir):
            if f.endswith("json"):
                filenames.append(f)
        return gr.Dropdown(choices=filenames)

    def _create_new_chat(self, new_filename, chat_files_dir):
        chat_history = []
        if not new_filename: 
            now = datetime.now()
            time_str = now.strftime('%Y-%m-%d-%H-%M-%S')
            new_filename =  f'Chat_{time_str}'
        files_list = [f for f in os.listdir(chat_files_dir) if f.endswith('json')]
        new_files_list = [new_filename] + files_list
        return chat_history, chat_history, gr.Dropdown(choices=new_files_list, value=new_filename)
    
    def _delete_chat(self, filename, chat_files_dir, chat_history):
        filepath = os.path.join(chat_files_dir, filename)
        os.remove(filepath)
        if chat_history:
            chat_history = []
            
        files_list = [f for f in os.listdir(chat_files_dir) if f.endswith('json')]
        return chat_history, chat_history, gr.Dropdown(choices=files_list), gr.Textbox(value=None)
        
    def _interface(self):
        with gr.Tab(label='Chat Inference'):   
            with gr.Accordion(label='Chat Logs', open=False):
                with gr.Row():   
                    with gr.Column():
                        self.new_chat_name =  gr.Textbox(
                            label='Name of New Chat',
                            interactive=True
                        )
                        self.chat_files_dropdown = gr.Dropdown(
                            choices=[f for f in os.listdir(r'chats') if f.endswith('json')],
                            label='Chat Files',
                            interactive=True
                        )  
                    with gr.Column():
                        self.new_chat_btn = gr.Button(value='New Chat')
                        self.load_chat_btn = gr.Button(value='Load Chat')
                        self.delete_chat_btn = gr.Button(value='Delete Chat')
                        
            self.chatbot = gr.Chatbot(
                show_label=False, 
                type="messages",
                height=500,
                )
            with gr.Row():
                self.msg = gr.Textbox(lines=1,scale=3, interactive=True, submit_btn=True)
                with gr.Column():
                    self.clear = gr.ClearButton([self.msg, self.chatbot, self.chat_history_state])
                    self.undo_btn = gr.Button(value='Undo')
                with gr.Column():
                    self.regen_btn = gr.Button(value='Regen. Last Response')
                    self.save_chat_btn = gr.Button('Save Chat')

            
        with gr.Tab(label='Settings'):
            with gr.Row():
                self.chat_files_dir = gr.Textbox(
                    value=self.default_chats_directory,
                    label='Chat Export Path',
                    placeholder=r'Enter absolute paths to be on safer side',
                    interactive=True
                )

                self.update_chat_dir_btn = gr.Button(value='Update Chat Directory')

            
            self.system_prompt_input = gr.Textbox(
                value=self.default_system_prompt,
                label='LLM System Prompt',
            )
            self.model_name_input = gr.Dropdown(
                choices=self.available_models,
                value=self.model_name_state.value, 
                label='Select Ollama Model', 
                interactive=True 
            )
            self.update_settings_btn = gr.Button(value='Update Settings')
            
    def _events(self):
        self.msg.submit(
            fn=self.response,
            inputs=[self.msg, self.chat_history_state, self.system_prompt_state, self.model_name_state],
            outputs= [self.msg, self.chat_history_state, self.chatbot],
        )
        self.undo_btn.click(
            fn = self._undo_last_turn,
            inputs=[self.chat_history_state],
            outputs=[self.chat_history_state, self.chatbot]
        )
        self.regen_btn.click(
            fn = self._regenerate_last_message,
            inputs=[self.chat_history_state, self.system_prompt_state, self.model_name_state],
            outputs= [self.msg, self.chat_history_state, self.chatbot],
        )
        self.save_chat_btn.click(
            fn=self._save_chat,
            inputs=[self.chat_history_state, self.chat_files_dir, self.new_chat_name],
            outputs=[self.new_chat_name, self.chat_files_dropdown]
        )
        self.load_chat_btn.click(
            fn=self._load_chat,
            inputs=[self.chat_files_dir, self.chat_files_dropdown],
            outputs=[self.chat_history_state, self.chatbot, self.new_chat_name]
        )
        self.new_chat_btn.click(
            fn=self._create_new_chat,
            inputs=[self.new_chat_name, self.chat_files_dir],
            outputs=[self.chat_history_state, self.chatbot, self.chat_files_dropdown]
        )
        self.delete_chat_btn.click(
            fn=self._delete_chat,
            inputs=[self.chat_files_dropdown, self.chat_files_dir,self.chat_history_state],
            outputs=[self.chat_history_state, self.chatbot, self.chat_files_dropdown, self.new_chat_name]
        )
        self.update_chat_dir_btn.click(
            fn=self.parse_json_files,
            inputs=self.chat_files_dir,
            outputs=self.chat_files_dropdown,
        )
        
        self.update_settings_btn.click(
            fn=self._update_settings,
            inputs=[self.system_prompt_input],
            outputs=[self.system_prompt_state]
        )
        
        self.model_name_input.change(
            fn=self._update_model,
            inputs=[self.model_name_input],
            outputs=[self.model_name_state]
        )

with gr.Blocks() as demo:
    ChatUI()

demo.launch()
