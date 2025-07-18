import gradio as gr
import json
import ollama
from datetime import datetime
import os

class ChatUI:
    def __init__(self):
        self.available_models = [i['model'] for i in ollama.list()['models']]
        self.default_model = self.available_models[0] if self.available_models else ''
        
        self.default_system_prompt = 'Act as an assistant.'

        self.default_chats_directory: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'chats')
        os.makedirs(self.default_chats_directory, exist_ok=True)
        
        self.chat_history_state = gr.State([])
        
        self._interface()
        self._events()   

    def response(
        self, 
        message, 
        chat_history, 
        system_prompt, 
        model_name,
        temperature, 
        top_k, 
        top_p
        ):
        try:
            chat_history.append({"role": "user", "content": message}) 
            messages = [{'role':'system','content':system_prompt}] + chat_history
            bot_message = ollama.chat(
                model = model_name, 
                messages=messages,
                options={
                    'temperature': temperature,
                    'top_k': top_k,
                    'top_p': top_p
                }
                )['message']['content']    
            chat_history.append({"role": "assistant", "content": bot_message})
        except ollama.ResponseError as e:
                    print(f"Ollama Error: {e}")
        return '', chat_history, chat_history
    
    def _undo_last_turn(self, chat_history):
        if len(chat_history) >= 2 and \
            chat_history[-2]['role'] == 'user' and \
            chat_history[-1]['role'] == 'assistant':
            for _ in range(2):
                chat_history.pop()
        return chat_history, chat_history
    
    def _regenerate_last_message(self, chat_history, system_prompt, model_name, temperature, top_k, top_p):
        if len(chat_history) >= 2 and \
            chat_history[-2]['role'] == 'user' and \
            chat_history[-1]['role'] == 'assistant':
            chat_history.pop()
            last_user_message = chat_history.pop()['content']
            _, chat_history, chat_history= self.response(last_user_message, chat_history, system_prompt, model_name, temperature, top_k, top_p)
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

        if not filename:
            print("No chat file selected to load.")
            return [], [], gr.Textbox(value="")
        
        filepath = os.path.join(chat_files_dir, filename)
        with open(filepath, "r", encoding='utf-8') as f:
            try:
                chat_history = json.load(f)
            except json.JSONDecodeError:
                print(f'Chat File Empty/Corrupted!')
        return chat_history, chat_history, gr.Textbox(value=filename)
    
    def _parse_chat_folder(self, chat_files_dir):
        json_file_paths = []
        for filename in os.listdir(chat_files_dir):
            if filename.endswith(".json"):
                full_path = os.path.join(chat_files_dir, filename)
                json_file_paths.append(full_path)

        json_file_paths.sort(key=os.path.getmtime, reverse=True)

        filenames = [os.path.basename(path) for path in json_file_paths]
        return filenames

    def _create_new_chat(self, chat_files_dir, filename):
        chat_history = []
        if not filename: 
            now = datetime.now()
            time_str = now.strftime('%Y-%m-%d-%H-%M-%S')
            filename =  f'Chat_{time_str}'
        updated_files_list = self._parse_chat_folder(chat_files_dir)
        updated_files_list = [filename] + updated_files_list
        return chat_history, chat_history, gr.Dropdown(choices=updated_files_list, value=filename), gr.Textbox(value=filename)
    
    def _delete_chat(self, chat_files_dir, filename, chat_history):
        filepath = os.path.join(chat_files_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        if chat_history:
            chat_history = []
            
        updated_files_list = self._parse_chat_folder(chat_files_dir)
        return chat_history, chat_history, gr.Dropdown(choices=updated_files_list, value=updated_files_list[0]), gr.Textbox(value=None)
        
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
                            choices=self._parse_chat_folder(self.default_chats_directory),
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
                self.chat_files_dir_input = gr.Textbox(
                    value=self.default_chats_directory,
                    label='Chat Export Path',
                    placeholder=r'Enter absolute paths to be on safer side',
                    interactive=True, 
                    submit_btn=True
                )

            self.system_prompt_input = gr.Textbox(
                value=self.default_system_prompt,
                label='LLM System Prompt',
                interactive=True, 
                submit_btn=True
            )
            
            self.model_name_input = gr.Dropdown(
                choices=self.available_models,
                value=self.default_model, 
                label='Select Ollama Model', 
                interactive=True 
            )
            with gr.Row():
                self.temperature_input = gr.Slider(
                    minimum=0,
                    maximum=1.0,
                    step=0.1,
                    value=0.9,
                    label='Temperature',
                    interactive=True
                )
                self.top_k_input = gr.Slider(
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=40,
                    label='Top-K',
                    interactive=True
                )
                self.top_p_input = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.9,
                    label='Top-P',
                    interactive=True
                )
            
    def _events(self):
        self.msg.submit(
            fn=self.response,
            inputs=[self.msg, self.chat_history_state, self.system_prompt_input, self.model_name_input, 
                    self.temperature_input, self.top_k_input, self.top_p_input],
            outputs= [self.msg, self.chat_history_state, self.chatbot],
        )
        self.undo_btn.click(
            fn = self._undo_last_turn,
            inputs=[self.chat_history_state],
            outputs=[self.chat_history_state, self.chatbot]
        )
        self.regen_btn.click(
            fn = self._regenerate_last_message,
            inputs=[self.chat_history_state, self.system_prompt_input, self.model_name_input,
                    self.temperature_input, self.top_k_input, self.top_p_input],
            outputs= [self.msg, self.chat_history_state, self.chatbot],
        )
        self.save_chat_btn.click(
            fn=self._save_chat,
            inputs=[self.chat_history_state, self.chat_files_dir_input, self.new_chat_name],
            outputs=[self.new_chat_name, self.chat_files_dropdown]
        )
        self.load_chat_btn.click(
            fn=self._load_chat,
            inputs=[self.chat_files_dir_input, self.chat_files_dropdown],
            outputs=[self.chat_history_state, self.chatbot, self.new_chat_name]
        )
        self.new_chat_btn.click(
            fn=self._create_new_chat,
            inputs=[self.chat_files_dir_input, self.new_chat_name],
            outputs=[self.chat_history_state, self.chatbot, self.chat_files_dropdown, self.new_chat_name]
        )
        self.delete_chat_btn.click(
            fn=self._delete_chat,
            inputs=[self.chat_files_dir_input, self.chat_files_dropdown, self.chat_history_state],
            outputs=[self.chat_history_state, self.chatbot, self.chat_files_dropdown, self.new_chat_name]
        )
        self.chat_files_dir_input.submit(
            fn=lambda chat_files_dir: gr.Dropdown(choices=self._parse_chat_folder(chat_files_dir)),
            inputs=self.chat_files_dir_input,
            outputs=self.chat_files_dropdown,
        )
        
        self.chat_files_dir_input.submit(
            fn=lambda chat_diles_dir: print(f'Chat Files Directory set to: {chat_diles_dir}'),
            inputs=[self.chat_files_dir_input]
        )
        
        self.system_prompt_input.submit(
            fn=lambda system_prompt: print(f'System Prompt set to: {system_prompt}'),
            inputs=[self.system_prompt_input]
        )
        
        self.model_name_input.change(
            fn=lambda model_name: print(f'Model name set to: {model_name}'),
            inputs=[self.model_name_input],
        )

with gr.Blocks() as demo:
    ChatUI()

demo.launch()
