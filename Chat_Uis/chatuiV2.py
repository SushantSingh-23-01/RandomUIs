import gradio as gr
import random
import json
import ollama

class ChatUI:
    def __init__(self):
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
    
    def _export_chat(self, chat_history, path):
        if not chat_history:
            return None  
        with open(path, "w", encoding='utf-8') as f:
            json.dump(chat_history, f, indent=4) 
            
    def _import_chat(self, file_obj):
        chat_history = []

        if file_obj is None: 
            print("No file uploaded for import.")
            return chat_history, chat_history
        
        with open(file_obj.name, "r", encoding='utf-8') as f:
            try:
                chat_history = json.load(f)
            except json.JSONDecodeError:
                print(f'Chat File Empty/Corrupted!')
        return chat_history, chat_history

    def _update_settings(self, system_prompt):
        print(f'System Prompt Set to: {system_prompt}')
        return system_prompt
    
    def _update_model(self, model_name):
        print(f'Model Set to: {model_name}')
        return model_name     
    
    def _interface(self):
        with gr.Tab(label='Chat Inference'):        
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

            
        with gr.Tab(label='Settings'):
            self.import_chat_file = gr.File(
                value='chat.json', 
                file_types=['.json'],
                label='Import Chat File',
                interactive=True,
                height=120
                )

            self.export_chat_dir = gr.Textbox(
                value=r'chat.json',
                label='Chat Export Path',
                placeholder=r'Ex: C:\User\save_dir\chat.json'
            )
            with gr.Row():
                self.import_button = gr.Button(value='Import Chat')
                self.export_button = gr.Button(value='Export Chat')
            
            self.system_prompt_input = gr.Textbox(
                value='Act as an assistant.',
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
        self.import_button.click(
            fn=self._import_chat,
            inputs=[self.import_chat_file],
            outputs=[self.chat_history_state, self.chatbot]
        )
        self.export_button.click(
            fn=self._export_chat,
            inputs=[self.chat_history_state, self.export_chat_dir],
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
