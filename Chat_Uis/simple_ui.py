import gradio as gr
import ollama

class ChatUI:
    def __init__(self):
        self.model = r'hf.co/unsloth/gemma-3-4b-it-GGUF:Q4_K_M'
        self.system_prompt = 'Act as a friend of user.'
        self.chat_history = gr.State([{'role':'system','content':self.system_prompt}])
        self._interface()
        self._events()   
    
    def response(self, message, chat_history):
        chat_history.append({"role": "user", "content": message})
        bot_message = ollama.chat(model = self.model, messages=chat_history)['message']['content']
        chat_history.append({"role": "assistant", "content": bot_message})
        return '', chat_history, chat_history
    
    def _undo_last_turn(self, chat_history):
        if len(chat_history) >= 2 and \
            chat_history[-2]['role'] == 'user' and \
            chat_history[-1]['role'] == 'assistant':
            for _ in range(2):
                chat_history.pop()
        return chat_history, chat_history
    
    def _interface(self):
        self.chatbot = gr.Chatbot(
            show_label=False, 
            type="messages",
            height=500,
            )
        with gr.Row():
            self.msg = gr.Textbox(lines=1,scale=3, interactive=True, submit_btn=True, stop_btn=True)
            with gr.Column():
                self.clear = gr.ClearButton([self.msg, self.chatbot, self.chat_history])
                self.undo_btn = gr.Button(value='undo')
            
    def _events(self):
        self.msg.submit(
            fn=self.response,
            inputs=[self.msg, self.chat_history],
            outputs= [self.msg, self.chat_history, self.chatbot],
        )
        self.undo_btn.click(
            fn = self._undo_last_turn,
            inputs=[self.chat_history],
            outputs=[self.chat_history, self.chatbot]
        )

with gr.Blocks() as demo:
    ChatUI()

demo.launch()
