from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import os
import base64
import gradio as gr
import sys

class PasswordEncrypter:
    def derive_key(self, password: str, salt: bytes) -> bytes:
        """Derives a Fernet key from a password and salt."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000, # More iterations = more secure, but slower
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_file_with_password(self, input_filepath, password, output_filepath):
        """
        Encrypts a file using a password. The salt is prepended to the encrypted data.
        Returns the path to the encrypted file.
        """
        salt = os.urandom(16)  # Generate a unique salt for each encryption
        key = self.derive_key(password, salt)
        fernet = Fernet(key)
        with open(input_filepath, 'rb') as f:
            original_data = f.read()

        encrypted_data = fernet.encrypt(original_data)

        # Prepend the salt to the encrypted data
        final_data_to_write = salt + encrypted_data

        with open(output_filepath, 'wb') as f:
            f.write(final_data_to_write)
    
    def decrypt_file_with_password(self, encrypted_filepath, password, output_filepath):
        """
        Encrypts a file using a password. The salt is prepended to the encrypted data.
        Returns the path to the encrypted file.
        """
        if output_filepath is None:
            if encrypted_filepath.endswith(".enc"):
                output_filepath = encrypted_filepath[:-4]
            else:
                output_filepath = encrypted_filepath + ".dec" # Fallback if no .enc

        with open(encrypted_filepath, 'rb') as f:
            full_encrypted_data = f.read()
        
        # Extract the salt (first 16 bytes)
        salt = full_encrypted_data[:16]
        actual_encrypted_data = full_encrypted_data[16:]

        key = self.derive_key(password, salt)
        fernet = Fernet(key)

        decrypted_data = fernet.decrypt(actual_encrypted_data)

        with open(output_filepath, 'wb') as f:
            f.write(decrypted_data)

    def encrypt_files(self, input_files, output_dir, password, progress=gr.Progress()):
        os.makedirs(output_dir, exist_ok=True)
        temp_filenames = []
        for _, file_path in enumerate(progress.tqdm(input_files, desc="Encrypting Files")):
            filename = os.path.basename(file_path)

            # Skip already encrypted files based on extension
            if filename.endswith(".enc"):
                print(f"Skipping already encrypted file: '{input_files}'")
                continue
            
            output_filepath = os.path.join(output_dir, filename + '.enc')

            self.encrypt_file_with_password(file_path, password, output_filepath)
            temp_filenames.append(output_filepath)  
        return temp_filenames
           
    
    def decrypt_files(self, input_files, output_folder, password, progress=gr.Progress()):
        os.makedirs(output_folder, exist_ok=True)
        temp_filenames = []
        for _, file_path in enumerate(progress.tqdm(input_files, desc="Encrypting Files")):
            filename = os.path.basename(file_path)

            # Only process files with .enc extension for decryption
            if not filename.endswith(".enc"):
                print(f"Skipping non-encrypted file: '{input_files}'")
                continue
            
            output_filepath = os.path.join(output_folder, filename[:-4])
            self.decrypt_file_with_password(file_path, password, output_filepath)
            temp_filenames.append(output_filepath)
        return temp_filenames
    
class UI:
    def __init__(self):
        self.encrypter = PasswordEncrypter()
        self._interface()
        self._events()
        
    def _encrypt_decrypt(self, action_type, password, confirm_password, input_files, output_folder):
        if not input_files:
            return "Please select files to process.", ""
        if not password:
            return "Please enter a password.", ""
        if not output_folder:
            return "Please enter an output directory.", ""
        
        if action_type == 'Encrypt':
            if password == confirm_password:
                processed_files_list = self.encrypter.encrypt_files(input_files, output_folder, password)
                status_message = 'Encryption complete! Reminder: Delete the original files if you wish.'
            else:
                processed_files_list = None
                status_message = "Passwords do not match. Please Check again!"
                
        elif action_type == 'Decrypt':
            processed_files_list = self.encrypter.decrypt_files(input_files, output_folder, password)
            status_message = 'Decryption complete! Reminder: Delete the decrypted files after usage if sensitive.'
        else:
            processed_files_list = None
            status_message = "Invalid action type."
            
        # Format the list of paths into a single multiline string for the Textbox
        processed_files_text = ''
        if processed_files_list:
            for i in processed_files_list:
                processed_files_text += '\n' + str(i) + '\t\u2713'
        else:
            'No files processed or matched criteria.'
        return status_message, processed_files_text
    
    def _open_output_directory(self, output_folder_path):
        """Opens the specified folder in the system's file explorer."""
        if not output_folder_path or not os.path.isdir(output_folder_path):
            return "Error: Output directory not found or invalid."

        if sys.platform == "win32":
            os.startfile(output_folder_path)
        elif sys.platform == "darwin":
            subprocess.run(['open', output_folder_path], check=True)
        else:
            subprocess.run(['xdg-open', output_folder_path], check=True)
        return "Output folder opened successfully."
        
    def _interface(self):
        gr.Markdown("# File Encryption/Decryption Tool")


        gr.Markdown('### File/Folder Locations')
        with gr.Row():
            with gr.Column():
                self.input_files = gr.File(
                    file_count='multiple',
                    label='Input Files',
                    height=150
                    )
        

                self.output_folder_path = gr.Textbox(
                    label='Output Directory',
                    placeholder='Enter full Output Directory', 
                    interactive=True
                )
                self.action_type = gr.Dropdown(
                    choices=["Encrypt", "Decrypt"], 
                    label="Select Action", 
                    value="Encrypt", 
                    interactive=True
                    )
                    
                with gr.Row():
                    self.password_input = gr.Textbox(
                        label="Encryption/Decryption Password", 
                        type="password", 
                        placeholder="Enter your secret password",
                        interactive=True
                        )
                    self.confirm_password_input = gr.Textbox(
                        label="Confirm Encryption/Decryption Password", 
                        type="password", 
                        placeholder="Enter your secret password",
                        interactive=True
                        )
                    
                with gr.Row():
                    self.process_btn = gr.Button(value='Encrypt/Decrypt', interactive=True)
                    self.open_out_folder = gr.Button(value='Open Output Folder', interactive=True)
                    
            with gr.Column():
                self.status_box = gr.Textbox(label='Status', interactive=False)

                self.file_out_status =  gr.Textbox(
                    label="Processed File Paths", 
                    lines=10,             # Display 10 lines, then scroll
                    max_lines=None,       # No maximum content lines
                    interactive=False,    
                    visible=True           
                )
        
    def _events(self):
        self.process_btn.click(
            fn=self._encrypt_decrypt,
            inputs=[self.action_type, self.password_input, self.confirm_password_input, self.input_files, self.output_folder_path],
            outputs=[self.status_box, self.file_out_status]
        )
        self.open_out_folder.click(
            fn = self._open_output_directory,
            inputs=[self.output_folder_path],
            outputs=[self.status_box]
        )
        
with gr.Blocks() as demo:
    UI()

demo.launch()
