from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import os
import base64
import gradio as gr

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
        
        print(f"File '{encrypted_filepath}' decrypted to '{output_filepath}'")

    def encrypt_files(self, input_files, output_dir, password, progress=gr.Progress()):
        os.makedirs(output_dir, exist_ok=True)
        self.temp_filenames = []
        for _, file_path in enumerate(progress.tqdm(input_files, desc="Encrypting Files")):
            filename = os.path.basename(file_path)

            # Skip already encrypted files based on extension
            if filename.endswith(".enc"):
                print(f"Skipping already encrypted file: '{input_files}'")
                continue
            
            output_filepath = os.path.join(output_dir, filename + '.enc')

            self.encrypt_file_with_password(file_path, password, output_filepath)
            self.temp_filenames.append(filename + '.enc')
           
    
    def decrypt_folders(self, input_files, output_folder, password, progress=gr.Progress()):
        os.makedirs(output_folder, exist_ok=True)
        self.temp_filenames = []
        for _, file_path in enumerate(progress.tqdm(input_files, desc="Encrypting Files")):
            filename = os.path.basename(file_path)

            # Only process files with .enc extension for decryption
            if not filename.endswith(".enc"):
                print(f"Skipping non-encrypted file: '{input_files}'")
                continue
            
            output_filepath = os.path.join(output_folder, filename[:-4])
            self.decrypt_file_with_password(file_path, password, output_filepath)
            self.temp_filenames.append(output_filepath)

class UI:
    def __init__(self):
        self.encrypter = PasswordEncrypter()
        self._interface()
        self._events()
        
    def _encrypt_decrypt(self, action_type, password, input_files, output_folder):
        print(f'\ninput files: {input_files}\noutput folder: {output_folder}')
        if action_type == 'Encrypt':
            self.encrypter.encrypt_files(input_files, output_folder, password)
            return 'Encryption complelte.\nReminder: Delete The original Files'
        elif action_type == 'Decrypt':
            self.encrypter.decrypt_folders(input_files, output_folder, password)
            return 'Decryption complelte.\nReminder: Delete the decrpyted Files after usage.'
        else:
            pass

    def _interface(self):
        gr.Markdown("# File Encryption/Decryption Tool")

        with gr.Column():
            gr.Markdown('### File/Folder Locations')
            with gr.Row():
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
            
            with gr.Row():
                self.action_type = gr.Dropdown(
                    choices=["Encrypt", "Decrypt"], 
                    label="Select Action", 
                    value="Encrypt", 
                    interactive=True
                    )
                
                self.password_input = gr.Textbox(
                    label="Encryption/Decryption Password", 
                    type="password", 
                    placeholder="Enter your secret password"
                    )

            self.process_btn = gr.Button(value='Encrypt/Decrypt', interactive=True)
                
            self.status_box = gr.Textbox(
                label='Status',
                interactive=False
            )
    
    def _events(self):
        self.process_btn.click(
            fn=self._encrypt_decrypt,
            inputs=[self.action_type, self.password_input, self.input_files, self.output_folder_path],
            outputs=[self.status_box]
        )
        
with gr.Blocks() as demo:
    UI()

demo.launch()
