from flexllmgen.encrypt_engine.encrypt import encrypt_tensor_aes_ctr, decrypt_tensor_aes_ctr, encrypt_tensor_cuda_aes_ctr, decrypt_tensor_cuda_aes_ctr
import torch

def encrypt_test():
    test_tensor = torch.randn(100, 100, dtype=torch.float32, device='cpu')
    encrypted_tensor = encrypt_tensor_aes_ctr(test_tensor)
    decrypted_tensor = decrypt_tensor_aes_ctr(encrypted_tensor)
    print("Encryption and decryption successful:", torch.allclose(test_tensor, decrypted_tensor))
    
if __name__ == "__main__":
    encrypt_test()
