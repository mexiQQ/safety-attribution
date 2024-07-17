import torch
import numpy as np
import matplotlib.pyplot as plt

# Load the matrices
matrix1 = torch.load('/home/jli265/workspace/alignment-attribution-code/out2/llama2-7b-chat-hf/structured/fluctuation/alpaca_cleaned_no_safety/attribution/atten_attr.pt')
matrix2 = torch.load('/home/jli265/workspace/alignment-attribution-code/out2/llama2-7b-chat-hf/structured/fluctuation/alpaca_cleaned_no_safety/attribution/mlp_attr.pt')
matrix3 = torch.load('/home/jli265/workspace/alignment-attribution-code/out2/llama2-7b-chat-hf/structured/fluctuation_set_difference/align/attribution/atten_attr.pt')
matrix4 = torch.load('/home/jli265/workspace/alignment-attribution-code/out2/llama2-7b-chat-hf/structured/fluctuation_set_difference/align/attribution/mlp_attr.pt')

def calculate_role_transfer(matrix1, matrix2, matrix3, matrix4):
    # Flatten the matrices for easy comparison
    mat1_flat = matrix1.flatten()
    mat2_flat = matrix2.flatten()
    mat3_flat = matrix3.flatten()
    mat4_flat = matrix4.flatten()

    # Calculate the total number of elements
    total_elements_att = mat1_flat.numel()
    total_elements_mlp = mat2_flat.numel()

    # Initialize role transfer arrays
    att_transfer = torch.zeros((2, 4), dtype=torch.float32)  # 2 original roles (0, 1) and 4 new roles (0, 1, 2, 3)
    mlp_transfer = torch.zeros((2, 4), dtype=torch.float32)

    # Calculate the role transfer percentage for attention module
    for i in range(2):
        for j in range(4):
            att_transfer[i, j] = torch.sum((mat1_flat == i) & (mat3_flat == j)).float() / total_elements_att

    # Calculate the role transfer percentage for MLP module
    for i in range(2):
        for j in range(4):
            mlp_transfer[i, j] = torch.sum((mat2_flat == i) & (mat4_flat == j)).float() / total_elements_mlp

    # Calculate block level transfer
    block_transfer_att = torch.zeros((matrix1.shape[0], 2, 4), dtype=torch.float32)
    for b in range(matrix1.shape[0]):
        block_flat1 = matrix1[b].flatten()
        block_flat3 = matrix3[b].flatten()
        for i in range(2):
            for j in range(4):
                block_transfer_att[b, i, j] = torch.sum((block_flat1 == i) & (block_flat3 == j)).float() / block_flat1.numel()

    block_transfer_mlp = torch.zeros((matrix2.shape[0], 2, 4), dtype=torch.float32)
    for b in range(matrix2.shape[0]):
        block_flat2 = matrix2[b].flatten()
        block_flat4 = matrix4[b].flatten()
        for i in range(2):
            for j in range(4):
                block_transfer_mlp[b, i, j] = torch.sum((block_flat2 == i) & (block_flat4 == j)).float() / block_flat2.numel()

    # Calculate global level transfer rates
    global_transfer = torch.zeros((2, 4), dtype=torch.float32)
    for i in range(2):
        for j in range(4):
            global_transfer[i, j] = (torch.sum((mat1_flat == i) & (mat3_flat == j)) + torch.sum((mat2_flat == i) & (mat4_flat == j))).float() / (total_elements_att + total_elements_mlp)

    return att_transfer, mlp_transfer, block_transfer_att, block_transfer_mlp, global_transfer

att_transfer, mlp_transfer, block_transfer_att, block_transfer_mlp, global_transfer = calculate_role_transfer(matrix1, matrix2, matrix3, matrix4)

print("Attention Module Role Transfer Percentages (global):\n", att_transfer.numpy())
print("MLP Module Role Transfer Percentages (global):\n", mlp_transfer.numpy())
print("Attention Module Role Transfer Percentages (block level):\n", block_transfer_att.numpy())
print("MLP Module Role Transfer Percentages (block level):\n", block_transfer_mlp.numpy())
print("Global Role Transfer Rates:\n", global_transfer.numpy())

