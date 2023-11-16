import os
import torch
import torch.distributed as dist

"""
This script demonstrates the use of PyTorch's distributed computing features for balanced and 
random tensor splitting and all-to-all communication in a distributed setting. It aims to 
benchmark the performance of balanced versus random tensor operations.
"""

def run(rank, world_size):
    """
    Executes the distributed all-to-all communication test.

    Args:
    - rank (int): The rank of the current process in the distributed setting.
    - world_size (int): The total number of processes in the distributed setting.

    The function initializes tensors, performs balanced and random splits, and conducts
    all-to-all communication to benchmark performance.
    """
    m, n, k = 2048, 960, 7680
    input_tensor = torch.empty(m, k, device=torch.cuda.current_device(), dtype=torch.float16)
    coeffs = torch.rand([world_size])
    coeffs /= torch.sum(coeffs)
    split_sizes = [int(m * coeff) for coeff in coeffs]
    split_sizes[-1] += (m - sum(split_sizes))

    balance_list = list(torch.chunk(input_tensor, world_size))
    split_list = list(torch.split(input_tensor, split_sizes))

    input_split_size = torch.tensor(split_sizes, device=torch.cuda.current_device())
    output_split_size = torch.tensor(split_sizes, device=torch.cuda.current_device())
    print("Input split sizes:", split_sizes)
    dist.all_to_all_single(output_split_size, input_split_size)
    print("Output split sizes:", output_split_size.tolist())

    output_list = [torch.empty((size, k), device=torch.cuda.current_device(), dtype=torch.float16) for size in output_split_size]

    warm_up_and_profile(rank, balance_list, split_list, output_list)


def warm_up_and_profile(rank, balance_list, split_list, output_list):
    """
    Warms up and profiles the distributed operations.

    Args:
    - rank (int): The rank of the current process.
    - balance_list (list): List of tensors for balanced all-to-all communication.
    - split_list (list): List of tensors split based on random sizes.
    - output_list (list): List of output tensors for storing the results of all-to-all communication.
    """
    for _ in range(5):
        dist.all_to_all(output_list, split_list)  # Warm-up
        dist.all_to_all(balance_list, balance_list)  # Warm-up

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(5):
        profile_all_to_all("balance_test", balance_list, balance_list)
    for _ in range(5):
        profile_all_to_all("rand_test", output_list, split_list)
    torch.cuda.cudart().cudaProfilerStop()


def profile_all_to_all(label, output_list, input_list):
    """
    Profiles a single all-to-all operation.

    Args:
    - label (str): A label for the profiling range.
    - output_list (list): List of output tensors for storing the results of all-to-all communication.
    - input_list (list): List of input tensors for all-to-all communication.

    This function profiles an all-to-all communication operation between input_list and output_list,
    timing and labeling the operation for performance analysis.
    """
    torch.cuda.nvtx.range_push(label)
    dist.all_to_all(output_list, input_list)
    torch.cuda.nvtx.range_pop()


def init_process(backend='nccl'):
    """
    Initializes the distributed process group.
    Args:
    - backend (str): The backend to use for distributed processing. Default is 'nccl'.

    This function sets up the environment for distributed computing by initializing the process group
    and setting the appropriate CUDA device.
    """
    rank = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv("WORLD_SIZE", '1'))
    device_count = torch.cuda.device_count()

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank % device_count)

if __name__ == "__main__":
    init_process(backend='nccl')
    run(dist.get_rank(), dist.get_world_size())

# nsys profile -c cudaProfilerApi -f true -o prof_comm torchrun --master-addr 127.0.0.1 --master-port 16588 --nnodes 1 --nproc-per-node=8 all_to_all_unbalance_test.py


