import datetime
import logging
import os
import time
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from typing import Callable, ContextManager, cast

# rework ddp_master to use spmd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.nn.parallel import DistributedDataParallel as DDP

# import composer


# globals --------------

DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
g_device_type = DEVICE_TYPE


def setup(rank, world_size, use_cuda=True):
    logging.getLogger().setLevel(logging.DEBUG if rank == 0 else logging.CRITICAL)

    if use_cuda:
        print(f"init for rank {rank}")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # set device for nccl pg for collectives
    if use_cuda == "nccl":
        print(f"--> init device for rank {rank}")
        torch.cuda.set_device(rank)


def teardown(rank) -> None:

    # Wait for all ranks to reach here before starting shutdown.
    dist.barrier()
    dist.destroy_process_group()
    logging.info(f"shut down process group on rank {rank}")


def formatted_print(rank, name, val, rank_only=False):
    if rank_only and not rank == 0:
        return
    print(f"{rank} --> {name} = {val}")


class State:
    output = None


def rank_sync_wrapper(hook):
    """Wrapper to insert monitored_barrier if using adaptive gradient accumulation.

    If a subset of ranks OOM, this monitored barrier fails and the error is caught so training can
    continue. Otherwise, two ranks would enter different barriers, resulting in deadlock.
    """

    def rank_sync_wrapper_hook(hook_state, bucket):
        print("syncing for: ", dist.get_rank())
        print(
            "enter sync: ",
            bucket.index(),
            "gradients length is: ",
            len(bucket.gradients()),
            "parameters length is: ",
            len(bucket.parameters()),
        )
        try:
            # Only put barrier in front of first bucket
            if bucket.index() == 0:
                print("Enter barrier")
                dist.barrier(group=hook_state["group"])
                print("Exit barrier")
            # Raise error because barrier in first bucket failed to go to no-op
            elif hook_state["hook_error"]:
                raise RuntimeError("Timed out")
        except RuntimeError as e:
            # barrier was tripped
            if "Timed out" in str(e):
                if bucket.index() == 0:
                    hook_state["hook_error"] = True

                def raise_timeout_error(fut):
                    del fut
                    raise e

                # Use a no-op hook and return the same gradients already on the device. If we don't
                # do the reduction, PyTorch will raise an internal error on the next backward pass
                # as the previous reduction hasn't been completed. After completing the no-op
                # reduction, re-raise the timeout error.
                fut = torch.futures.Future()
                fut.set_result(bucket.buffer())
                return fut.then(raise_timeout_error)
            else:
                raise
        print("exiting sync")
        print(
            "bucket is: ",
            bucket.is_last(),
            "length of parameters is: ",
            len(bucket.parameters()),
        )
        print("length of gradients is: ", len(bucket.gradients()))

        return hook(hook_state["nested_state"], bucket)

    return rank_sync_wrapper_hook


@contextmanager
def get_sync_context(sync_context, model, is_final_microbatch):
    auto_sync_context = nullcontext
    no_sync_context = cast(Callable[[], ContextManager], model.no_sync)

    if sync_context == "SINGLE":
        print("Getting single sync context")
        context = auto_sync_context if is_final_microbatch else no_sync_context
        with context():
            yield
    elif sync_context == "MULTI":
        with auto_sync_context():
            yield


def work_main(rank, world_size):

    sync_strategy = "SINGLE"
    use_amp = True
    batch_size = 512
    seq_len = 128
    opt_grad_accum = 2

    # Generating random datasets, this shouldn't matter, just need some tensors
    input_ids = torch.randint(0, 10000, size=(batch_size, seq_len), dtype=torch.long)
    token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    mask_idxs = torch.randint(0, seq_len, size=(int(seq_len * 0.25),))
    attention_mask[:, mask_idxs] = 0
    labels = -100 * torch.ones((batch_size, 128), dtype=torch.long)
    # Making random labels, this is non-sensical but it shouldn't matter
    labels[torch.where(attention_mask == 1)] = input_ids[
        torch.where(attention_mask == 1)
    ]

    # Setting up the model and distributed stuff
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)
    assert torch.cuda.current_device() == dist.get_rank()

    config = transformers.AutoConfig.from_pretrained("bert-base-uncased")
    model = transformers.AutoModelForMaskedLM.from_config(config)

    model = model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        find_unused_parameters=False,
    )

    dist_callback_obj = {"nested_state": None, "group": None}
    model.register_comm_hook(dist_callback_obj, rank_sync_wrapper(allreduce_hook))

    optimizer = torch.optim.Adam(model.parameters())

    state = State()

    curr_grad_accum = 1
    curr_batch_size = batch_size // curr_grad_accum
    done = False
    while curr_batch_size >= 1 and not done:
        dist_callback_obj["group"] = torch.distributed.new_group(
            backend="gloo", timeout=datetime.timedelta(seconds=5)
        )
        dist_callback_obj["hook_error"] = False

        found_cuda_oom = 0

        dist.barrier()

        # dist.barrier()

        state.output = None

        try:
            for i in range(curr_grad_accum):
                is_final_microbatch = i == (curr_grad_accum - 1)
                print("is final microbatch: ", is_final_microbatch)

                # Putting things onto device
                curr_input_ids = input_ids[
                    i * curr_batch_size : (i + 1) * curr_batch_size
                ].to(device)
                curr_token_type_ids = token_type_ids[
                    i * curr_batch_size : (i + 1) * curr_batch_size
                ].to(device)
                curr_attention_mask = attention_mask[
                    i * curr_batch_size : (i + 1) * curr_batch_size
                ].to(device)
                curr_labels = labels[
                    i * curr_batch_size : (i + 1) * curr_batch_size
                ].to(device)

                print(f"before model forward: {dist.get_rank()}")
                sync_context = get_sync_context(
                    sync_strategy, model, is_final_microbatch
                )
                with sync_context, torch.cuda.amp.autocast(use_amp):
                    state.output = model(
                        input_ids=curr_input_ids,
                        token_type_ids=curr_token_type_ids,
                        attention_mask=curr_attention_mask,
                        labels=curr_labels,
                    )

                    print(f"after model forward {dist.get_rank()}")

                    print(f"before model backward {dist.get_rank()}")

                    state.loss = state.output.loss
                    state.loss.backward()
                print(f"after model backward {dist.get_rank()}")

        except RuntimeError as e:
            print("error is: ", e)
            if "Timed out" in str(e):
                print(
                    f"rank: {dist.get_rank()} timed out, another rank likely out of memoried"
                )
                found_cuda_oom = 1
            elif "CUDA out of memory" in str(e):
                print(f"rank: {dist.get_rank()} out of memoried")
                found_cuda_oom = 1
            else:
                raise

        print("waiting for found cuda oom")

        found_cuda_oom = torch.tensor([found_cuda_oom], dtype=torch.uint8).to(device)
        dist.all_reduce(found_cuda_oom, op=torch.distributed.ReduceOp.MAX)

        print("found cuda oom is: ", found_cuda_oom.item())

        # Do grad accum here
        if found_cuda_oom.item() >= 1:
            print(f"trying to delete output {dist.get_rank()}")

            if state.output is not None:
                print(f"deleting outputs for: {dist.get_rank()}")
                del state.loss
                del state.output
                del state
                state = State()
            else:
                print("state output is None.")

            torch.cuda.empty_cache()
            print("cleared cache", dist.get_rank())

            curr_grad_accum *= 2
            curr_batch_size = batch_size // curr_grad_accum

            print(
                f"rank {dist.get_rank()} incrementing grad accum to: {curr_grad_accum} batch size now is: {curr_batch_size}"
            )
            continue

        done = True

    print("done is: ", done)

    optimizer.step()
    print("finished optimization")


# --------- main above -------------------------


def main(rank, world_size, use_cuda=True):

    # init
    setup(rank, world_size, use_cuda)

    _world_size = dist.get_world_size()
    logging.info(f"--> World size = {_world_size}")

    # main work
    work_main(rank, world_size)

    # teardown
    teardown(rank)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29505"
    world_size = 4
    use_cuda = DEVICE_TYPE == "cuda"
    print(f"use_cuda == {use_cuda}")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
