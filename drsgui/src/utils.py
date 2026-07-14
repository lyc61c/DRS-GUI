"""Small shared helpers."""

import math


def get_chunk(items, number_of_chunks, chunk_index):
    if number_of_chunks < 1:
        raise ValueError("number_of_chunks must be positive")
    chunk_size = max(1, math.ceil(len(items) / number_of_chunks))
    chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
    return chunks[chunk_index] if chunk_index < len(chunks) else []
