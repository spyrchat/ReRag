def batchify(items, batch_size):
    """
    Splits a list of items into smaller batches of a specified size.

    Args:
        items (list): The list of items to be split into batches.
        batch_size (int): The maximum number of items in each batch.
    """
    for i in range(0, len(items), batch_size):
        # Yield a slice of the list from index `i` to `i + batch_size`
        yield items[i:i + batch_size]
