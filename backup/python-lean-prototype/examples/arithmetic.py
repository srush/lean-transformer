def arithmetic(n: int, a: Tensor["n", "n"], b: Tensor["n", "n"]):
    mixed = (a + b) * (a - b) / 2
    return mixed @ b

