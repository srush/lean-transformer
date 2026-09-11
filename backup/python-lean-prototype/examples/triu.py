def triu(n: int):
    rows = arange(n).unsqueeze()
    columns = arange(n)
    return where(rows <= columns, 1, 0)

