import composo as cp


@cp.Composo.link()
def hello_world(
    name: cp.StrParam,
):
    return f"Hello, {name}!"


hello_world("Armin")
