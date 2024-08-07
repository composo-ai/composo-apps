import composo as cp


@cp.Composo.link(development_mode=True)
def hello_world(
    name: cp.StrParam,
):
    return f"Hello, {name}!"


hello_world("Seb")



# \begin{minipage}{\linewidth}
# \begin{lstlisting}
# import composo as cp

# @cp.Composo.link(development_mode=True)
# def hello_world(
#     name: cp.StrParam,
# ):
#     return f"Hello, {name}!"

# hello_world("Seb")
# \end{lstlisting}
# \end{minipage}
