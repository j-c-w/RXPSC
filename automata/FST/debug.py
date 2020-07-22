def to_graphviz(nodes, edges):
    result = "digraph {\n"
    for (src, dst) in edges:
        result += str(src) + "->" + str(dst) + "\n"

    result += "}"

    return result
