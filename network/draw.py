from graphviz import Digraph


# Define the ResidualDenseBlock_out structure
def draw_residual_dense_block():
    dot = Digraph(comment='ResidualDenseBlock_out')

    # Define nodes
    dot.node('Input', 'Input (D x H x W)')
    dot.node('Conv1', 'Conv1\n32 filters\n3x3')
    dot.node('LReLU1', 'LeakyReLU')
    dot.node('Cat1', 'Concat\n(Input, Conv1)')

    dot.node('Conv2', 'Conv2\n32 filters\n3x3')
    dot.node('LReLU2', 'LeakyReLU')
    dot.node('Cat2', 'Concat\n(Input, Conv1, Conv2)')

    dot.node('Conv3', 'Conv3\n32 filters\n3x3')
    dot.node('LReLU3', 'LeakyReLU')
    dot.node('Cat3', 'Concat\n(Input, Conv1, Conv2, Conv3)')

    dot.node('Conv4', 'Conv4\n32 filters\n3x3')
    dot.node('LReLU4', 'LeakyReLU')
    dot.node('Cat4', 'Concat\n(Input, Conv1, Conv2, Conv3, Conv4)')

    dot.node('Conv5', 'Conv5\nOutput filters\n3x3')
    dot.node('Output', 'Output')

    # Define edges
    dot.edge('Input', 'Conv1')
    dot.edge('Conv1', 'LReLU1')
    dot.edge('LReLU1', 'Cat1')
    dot.edge('Cat1', 'Conv2')
    dot.edge('Conv2', 'LReLU2')
    dot.edge('LReLU2', 'Cat2')
    dot.edge('Cat2', 'Conv3')
    dot.edge('Conv3', 'LReLU3')
    dot.edge('LReLU3', 'Cat3')
    dot.edge('Cat3', 'Conv4')
    dot.edge('Conv4', 'LReLU4')
    dot.edge('LReLU4', 'Cat4')
    dot.edge('Cat4', 'Conv5')
    dot.edge('Conv5', 'Output')

    return dot


# Draw and render the graph
dot = draw_residual_dense_block()
dot.render('ResidualDenseBlock_out_graphviz', format='png')
