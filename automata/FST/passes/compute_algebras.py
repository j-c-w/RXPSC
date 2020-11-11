import rxp_pass
import automata.FST.single_compiler as sc
import automata.FST.sjss as sjss

algebra_cache = {}
class ComputeAlgebraPass(rxp_pass.Pass):
    def __init__(self):
        super(ComputeAlgebraPass, self).__init__("ComputeAlgebra")

    def execute(self, automata_components, options):
        groups = []
        group_index = 0
        for cc_list in automata_components:
            group = []
            equation_index = 0
            for cc in cc_list:
                if options.print_file_info:
                    print "Compiling equation from group ", group_index
                    print "Equation index", equation_index

                global algebra_cache
                graph_hash = sjss.hash_graph(cc.automata)
                if options.use_algebra_cache and graph_hash in algebra_cache:
                    depth_eqn = algebra_cache[graph_hash]

                    if depth_eqn is not None:
                        depth_eqn = depth_eqn.clone()
                else:
                    depth_eqn = sc.compute_depth_equation(cc.automata, options)
                    if depth_eqn is None:
                        algebra_cache[graph_hash] = None
                    else:
                        algebra_cache[graph_hash] = depth_eqn.clone()

                if not depth_eqn:
                    # Means that the graph was too big for the current
                    # setup.
                    continue

                edges_not_in_graph = False
                for edge in depth_eqn.all_edges():
                    if edge not in cc.automata.edges:
                        edges_not_in_graph = True
                        print "Edge", edge, "not in graph"
                if edges_not_in_graph:
                    print "Graph", cc.automata
                    print "Equation", depth_eqn
                    assert False

                if options.print_algebras:
                    print depth_eqn
                    print "Hash: ", depth_eqn.structural_hash()
                if options.algebra_size_threshold and depth_eqn.size() > options.algebra_size_threshold:
                    print "Omitting equation due to size"
                else:
                    cc.algebra = depth_eqn
                    group.append(cc)
                    equation_index += 1

            groups.append(group)
            group_index += 1

        for x in groups:
            for y in x:
                assert y.algebra is not None
        return groups
