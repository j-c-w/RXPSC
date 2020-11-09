from jinja2 import Environment, FileSystemLoader
import automata.FST.sjss as sjss

# Given an automata, return a string of a simluator for it.
def generate_simulator_for(simple_automata, translator):
    env = Environment(loader=FileSystemLoader('automata/FST/generator/templates'))
    template = env.get_template('python_simulator.template')

    content = template.render(
            group_numbers=[0],
            graphs=[simple_automata],
            translators=[translator]
            )

    return content


# Return a list of simulators for the entire CC group.
def get_simulators_for(cc_group):
    results = []
    # First, get the physical one:
    underlying = generate_simulator_for(sjss.automata_to_nodes_and_edges(cc_group.physical_automata.component), None)
    results.append(underlying)

    # Then, get all the ones that translate to this:
    for translator in cc_group.translators:
        results.append(generate_simulator_for(sjss.automata_to_nodes_and_edges(cc_group.physical_automata.component), translator))

    return results


def write_simulators_for(cc_group, filename, selected_indexes=None):
    strings = get_simulators_for(cc_group)
    if selected_indexes is not None:
        for i in range(len(strings) - 1, -1, -1):
            if i not in selected_indexes:
                del strings[i]

    for i in range(len(strings)):
        with open(filename + "_" + str(i) + ".py", 'w') as f:
            f.write(strings[i])
