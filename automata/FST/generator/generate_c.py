from jinja2 import Environment, FileSystemLoader

def generate_wrapper():
    env = Environment(loader=FileSystemLoader('automata/FST/generator/templates'))
    template = env.get_template('c_simulator.template')
