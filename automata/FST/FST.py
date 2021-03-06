# This is a counter to keep track of the state IDs
state_id = 0

class FST(object):
    def __init__(self):
        self.states = []
        self.start_state = None

    def add_state(state):
        self.states.append(state)

    def set_start_state(id):
        self.start_state = id

class FSTState(object):
    def __init__(self):
        global state_id
        state_id += 1
        self.id = state_id
        self.lookuptable = None

    # Set the lookup table: needs to be from
    # char -> char, next state ID
    def set_lookuptable(self, lookuptable):
        self.lookuptable = lookuptable

class SingleStateTranslator(object):
    def __init__(self, lookup, modifications, unifier=None, overapproximation_factor=0.0):
        self.lookup = lookup
        self.modification_count = len(modifications)
        self.modifications = modifications
        self.unifier = unifier
        self._overapproximation_factor = overapproximation_factor

    def __str__(self):
        return str(self.lookup)

    def has_structural_additions(self):
        return self.modification_count > 0

    def isempty(self):
        for entry in self.lookup:
            if self.lookup[entry] != entry:
                return False
        return True

    def __getitem__(self, idx):
        return self.lookup[idx]

    def apply(self, input):
        output = []
        for character in input:
            output.append(chr(self[ord(character)]))
        return ''.join(output)

    def overapproximation_factor(self):
        return self._overapproximation_factor

    def to_string(self):
        return "# Overapprox fact. " + str(self.overapproximation_factor()) + "\n" + "{" + ",".join(["" + str(x) + ": " + str(self[x]) + "" for x in range(0, 256)]) + "}"

class EmptySingleStateTranslator(SingleStateTranslator):
    def __init__(self):
        lookup = {}
        for i in range(0, 256):
            lookup[i] = i
        super(EmptySingleStateTranslator, self).__init__(lookup, [])


class SymbolReconfiguration(object):
    def __init__(self, lookup, modifications):
        self.lookup = lookup
        self.modifications = modifications

    def has_structural_additions(self):
        return len(self.modifications) > 0

    def overapproximation_factor(self):
        # TODO --- I expect some overapproximation here too(?)
        return 0.0

    def isempty(self):
        return False

# This is an empty unifier for statistics gathering under
# the assumption that our unifier is all-powerful.
class AllPowerfulUnifier(object):
    def __init__(self, modifications):
        self.modifications = modifications

    def has_structural_additions(self):
        return len(self.modifications) > 0

    def overapproximation_factor(self):
        return 0.0

    def isempty(self):
        return False
