import compilation_statistics

class ComparisonCache(object):
    def __init__(self, type):
        self.type = type
        self.initialized = False
        self.lookup = {}

    def dump_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write(self.type + '\n')
            results = []
            for entry in self.lookup:
                results.append(str(entry) + ' -> ' + ','.join([str(x) for x in self.lookup[entry]]))
            f.write('\n'.join(results))

    def add_compiles_to(self, hash_from, hash_to):
        if hash_from in self.lookup:
            # Don't want to just pile everything up in the cross-comparison cache
            if not hash_to in self.lookup[hash_from]:
                self.lookup[hash_from].append(hash_to)
        else:
            self.lookup[hash_from] = [hash_to]

    def compilesto(self, hash_from, hash_to):
        assert self.initialized
        if hash_from in self.lookup and hash_to in self.lookup[hash_from]:
            return True
        else:
            return False

    def from_file(self, filename):
        self.lookup = {}
        self.initialized = True
        with open(filename, 'r') as f:
            first_line = True
            for line in f.readlines():
                if first_line:
                    first_line = False
                    self.type = line.split(',')[0]
                else:
                    # Otherwise, each line has a <structure hash> -> <structure hash> that indicates a successful comparison.
                    from_id, to_id = line.split(' -> ')

                    self.lookup[int(from_id)] = [int(x) for x in to_id.split(',')]
