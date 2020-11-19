import rxp_pass

class PrintAlgebraPass(rxp_pass.Pass):
    def __init__(self):
        super(PrintAlgebraPass, self).__init__("PrintAlgebraPass")

    def execute(self, groups, options):
        for i in range(len(groups)):
            for j in range(len(groups[i])):
                if groups[i][j] is not None:
                    print groups[i][j].algebra
