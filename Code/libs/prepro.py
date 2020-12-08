from feature_engineering import make_func


class Prepro:
    def __init__(self, param):
        self.param = param

    def __call__(self, module):
        piplines = []
        for func_name in self.param["piplines"]:
            piplines.append((make_func(func_name), self.param[func_name]))

        for func, param in piplines:
            module = func(module, param)


        print("sccessfully comlete prepro")
        return module



