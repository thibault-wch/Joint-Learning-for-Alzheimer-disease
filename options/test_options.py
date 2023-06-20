from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--load_path', type=str, default=1,
                            help='the load path of the netG')

        self.isTrain = False
        return parser
