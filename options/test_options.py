from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--dataset_mode', type=str, default='haadf', help='Dataset mode: aligned or haadf')
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=150, help='how many test images to run')
        self.parser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy', help='the path for clustered results of encoded features')
        self.parser.add_argument('--use_encoded_image', action='store_true', help='if specified, encode the real image to get the feature map')
        self.parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
        self.parser.add_argument("--engine", type=str, help="run serialized TRT engine")
        self.parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")
        self.parser.add_argument('--keep_ratio', type=float, default=0.7,
                            help='Minimum non-zero fraction in a patch; set 0 at test time to keep every tile')
        self.parser.add_argument('--pos_geom_aug_types', type=str, nargs='+', default=[], help='List of positive geometric augmentation types')
        self.parser.add_argument('--neg_geom_aug_types', type=str,
            default='ori,hflip,rot180', help='Geometric augmentations for negative patches')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--intensity_aug_prob', type=float, default=0.4, help='Probability of intensity augmentation for each patch')

        self.isTrain = False