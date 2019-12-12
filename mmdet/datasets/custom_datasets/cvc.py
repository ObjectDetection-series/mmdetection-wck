from ..custom import CustomDataset
from ..registry import DATASETS


@DATASETS.register_module
class CvcDataset(CustomDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 img_prefix='',
                 proposal_file=None,
                 test_mode=False):
        super(CvcDataset, self).__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            img_prefix=img_prefix,
            proposal_file=proposal_file,
            test_mode=test_mode)
