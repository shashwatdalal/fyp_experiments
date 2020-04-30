from torchtext.datasets import LanguageModelingDataset


class LocalLanguageModelingDataset(LanguageModelingDataset):
    name = ''
    dirname = ''
    urls = []

    @classmethod
    def splits(cls, text_field, root='.data', train=None, test=None, **kwargs):
        return super(LocalLanguageModelingDataset, cls).splits(
            root=root, train=train, test=test,
            text_field=text_field, **kwargs)
