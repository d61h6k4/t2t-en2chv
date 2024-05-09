import gzip
from datasets import (
    BuilderConfig,
    GeneratorBasedBuilder,
    DownloadManager,
    StreamingDownloadManager,
    Version,
    SplitGenerator,
    Split,
    DatasetInfo,
    Features,
    Value,
)
from typing import Union
from pathlib import Path


class TatoebaChallengeConfig(BuilderConfig):
    """Builder config for Tatoeba challenge dataset."""

    def __init__(self, name: str, version: str, **kwargs):
        assert version == "2023-09-26", "Only v2023-09-26 is supported"
        super().__init__(
            name=name, version=Version(version.replace("-", ".")), **kwargs
        )
        self.version_str = version
        self.data_url = (
            f"https://object.pouta.csc.fi/Tatoeba-Challenge-v{version}/{name}.tar"
        )


class TatoebaChallenge(GeneratorBasedBuilder):
    """Tatoeba challenge dataset."""

    BUILDER_CONFIG_CLASS = TatoebaChallengeConfig
    BUILDER_CONFIGS = [
        TatoebaChallengeConfig(name="chv-eng", version="2023-09-26"),
        TatoebaChallengeConfig(name="chv-rus", version="2023-09-26"),
    ]

    def _info(self) -> DatasetInfo:
        src, trg = self.config.name.split("-")
        return DatasetInfo(
            description="""
               The Tatoeba Translation Challenge.
               You can find more about the data here: https://github.com/Helsinki-NLP/Tatoeba-Challenge

                Here we have only Chuvash-English subset.
               We do not use official dataset https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt
               here because chv-eng of the dataset contains only test split.
               """,
            features=Features(
                {
                    "id": Value("string"),
                    src: Value("string"),
                    trg: Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/Helsinki-NLP/Tatoeba-Challenge",
            citation="""
           @inproceedings{tiedemann-2020-tatoeba,
               title = "The {T}atoeba {T}ranslation {C}hallenge {--} {R}ealistic Data Sets for Low Resource and Multilingual {MT}",
               author = {Tiedemann, J{\"o}rg},
               booktitle = "Proceedings of the Fifth Conference on Machine Translation",
               month = nov,
               year = "2020",
               address = "Online",
               publisher = "Association for Computational Linguistics",
               url = "https://www.aclweb.org/anthology/2020.wmt-1.139",
               pages = "1174--1182"
           }
           """,
        )

    def _split_generators(
        self, dl_manager: Union[DownloadManager, StreamingDownloadManager]
    ):
        dl_dir = dl_manager.download_and_extract(self.config.data_url)
        assert isinstance(dl_dir, str)

        path_to_folder = (
            Path(dl_dir)
            / "data"
            / "release"
            / f"v{self.config.version_str}"
            / self.config.name
        )
        return [
            SplitGenerator(
                name=Split.TRAIN._name,
                gen_kwargs={
                    "langs": path_to_folder / "train.id.gz",
                    "src": path_to_folder / "train.src.gz",
                    "trg": path_to_folder / "train.trg.gz",
                },
            ),
            SplitGenerator(
                name=Split.TEST._name,
                gen_kwargs={
                    "langs": path_to_folder / "test.id",
                    "src": path_to_folder / "test.src",
                    "trg": path_to_folder / "test.trg",
                },
            ),
        ]

    def _generate_examples(self, langs: Path, src: Path, trg: Path):
        if langs.suffix == ".gz":
            assert src.suffix == trg.suffix
            assert trg.suffix == langs.suffix

            opener = gzip.open
        else:
            opener = open

        with opener(langs, "rb") as langs_src:
            with opener(src, "rb") as src_src:
                with opener(trg, "rb") as trg_src:
                    for id_, (langs_line, src_line, trg_line) in enumerate(
                        zip(langs_src, src_src, trg_src)
                    ):
                        langs_row = langs_line.decode("utf8").strip().split("\t")
                        # train contains 3 symbols
                        if len(langs_row) == 3:
                            _, src_lang, trg_lang = langs_row
                        else:
                            src_lang, trg_lang = langs_row

                        yield (
                            id_,
                            {
                                "id": id_,
                                src_lang: src_line.decode("utf8").strip(),
                                trg_lang: trg_line.decode("utf8").strip(),
                            },
                        )
