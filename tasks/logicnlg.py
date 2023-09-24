# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The WikiTableQuestions dataset is a large-scale dataset for the task of question answering on semi-structured tables."""

import os

import datasets

import json
import pandas as pd

_HOMEPAGE = "https://finqasite.github.io/index.html"

_GIT_ARCHIVE_URL = (
    "https://github.com/wenhuchen/LogicNLG/archive/refs/heads/master.zip"
)

class LogicNLG(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = datasets.Features(
            {
                # "filename": datasets.Value("string"),
                "table": datasets.Value("string"),
                "sentences": datasets.features.Sequence(datasets.Value("string")),
            }
        )
        return datasets.DatasetInfo(
            features=features,
        )

    def _split_generators(self, dl_manager):
        extracted_path = dl_manager.download_and_extract(_GIT_ARCHIVE_URL)
        print(extracted_path)
        train_file = os.path.join(extracted_path, "LogicNLG-master/data/train_lm.json")
        dev_file = os.path.join(extracted_path, "LogicNLG-master/data/val_lm.json")
        test_file = os.path.join(extracted_path, "LogicNLG-master/data/test_lm.json")
        tables_path = dl_manager.extract(os.path.join(extracted_path, "LogicNLG-master/all_csv.zip"))

        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"main_filepath": train_file, "tables_path": os.path.join(tables_path, "data/all_csv")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"main_filepath": dev_file, "tables_path": os.path.join(tables_path, "data/all_csv")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"main_filepath": test_file, "tables_path": os.path.join(tables_path, "data/all_csv")},
            ),
        ]


    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, main_filepath, tables_path):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(main_filepath, encoding="utf-8") as f:
            # skip the first line since it is the tsv header
            lines = json.load(f)
            for k, example in lines.items():
                # get the table csv
                table_path = os.path.join(tables_path, k)
                df = pd.read_csv(table_path, sep = "#")
                # convert the df to a dict
                table = df.to_dict(orient='list')
                # some values could be nans, so we will convert all row items into strings
                table = {k: [str(each) for each in v] for k, v in table.items()}
                sentences = [each[0] for each in example]
                yield k, {
                "table": str(table),
                "sentences": sentences,
            }