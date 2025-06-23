# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Preprocess the FinQA dataset for search-based multi-turn training."""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


def make_prefix(dp: dict, template_type: str) -> str:
    question = dp["question"]
    if template_type == "base":
        prefix = (
            "Answer the given question. "
            "You must conduct reasoning inside <think> and </think> first every time you get new information. "
            "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
            "and it will return the top searched results between <information> and </information>. "
            "You can search as many times as your want. "
            "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
            "without detailed illustrations. For example, <answer> Beijing </answer>. Question: "
            f"{question}\n"
        )
    else:
        raise NotImplementedError
    return prefix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/finqa_search")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--template_type", type=str, default="base")
    args = parser.parse_args()

    data_source = "finqa"

    dataset = datasets.load_dataset("llk010502/FinQA_Combined_dataset")

    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    def make_map_fn(split: str):
        def process_fn(example, idx):
            example["question"] = example["question"].strip()
            if example["question"] and example["question"][-1] != "?":
                example["question"] += "?"
            question = make_prefix(example, template_type=args.template_type)
            solution = {"target": example["answer"]}
            support = example.get("information", example.get("support", ""))
            tools_kwargs = {"search": {"create_kwargs": {"document": support}}}
            extra_info = {
                "split": split,
                "index": idx,
                "need_tools_kwargs": True,
                "tools_kwargs": tools_kwargs,
            }
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "fact-reasoning",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "support": support,
                "extra_info": extra_info,
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    validation_dataset = validation_dataset.map(function=make_map_fn("validation"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    validation_dataset.to_parquet(os.path.join(local_dir, "validation.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
