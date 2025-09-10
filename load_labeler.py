import os
import re
import sys
import json
import torch
import importlib.util
import numpy as np
from wrench.labelmodel import Snorkel, MajorityVoting
from wrench.labelmodel import Snorkel
from wrench.dataset.dataset import TextDataset, NumericDataset

class WeakSoftLabelGenerator:
    def __init__(self, dataset_path, lf_dir, num_classes=4, device=None):
        self.dataset_path = dataset_path
        self.lf_dir = lf_dir
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def load_json_dataset(self, split):
        file_path = os.path.join(self.dataset_path, f"{split}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        # convert sang format wrench-like
        examples = [{"text": d["data"]["text"]} for d in data]
        labels = [d.get("label", -1) for d in data]  # test có thể ko có label
        return examples, labels


    def sort_filenames(self, filename):
        return int(re.search(r'\d+', filename).group())

    def load_LFs(self):
        file_paths = []
        for f in os.listdir(self.lf_dir):
            if f.endswith(".py"):
                file_paths.append(os.path.join(self.lf_dir, f))
        file_paths = sorted(file_paths, key=self.sort_filenames)
        return file_paths

    def apply_LFs(self, examples):
        weak_label_matrix = []
        allowed_labels = set(range(self.num_classes))
        lf_files = self.load_LFs()

        for file_path in lf_files:
            print(f"Applying LF from {file_path}")
            module_spec = importlib.util.spec_from_loader("temp_module", loader=None)
            module = importlib.util.module_from_spec(module_spec)

            with open(file_path, "r", encoding="utf-8") as f:
                code_string = f.read()

            try:
                exec(code_string, module.__dict__)
                sys.modules["temp_module"] = module
                from temp_module import label_function

                weak_labels = []
                for ex in examples:
                    weak_label = label_function(ex["text"])
                    if weak_label not in allowed_labels:
                        weak_label = -1
                    weak_labels.append(weak_label)
                weak_label_matrix.append(weak_labels)

            except Exception as e:
                print(f"Error in {file_path}: {e}")

        weak_label_matrix = np.array(weak_label_matrix).T
        return weak_label_matrix

    def generate_labels(self):
        # Load datasets
        train_examples, train_labels = self.load_json_dataset("train")
        valid_examples, valid_labels = self.load_json_dataset("valid")
        test_examples, test_labels   = self.load_json_dataset("test")

        # Weak labels
        print("Generating weak labels...")
        train_weak = self.apply_LFs(train_examples)
        valid_weak = self.apply_LFs(valid_examples)
        test_weak  = self.apply_LFs(test_examples)

        
        train_data = TextDataset(
            split='train',
            examples=train_examples,
            weak_labels=train_weak,
            labels=train_labels
        )
        train_data.n_class = 4  # gán số lớp

        valid_data = TextDataset(
            split='valid',
            examples=valid_examples,
            weak_labels=valid_weak,
            labels=valid_labels
        )
        valid_data.n_class = 4  # gán số lớp
        test_data = TextDataset(
            split='test',
            examples=test_examples,
            weak_labels=test_weak,
            labels=test_labels
        )
        test_data.n_class = 4  # gán số lớp

        # Save weak labels
        with open("train_weak_labels.json", "w", encoding="utf-8") as f:
            json.dump([
                {"text": ex["text"], "weak_labels": wl.tolist()}
                for ex, wl in zip(train_examples, train_weak)
            ], f, ensure_ascii=False, indent=2)

        # Soft labels (dùng Snorkel)
        print("Generating soft labels with Snorkel...")
        snorkel = Snorkel()
        snorkel.fit(
            dataset_train=train_data,
            dataset_valid=valid_data,
            metric="f1_macro"
        )
        soft_labels = snorkel.predict(train_data)

        with open("train_soft_labels.json", "w", encoding="utf-8") as f:
            json.dump([
                {"text": ex["text"], "soft_label": sl.tolist() if isinstance(sl, np.ndarray) else sl}
                for ex, sl in zip(train_examples, soft_labels)
            ], f, ensure_ascii=False, indent=2)

        print("✅ Saved train_weak_labels.json and train_soft_labels.json")



if __name__ == "__main__":
    generator = WeakSoftLabelGenerator(
        dataset_path="D:/RandD/Python/alchemist/data_home/agnews",  # thư mục chứa train.json, valid.json, test.json
        lf_dir="D:/RandD/Python/alchemist/data_home/agnews/ScriptoriumWS/qwen-local",          # thư mục chứa LF .py
        num_classes=4
    )
    generator.generate_labels()
