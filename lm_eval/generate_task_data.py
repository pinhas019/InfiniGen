import argparse
import json
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--task-name", type=str, required=True)
    parser.add_argument("--num-fewshot", type=int, default=5)
    args = parser.parse_args()

    # For demo, we'll just use HuggingFace's "openbookqa" dataset
    if args.task_name == "openbookqa":
        dataset = load_dataset("openbookqa", "main", split="validation[:{}]".format(args.num_fewshot))
        with open(args.output_file, "w", encoding="utf-8") as f:
            for example in dataset:
                json.dump(example, f)
                f.write("\n")
        print(f"Saved {args.num_fewshot} examples to {args.output_file}")
    else:
        raise ValueError(f"Task {args.task_name} not supported in this demo")

if __name__ == "__main__":
    main()
