import asyncio
import subprocess
import json
import visualize
import random
import time
from genlm.control import PromptedLLM, JsonSchema
from prompts import system_prompt
from sampler import VegaLiteSampler
from potential import PowerLLM
import argparse
import os
import pandas as pd

MODEL_NAME = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


async def run(prompt: str, dataset_url: str, config: dict) -> None:
    prompt = f"""{prompt}
Use the data URL: {dataset_url}
The columns in the dataset are: {pd.read_csv(dataset_url).columns.tolist()}
"""
    llm = PromptedLLM.from_name(MODEL_NAME, backend="mlx", temperature=1.0)
    if config["lhts"]:
        potential = PowerLLM(llm, 4.0)
    else:
        potential = llm
    llm.prompt_ids = llm.model.tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )

    json_prefix = ""
    completion_prefix = f"""```json
{json_prefix}"""
    json_prefix_bytes = json_prefix.encode("utf-8")
    llm.prompt_ids.extend(llm.model.tokenizer.encode(completion_prefix))
    with open("src/vegalite.schema.json") as f:
        schema = json.load(f)
    schema_potential = JsonSchema(schema, validate=False)
    coerced_schema = schema_potential.coerce(
        llm, f=lambda x: json_prefix_bytes + b"".join(x)
    )
    sampler = VegaLiteSampler(potential, coerced_schema)

    start = time.time()
    sequences = await sampler.smc(
        n_particles=2,
        max_tokens=250,
        ess_threshold=0.9
    )
    elapsed_time = time.time() - start
    response: dict[tuple, float] = dict(sequences.decoded_posterior)

    try:
        sample = random.choices(list(response.keys()), weights=list(response.values()))[
            0
        ]
    except IndexError:
        print("Ran out of tokens!")
        for k in dict(sequences.posterior).keys():
            content = b"".join(list(k)).decode()
            print(f"tok / sec: {len(content) / elapsed_time}")
            print(content)
        raise IndexError

    content = "".join(list(sample))
    print(f"tok / sec: {len(content) / elapsed_time}")
    print(content)
    with open("temp.json", "w") as f:
        f.write(content)
    visualize.convert_and_open()


def base(prompt: str, dataset_url: str) -> None:
    prompt = f"""{prompt}
Use the data URL: {dataset_url}
The columns in the dataset are: {pd.read_csv(dataset_url).columns.tolist()}
"""
    subprocess.run(
        [
            "mlx_lm.generate",
            "--prompt",
            prompt,
            "--model",
            MODEL_NAME,
            "--max-tokens",
            "4000",
            "--temp",
            "0.25",
            "--system-prompt",
            system_prompt,
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="Make a scatter plot of age and height. Use the tests/dataset.csv dataset",
    )
    parser.add_argument(
        "--lhts",
        action="store_true",
    )
    parser.add_argument("--data", type=str, default="tests/dataset.csv")
    args = parser.parse_args()
    config = {
        "lhts": args.lhts
    }
    asyncio.run(run(args.prompt, args.data, config))
    # base(args.prompt, args.data)
