import asyncio
import json
import re
import visualize
import random
import time
from genlm.control import PromptedLLM, direct_token_sampler, AWRS, JsonSchema
from prompts import system_prompt
from Potential import VegaLitePotential, VegaLiteSampler
import argparse
import os

MODEL_NAME = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

async def run(prompt: str, dataset_url: str) -> None:
    llm = PromptedLLM.from_name(MODEL_NAME, backend="mlx")
    llm.prompt_ids = llm.model.tokenizer.apply_chat_template(
            conversation=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    json_prefix = f"""{{"data": {{"url": "{dataset_url}"}},
"""
    completion_prefix = f"""```json
{json_prefix}"""
    json_prefix_bytes = json_prefix.encode('utf-8')
    llm.prompt_ids.extend(llm.model.tokenizer.encode(completion_prefix))
    with open("src/vegalite.schema.json") as f:
        schema = json.load(f)
    schema_potential = JsonSchema(schema, validate=False)
    coerced_schema = schema_potential.coerce(llm, f=lambda x: json_prefix_bytes + b"".join(x))
    sampler = AWRS(llm, coerced_schema)

    # sampler = direct_token_sampler(llm)

    start = time.time()
    sequences = await sampler.smc(
        n_particles=1,
        max_tokens=250,
        ess_threshold=0.5,
    )
    elapsed_time = time.time() - start
    response: dict[tuple, float] = dict(sequences.decoded_posterior)

    sample = random.choices(list(response.keys()), weights=list(response.values()))[0]
    content = json_prefix + "".join(list(sample))
    print(f"tok / sec: {len(content) / elapsed_time}")
    print(content)
    with open("temp.json", "w") as f:
        f.write(content)
    visualize.convert_and_open()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Make a scatter plot of age and height. Use the tests/dataset.csv dataset")
    parser.add_argument("--data", type=str, default="tests/dataset.csv")
    args = parser.parse_args()
    asyncio.run(run(args.prompt, args.data))
