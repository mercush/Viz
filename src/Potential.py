import time
from typing import Literal

import numpy as np
import pandas as pd
from genlm.control import AWRS, Potential, PromptedLLM
from genlm.control.sampler.token import TokenSampler
from prefix import prefix_check, complete_check, process_dataset_url

# def process_dataset_url(dataset_url: str) -> dict:
#     """Return a dict mapping dataset URLs to their processed data for VegaLite visualization.
#     Args:
#         dataset_url: URL of the dataset to process
#
#     Returns:
#         Dict with 'fields' key containing list of field dicts with 'name', 'type, 'distinctValues'
#     """
#     # Read the CSV file
#     df = pd.read_csv(dataset_url)
#
#     # Process each field
#     fields = []
#     for col in df.columns:
#         field_info = {
#             'name': col,
#             'type': infer_vegalite_type(df[col]),
#             'distinctValues': int(df[col].nunique())
#         }
#         fields.append(field_info)
#
#     return {'fields': fields}
#
# def infer_vegalite_type(series: pd.Series) -> str:
#     """Infer VegaLite field type from pandas Series.
#
#     Args:
#         series: Pandas Series to infer type from
#
#     Returns:
#         One of 'quantitative', 'nominal', 'ordinal', 'temporal'
#     """
#     # Check if temporal
#     if pd.api.types.is_datetime64_any_dtype(series):
#         return 'temporal'
#
#     # Check if numeric
#     if pd.api.types.is_numeric_dtype(series):
#         return 'quantitative'
#
#     # Default to nominal for strings/objects
#     return 'nominal'

class VegaLiteSampler(TokenSampler):
    """Custom sampler for preventing proofs."""

    def __init__(self, llm, potential):
        self.llm = llm
        self.potential = potential
        self.AWRS = AWRS(self.llm, self.potential)
        super().__init__(target=self.llm * self.potential)

    async def sample(self, context, draw=None):
        if complete_check(b''.join(context).decode):
            return self.llm.eos, 0.0, np.nan
        return await self.AWRS.sample(context)

class VegaLitePotential(Potential):
    """Binary potential that prohibits comments."""

    def __init__(self, llm: PromptedLLM, dataset_url: str) -> None:
        super().__init__(llm.vocab, llm.token_type, llm.eos)
        self.dataset_url = dataset_url
        self.dataset = process_dataset_url(dataset_url)

    async def prefix(self, context: list) -> float:  # type: ignore[override]
        return 0.0 if prefix_check(b''.join(context), self.dataset) else float('-inf')

    async def complete(self, _: list) -> float:  # type: ignore[override]
        return 0.0

if __name__ == "__main__":
    # Test with the dataset
    result = process_dataset_url('tests/dataset.csv')

    print("Dataset processing result:")
    print(f"Number of fields: {len(result['fields'])}")
    print("\nField details:")
    for field in result['fields']:
        print(f"  - {field['name']}: {field['type']} (distinct values: {field['distinctValues']})")
