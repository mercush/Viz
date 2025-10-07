from genlm.control import AWRS
from genlm.control.sampler.token import TokenSampler

class VegaLiteSampler(TokenSampler):
    """Custom sampler for preventing proofs."""

    def __init__(self, llm, potential):
        self.llm = llm
        self.potential = potential
        self.AWRS = AWRS(self.llm, self.potential)
        super().__init__(target=self.llm * self.potential)

    async def sample(self, context, draw=None):
        s = b''.join(context)# .decode()
        open_count = s.count(b"{")
        close_count = s.count(b"}")
        if open_count == close_count and open_count > 0:
            return self.llm.eos, 0.0, 0.0
        return await self.AWRS.sample(context)
