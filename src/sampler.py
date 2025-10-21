from genlm.control.sampler.token import TokenSampler
from genlm.control import PromptedLLM
from genlm.control import AWRS
from genlm.control.potential.stateful import StatefulPotential

class VegaLiteSampler(TokenSampler):
    """Custom sampler for preventing proofs."""

    def __init__(self, llm: PromptedLLM, potential):
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

class PowerPotential(StatefulPotential):
    def __init__(self, llm: PromptedLLM, power: float = 4.0):
        self.llm = llm
        assert self.llm.temperature == 1.0
        self.power = power
        super().__init__(llm.vocab)

    async def prefix(self, context):
        """Score a prefix context using the state management system."""
        log_weight = await self.llm.log_probability(context)
        return (self.power - 1.0) * log_weight

    async def complete(self, context):
        """Score a complete context."""
        return 0.0
