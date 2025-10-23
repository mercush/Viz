from genlm.control.sampler.token import TokenSampler
from genlm.control import PromptedLLM
from genlm.control import AWRS
from genlm.control.potential.stateful import StatefulPotential
from genlm.control.potential.base import Potential
from genlm.control.util import fast_sample_lazyweights

class VegaLiteSampler(TokenSampler):
    """Custom sampler for preventing proofs."""

    def __init__(self, llm: Potential, potential):
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

class PowerSampler(Potential):
    """Custom sampler for preventing proofs."""

    def __init__(self, llm: PromptedLLM, power: float):
        self.llm = llm
        self.power = power
        super().__init__(vocabulary=self.llm.vocab)

    async def sample(self, context, draw=None):
        logws = await self.llm.logw_next(context)
        logps = logws.normalize()
        token = fast_sample_lazyweights(logps)
        logps_base = (logws.weights * self.llm.temperature).normalize()
        weight = logps_base[token] * self.power - logps[token]
        return token, weight, logps[token]
