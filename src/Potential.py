from genlm.control import AWRS, PromptedLLM
from genlm.control.potential.base import Potential
from genlm.control.sampler.token import TokenSampler

class PowerLLM(Potential):
    def __init__(self, llm:PromptedLLM, power: float):
        super().__init__(llm.vocab)
        self.llm = llm
        self.power = power
        assert self.llm.temperature == 1.0

    async def logw_next(self, context):
        proposal_logps = await self.llm.logw_next(context)
        return proposal_logps.spawn(proposal_logps.weights * self.power)
    
    async def prefix(self, context):
        return 0.0
    
    async def complete(self, context):
        return 0.0
