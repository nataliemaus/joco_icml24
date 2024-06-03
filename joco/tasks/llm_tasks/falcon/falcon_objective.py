from joco.paths import PATH_TO_FALCON_MODEL
from joco.tasks.llm_tasks.llm_objective import LLMObjective
from joco.tasks.llm_tasks.utils.llm_utils import HuggingFaceLanguageModel


class FalconObjective(LLMObjective):
    """The goal of this task is to optimize prompts consisting
    of four words that, when passed into the LLM, generate toxic text outputs"""

    def set_llm_name(self):
        self.llm_name = "falcon"

    def get_language_model(
        self,
        max_gen_length,
        n_tokens,
        seed_text,
        num_gen_seq,
    ):
        seed_text_len = 0
        if seed_text is not None:
            seed_text_len = len(seed_text.split()) * 2
        max_num_tokens = max_gen_length + n_tokens + seed_text_len
        lm = HuggingFaceLanguageModel(max_num_tokens, num_gen_seq)
        lm.load_model(self.llm_name, PATH_TO_FALCON_MODEL)
        return lm
