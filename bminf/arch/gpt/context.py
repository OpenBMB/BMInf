from ..base import InferenceContext

class GPTInferenceContext(InferenceContext):
    def __init__(self, past_kv, input_length):
        self.past_kv = past_kv
        self.input_length = input_length
        