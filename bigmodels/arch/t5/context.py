from ..base import InferenceContext

class T5InferenceContext(InferenceContext):
    def __init__(self, hidden_states, input_length):
        self.hidden_states = hidden_states
        self.input_length = input_length


        self.encoder_layers_kv = None
        self.decoder_position_bias = None
        self.past_kv = None
        self.encoder_mask = None
    