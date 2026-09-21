"""Mock prepare.py for dry-run/tests: same public names as the real one."""

MAX_SEQ_LEN = 256
TIME_BUDGET = 300
EVAL_TOKENS = 524288


class _Torch:
    @staticmethod
    def manual_seed(seed):
        pass


torch = _Torch()

_BPB_OVERRIDE = None


class Tokenizer:
    pass


def make_dataloader(tokenizer, batch_size, seq_len, split):
    return iter([])


def evaluate_bpb(model, tokenizer, batch_size):
    """Fake metric: returns whatever mock train.py computed for itself."""
    return _BPB_OVERRIDE
