import pytest
from transformers import AutoTokenizer


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("google/mt5-base")


def test_smoke(tokenizer):
    inp = "—Ун ҫинчен хыпар ҫук-и? —пӗлесшӗн ҫуннӑ «бензин королӗсем», ҫӗр айне чавса лартнӑ цистернӑсем патне черетлӗ бензовоз пырса ҫитсен."
    assert tokenizer.decode(tokenizer(inp)["input_ids"]) == inp + "</s>"
