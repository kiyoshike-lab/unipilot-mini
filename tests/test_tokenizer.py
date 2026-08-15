from tokenizer.tokenizer import BPETokenizer


def test_japanese_round_trip_and_save_load(tmp_path):
    text = "今日は大学の課題を進めます。"
    tokenizer = BPETokenizer(); tokenizer.train([text, text, "明日は試験です。"], 300)
    assert tokenizer.decode(tokenizer.encode(text)) == text
    path = tmp_path / "vocab.json"; tokenizer.save(path)
    restored = BPETokenizer.load(path)
    assert restored.decode(restored.encode(text)) == text


def test_special_tokens():
    tokenizer = BPETokenizer(); ids = tokenizer.encode("<BOS><USER>こんにちは<EOS>")
    assert ids[0] == tokenizer.bos_id and ids[-1] == tokenizer.eos_id
