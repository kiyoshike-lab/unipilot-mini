"""Web-side budget contract; deliberately separate from Foundation research tests."""
from api.main import chat_prompt
from tokenizer.tokenizer import BPETokenizer


def test_current_chat_framing_fits_reserved_budget():
    assert len(chat_prompt('').encode('utf-8')) <= 192
    tokenizer=BPETokenizer.load('tokenizer/vocab-v02-512.json')
    for text in ['a'*288, 'あ'*96, '🙂'*72]:
        assert len(tokenizer.encode(chat_prompt(text)))+32 <= 512
