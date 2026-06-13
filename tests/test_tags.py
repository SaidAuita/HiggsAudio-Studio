"""Реальный юнит-тест чистой логики фильтра тегов (без torch/модели).
Запуск: python tests/test_tags.py  (или pytest)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from director import filter_tags, WHITELIST


def test_keeps_valid():
    assert filter_tags("<|emotion:elation|>Привет") == "<|emotion:elation|>Привет"


def test_strips_invalid():
    assert filter_tags("<|emotion:bogus|>Привет <|sfx:laughter|>ха") == "Привет <|sfx:laughter|>ха"


def test_inline_sfx_kept():
    assert filter_tags("<|sfx:cough|>кхм, начнём") == "<|sfx:cough|>кхм, начнём"


def test_whitelist_counts():
    assert len(WHITELIST["emotion"]) == 21
    assert len(WHITELIST["prosody"]) == 10
    assert len(WHITELIST["style"]) == 3
    assert len(WHITELIST["sfx"]) == 9


if __name__ == "__main__":
    test_keeps_valid()
    test_strips_invalid()
    test_inline_sfx_kept()
    test_whitelist_counts()
    print("OK: all tag tests passed (43 tags: 21+10+3+9)")
