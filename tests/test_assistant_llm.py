"""Tests verifying Assistant NLP regex parsing and prompt parsing."""

import pytest
from assistant.assistant import find_transaction_id


def test_assistant_finds_the_transaction_id():
    assert find_transaction_id("Why was transaction #3487632 flagged?") == 3487632
    assert find_transaction_id("hello") is None
    assert find_transaction_id("transaction 1234 looks suspicious") == 1234


def test_assistant_ignores_short_numbers():
    """Numbers shorter than 4 digits should be ignored."""
    assert find_transaction_id("3 items in cart") is None
    assert find_transaction_id("score: 99 out of 100") is None
