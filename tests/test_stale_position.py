"""Tests for stale position detection and consecutive gate failure counting."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from db.models import init_tables, get_consecutive_gate_failures


class TestGetConsecutiveGateFailures:
    def _insert_decision(self, conn, gates_passed, current_shares, date_str="2026-03-18"):
        conn.execute(
            "INSERT INTO decisions (date, timestamp, gates_passed, current_shares, symbol, status) "
            "VALUES (?, ?, ?, ?, 'TQQQ', 'COMPLETE')",
            [date_str, f"{date_str}T15:50:00", gates_passed, current_shares],
        )
        conn.commit()

    def test_no_records_returns_zero(self, db_conn):
        assert get_consecutive_gate_failures(db_conn) == 0

    def test_single_failure_while_holding(self, db_conn):
        self._insert_decision(db_conn, gates_passed=0, current_shares=67)
        assert get_consecutive_gate_failures(db_conn) == 1

    def test_consecutive_failures(self, db_conn):
        for i in range(5):
            self._insert_decision(db_conn, gates_passed=0, current_shares=67, date_str=f"2026-03-{13+i:02d}")
        assert get_consecutive_gate_failures(db_conn) == 5

    def test_resets_on_gate_pass(self, db_conn):
        # Older: 3 failures, then a pass, then 2 more failures
        self._insert_decision(db_conn, gates_passed=0, current_shares=67, date_str="2026-03-10")
        self._insert_decision(db_conn, gates_passed=0, current_shares=67, date_str="2026-03-11")
        self._insert_decision(db_conn, gates_passed=0, current_shares=67, date_str="2026-03-12")
        self._insert_decision(db_conn, gates_passed=1, current_shares=67, date_str="2026-03-13")
        self._insert_decision(db_conn, gates_passed=0, current_shares=67, date_str="2026-03-14")
        self._insert_decision(db_conn, gates_passed=0, current_shares=67, date_str="2026-03-17")
        # Most recent 2 are failures, then a pass breaks the streak
        assert get_consecutive_gate_failures(db_conn) == 2

    def test_resets_on_zero_position(self, db_conn):
        # Failures with no position don't count
        self._insert_decision(db_conn, gates_passed=0, current_shares=0, date_str="2026-03-10")
        self._insert_decision(db_conn, gates_passed=0, current_shares=67, date_str="2026-03-11")
        self._insert_decision(db_conn, gates_passed=0, current_shares=67, date_str="2026-03-12")
        # Most recent 2 are failures with position, then zero-position breaks streak
        assert get_consecutive_gate_failures(db_conn) == 2

    def test_ignores_sqqq_decisions(self, db_conn):
        """Only TQQQ decisions count for stale position check."""
        self._insert_decision(db_conn, gates_passed=0, current_shares=67, date_str="2026-03-17")
        # Insert an SQQQ decision (should be ignored)
        db_conn.execute(
            "INSERT INTO decisions (date, timestamp, gates_passed, current_shares, symbol, status) "
            "VALUES (?, ?, ?, ?, 'SQQQ', 'COMPLETE')",
            ["2026-03-18", "2026-03-18T15:50:00", 1, 100],
        )
        db_conn.commit()
        assert get_consecutive_gate_failures(db_conn) == 1

    def test_gate_pass_while_holding_resets(self, db_conn):
        """A day where gates passed (even if no trade executed) resets the streak."""
        self._insert_decision(db_conn, gates_passed=0, current_shares=67, date_str="2026-03-14")
        self._insert_decision(db_conn, gates_passed=1, current_shares=67, date_str="2026-03-17")
        self._insert_decision(db_conn, gates_passed=0, current_shares=67, date_str="2026-03-18")
        assert get_consecutive_gate_failures(db_conn) == 1
