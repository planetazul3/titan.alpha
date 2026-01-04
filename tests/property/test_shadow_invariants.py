
import pytest
from hypothesis import given, strategies as st
from datetime import datetime, timezone
from execution.shadow_store import ShadowTradeRecord

@st.composite
def shadow_trade_record_strategy(draw):
    """Generate valid ShadowTradeRecords."""
    return ShadowTradeRecord(
        trade_id=draw(st.uuids()).hex,
        # Generate naive and force UTC to avoid Hypothesis version conflicts
        timestamp=draw(st.datetimes(max_value=datetime.now().replace(microsecond=0))).replace(tzinfo=timezone.utc),
        contract_type=draw(st.sampled_from(["RISE_FALL", "TOUCH", "RANGE"])),
        direction=draw(st.sampled_from(["CALL", "PUT"])),
        probability=draw(st.floats(min_value=0.0, max_value=1.0)),
        reconstruction_error=draw(st.floats(min_value=0.0, max_value=100.0)),
        regime_state=draw(st.sampled_from(["TRUSTED", "CAUTION", "VETO"])),
        entry_price=draw(st.floats(min_value=0.1, max_value=10000.0)),
        model_version="prop_v1"
    )

class TestShadowInvariants:
    """Property-based tests for Shadow Store logical invariants."""
    
    @given(shadow_trade_record_strategy())
    def test_record_immutability_logic(self, record):
        """Invariant: with_outcome returns NEW record, original untouched."""
        # Mutation check
        original_ts = record.timestamp
        
        resolved = record.with_outcome(outcome=True, exit_price=105.0)
        
        assert resolved is not record
        assert resolved.trade_id == record.trade_id
        assert resolved.outcome is True
        # Original must be None (unless generated that way, but strategy defaults outcome=None via init default?)
        # Wait, strategy calls init. outcome defaults to None.
        assert record.outcome is None
        assert record.timestamp == original_ts
        
    @given(shadow_trade_record_strategy())
    def test_resolution_completeness(self, record):
        """Invariant: Resolved record has timestamps and prices."""
        resolved = record.with_outcome(outcome=False, exit_price=95.0)
        
        assert resolved.is_resolved()
        assert resolved.exit_price is not None
        assert resolved.resolved_at is not None
        assert resolved.resolved_at >= resolved.timestamp # Causality
