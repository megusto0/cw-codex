from __future__ import annotations

import xml.etree.ElementTree as ET

from app.pipeline import _parse_interval_stream, _parse_simple_stream


def test_parse_simple_stream_preserves_expected_columns_for_empty_node():
    node = ET.fromstring("<basis_heart_rate></basis_heart_rate>")

    frame = _parse_simple_stream(node, value_keys=("value",))

    assert frame.empty
    assert list(frame.columns) == ["timestamp", "value"]


def test_parse_interval_stream_preserves_expected_columns_for_empty_node():
    node = ET.fromstring("<temp_basal></temp_basal>")

    frame = _parse_interval_stream(node, value_keys=("value",))

    assert frame.empty
    assert list(frame.columns) == ["timestamp", "end_timestamp", "value"]
