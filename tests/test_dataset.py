import numpy as np
import pytest
from auralbench import dataset as m


def test_record_example():
    value = m.record({"id": "sample-1"})
    assert value["split"] == "test" and value["reference_audio"] is None


def test_record_boundaries():
    for value in [
        {},
        {"id": "../x"},
        {"id": "x", "extra": 1},
        {"id": "x", "reference_audio": "../a.wav", "candidate_audio": "b.wav"},
        {"id": "x", "reference_audio": "a.wav"},
        {"id": "x", "split": "unknown"},
    ]:
        with pytest.raises(ValueError):
            m.record(value)


def test_record_invariants():
    value = m.record({"id": "x", "reference_text": "你好", "hypothesis_text": "你好"})
    assert set(value) == {
        "id",
        "reference_text",
        "hypothesis_text",
        "reference_audio",
        "candidate_audio",
        "speaker",
        "split",
    }
    assert m.record(value) == value
    for path in ["/a.wav", "C:/a.wav", "../a.wav", "a/../../b.wav"]:
        with pytest.raises(ValueError):
            m.record({"id": "x", "reference_audio": path, "candidate_audio": "b.wav"})


def test_records_encode_example():
    assert m.records_encode([{"id": "a"}]).endswith("\n")


def test_records_encode_boundaries():
    assert m.records_encode([]) == ""
    with pytest.raises(ValueError):
        m.records_encode([{"id": "x"}, {"id": "x"}])
    with pytest.raises(ValueError):
        m.records_encode([{"id": "x", "reference_text": np.nan}])


def test_records_encode_invariants():
    import json

    values = [{"id": "a", "reference_text": "你好"}, {"id": "b"}]
    encoded = m.records_encode(values)
    assert [json.loads(line) for line in encoded.splitlines()] == [
        m.record(value) for value in values
    ]
    assert encoded == m.records_encode([dict(reversed(list(value.items()))) for value in values])


def test_records_decode_example():
    values = [m.record({"id": "a", "reference_text": "你好"})]
    assert m.records_decode(m.records_encode(values)) == values


def test_records_decode_boundaries():
    for text in ["null", "{}", '{"id":"x"}', '{"id":"x","id":"y"}', '{"value":NaN}']:
        with pytest.raises(ValueError):
            m.records_decode(text)
    assert m.records_decode(" \n\n") == []


def test_records_decode_invariants():
    values = [m.record({"id": "a"}), m.record({"id": "b"})]
    encoded = m.records_encode(values)
    assert m.records_encode(m.records_decode(encoded)) == encoded
    with pytest.raises(ValueError, match="duplicate record IDs"):
        m.records_decode(encoded + encoded)
    with pytest.raises(ValueError, match="line 2"):
        m.records_decode(m.records_encode(values[:1]) + "bad")


def test_speaker_split_example():
    values = [{"id": str(i), "speaker": str(i // 2)} for i in range(8)]
    result = m.speaker_split(values, 0.5, 0.25, 3)
    assert [len(result[key]) for key in ["train", "validation", "test"]] == [4, 2, 2]


def test_speaker_split_boundaries():
    assert m.speaker_split([]) == {"train": [], "validation": [], "test": []}
    for train, validation in [(-1, 0.1), (0.9, 0.2), (np.nan, 0.1)]:
        with pytest.raises(ValueError):
            m.speaker_split([], train, validation)
    with pytest.raises(ValueError):
        m.speaker_split([], seed=-1)


def test_speaker_split_invariants():
    values = [{"id": str(i), "speaker": "s" + str(i // 3)} for i in range(18)]
    result = m.speaker_split(values, 0.5, 0.25, 11)
    assert result == m.speaker_split(values, 0.5, 0.25, 11)
    seen = {}
    for split, records in result.items():
        for value in records:
            assert value["speaker"] not in seen or seen[value["speaker"]] == split
            seen[value["speaker"]] = split
            assert value["split"] == split
    assert sum(map(len, result.values())) == len(values)


def test_shard_records_example():
    values = [{"id": str(i)} for i in range(10)]
    assert len(m.shard_records(values, 0, 1)) == 10


def test_shard_records_boundaries():
    for index, count in [(-1, 2), (2, 2), (0, 0), (True, 2), (0, 1025)]:
        with pytest.raises(ValueError):
            m.shard_records([], index, count)
    assert m.shard_records([], 0, 2) == []


def test_shard_records_invariants():
    values = [{"id": str(i)} for i in range(31)]
    for count in range(1, 8):
        shards = [m.shard_records(values, index, count) for index in range(count)]
        ids = [value["id"] for shard in shards for value in shard]
        assert len(ids) == len(set(ids)) == len(values)
        assert set(ids) == {value["id"] for value in values}


def test_dataset_digest_example():
    assert len(m.dataset_digest([{"id": "a"}])) == 64
