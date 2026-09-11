from scripts.closure_resource_monitor import interval_stats, valid_memory


def test_partial_intervals_are_weighted_and_missing_is_not_zero():
    rows = [dict(interval_start=0., t=2., usage=20.),
            dict(interval_start=2., t=3., usage=80.),
            dict(interval_start=3., t=4., usage=None)]
    result = interval_stats(rows, 'usage', 1., 4.)
    assert result == dict(mean=50., sampled_peak=80., samples=2, covered_seconds=2.)


def test_no_samples_remains_unavailable():
    assert interval_stats([], 'usage', 0., 1.)['mean'] is None
    assert interval_stats([dict(interval_start=0., t=1., usage=None)], 'usage', 0., 1.)['sampled_peak'] is None


def test_nvml_unsupported_memory_is_not_zero_or_uint64_peak():
    assert valid_memory(None) is None
    assert valid_memory(2**64-1) is None
    assert valid_memory(0) == 0
    assert valid_memory(1024) == 1024
