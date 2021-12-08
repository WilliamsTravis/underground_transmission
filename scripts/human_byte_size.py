"""Human Byte Size - return nice object size strings.

Adapted from: Mitch McMabers, https://stackoverflow.com/questions/12523586/
python-format-size-application-converting-b-to-kb-mb-gb-tb

Created on Tue Dec  7 15:25:03 2021

@author: travis
"""
import sys


METRIC_LABELS = ["B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
BINARY_LABELS = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"]
PRECISION_OFFSETS = [0.5, 0.05, 0.005, 0.0005]
PRECISION_FORMATS = ["{}{:.0f} {}", "{}{:.1f} {}", "{}{:.2f} {}",
                     "{}{:.3f} {}"]


def hbsize(obj, metric=False, precision=1):
    """Human-readable object byte size."""
    assert precision >= 0, "precision must be greater than or equal to 0."
    assert precision <= 3, "precision must be less than or equal to 3."

    size = sys.getsizeof(obj)

    if metric:
        unit_labels = METRIC_LABELS
    else:
        unit_labels = BINARY_LABELS

    last_label = unit_labels[-1]
    unit_step = 1000 if metric else 1024
    unit_step_thresh = unit_step - PRECISION_OFFSETS[precision]

    is_negative = size < 0
    if is_negative: # Faster than ternary assignment or always running abs
        size = abs(size)

    for unit in unit_labels:
        if size < unit_step_thresh:
            # VERY IMPORTANT:
            # Only accepts the CURRENT unit if we're BELOW the threshold
            # where float rounding behavior would place us into the NEXT
            # unit: F.ex. when rounding a float to 1 decimal, any number
            # ">= 1023.95" will be rounded to "1024.0". Obviously we don't
            # want ugly output such as "1024.0 KiB", since the proper term
            # for that is "1.0 MiB".
            break
        if unit != last_label:
            # We only shrink the number if we HAVEN'T reached the last
            # unit.
            # NOTE: These looped divisions accumulate floating point
            # rounding errors, but each new division pushes the rounding
            # errors further and further down in the decimals, so it'
            # doesn't matter at all.
            size /= unit_step

    if is_negative:
        sign = "-"
    else:
        sign = ""

    out = PRECISION_FORMATS[precision].format(sign, size, unit)

    return out
