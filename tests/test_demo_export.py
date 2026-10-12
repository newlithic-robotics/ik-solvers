import numpy as np

from newlithic_lerobot.demo_export import PHASE, drop_sync_claps, phases_from_gap, two_level_threshold


def gap_signal(fps=30.0, seconds=20.0, holds=((5.0, 9.0), (12.0, 16.0)), open_mm=24.0, closed_mm=9.5, ramp_s=0.5):
    t = np.arange(int(seconds * fps)) / fps
    g = np.full(len(t), open_mm)
    for a, b in holds:
        g[(t >= a) & (t <= b)] = closed_mm
        ramp = (t > a - ramp_s) & (t < a)
        g[ramp] = open_mm + (closed_mm - open_mm) * (t[ramp] - (a - ramp_s)) / ramp_s
    return t, g / 1000.0


def test_two_level_threshold_sits_between_levels():
    t, g = gap_signal()
    lo, hi = two_level_threshold(g)
    assert 0.0095 < lo < hi < 0.024


def test_phases_from_gap_finds_holds_and_labels_phases():
    t, g = gap_signal()
    phase, hold_id, thr, events = phases_from_gap(g, t)
    # contact = the first frame within 1 mm of the held gap: the last ramp frame before the plateau
    assert len(events) == 2 and [e["contact"] for e in events] == [149, 359]
    assert abs(events[0]["gap_mm"] - 9.5) < 0.01
    # contact frame is a grasp, the plateau a hold, the frame after the release a release, well before: approach
    assert phase[150] == PHASE["grasp"] and phase[200] == PHASE["hold"] and phase[271] == PHASE["release"] and phase[120] == PHASE["approach"]
    assert hold_id[200] == 0 and hold_id[400] == 1 and hold_id[10] == -1
    assert set(np.unique(hold_id)) == {-1, 0, 1}
    # between the release window (0.5 s after 9.0 s) and the next approach (1.5 s before 12.0 s) the hand transports
    assert phase[300] == PHASE["transport"]


def test_short_closing_is_not_a_hold():
    t, g = gap_signal(holds=((5.0, 5.1),), ramp_s=0.05)          # closed for 0.15 s: under the 0.3 s minimum
    phase, hold_id, _, events = phases_from_gap(g, t)
    assert events == [] and np.all(phase == PHASE["transport"])


class _H:
    def __init__(self, t, phase, hold_id, events):
        self.t, self.phase, self.hold_id, self.events = t, phase, hold_id, events


def test_drop_sync_claps_relabels_simultaneous_short_closings():
    t, ga = gap_signal(holds=((3.0, 3.8), (8.0, 12.0)))
    _, gb = gap_signal(holds=((3.1, 3.9),))
    pa, ha, _, ea = phases_from_gap(ga, t)
    pb, hb, _, eb = phases_from_gap(gb, t)
    hands = {"left": _H(t, pa, ha, ea), "right": _H(t, pb, hb, eb)}
    dropped = drop_sync_claps(hands)
    assert sorted(d[0] for d in dropped) == ["left", "right"]
    assert hands["left"].events[0]["sync_clap"] and "sync_clap" not in hands["left"].events[1]
    assert np.all(hands["left"].phase[hands["left"].hold_id == 0] == PHASE["transport"]) or not np.any(hands["left"].hold_id == 0)
    assert np.any(hands["left"].hold_id == 1)                 # the real grasp survives
