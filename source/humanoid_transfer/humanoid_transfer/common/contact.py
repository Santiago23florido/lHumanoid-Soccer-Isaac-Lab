"""How faithfully a gait's contact transitions are reproduced, and how to resample them.

The gait signature (:mod:`.gait`) compares duty factors, stride lengths and stride
periods, which a body can match while getting every contact transition wrong: lifting a
foot early and landing it late give the same duty. This module measures the
transitions themselves, plan against body, in body units:

* **Light contact** -- a foot carrying more than the contact threshold but at most
  :data:`LIGHT_LOAD` of the body's weight: the load transfer of double support. How
  much of it a plan asks for, how much a body produces, and how often the body's foot
  is in the air where the plan asks for it (it lifted early or landed late).
* **Grazes** -- a foot carrying weight where the plan lifts it.
* **Events** -- each planned lift-off and touchdown matched to the body's nearest event
  of the same kind on the same foot: the lag, in pendulum times.
* **Clearance** -- the sole's highest point above its own stance level over each swing.

**Resampling across control rates.** A duty factor is a fraction of time: sampled at
any rate, a binary contact signal estimates it without bias. Interpolating the *load*
linearly and then thresholding it does not. Between a stance sample (about half the
weight) and a swing sample (zero) the interpolated load stays above a 5 % threshold for
about nine tenths of the interval, so every swing boundary reads as contact and every
stance as longer. Measured on the G1 teacher's captures resampled to the NAO's step
(0.126 against 0.077 ``T``), that adds 0.03 to 0.06 to each duty factor, while nearest
sampling and interpolation in log units (``log(load + 0.01)``, the generator's own
space) stay within 0.01. :func:`resample` offers all three.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .skill_state import DEFAULT_CONTACT_THRESHOLD, SKILL_LAYOUT, ZLayout

LIGHT_LOAD = 0.25
"""Upper bound, in body weights, of a light contact: a load transfer, not a stance."""

LOAD_EPSILON = 0.01
"""The generator's load warp, ``log(load + LOAD_EPSILON)``."""

LOAD_MODES = ("linear", "log", "nearest")


def resample(
    z: np.ndarray,
    done: np.ndarray,
    source_dt: float,
    target_dt: float,
    loads: str = "linear",
    layout: ZLayout = SKILL_LAYOUT,
) -> tuple[np.ndarray, np.ndarray]:
    """A ``(T, N, 27)`` sequence sampled every ``source_dt`` resampled every ``target_dt``.

    Every field is interpolated linearly in time; the feet's loads by ``loads``:
    ``linear`` (as every other field), ``log`` (linearly in ``log(load + 0.01)``) or
    ``nearest`` (the nearest original sample). A new sample ends an episode if any
    original step since the previous one did. Returns the samples and their ``done``.
    """
    if loads not in LOAD_MODES:
        raise ValueError(f"unknown load mode {loads!r}; expected one of {LOAD_MODES}")
    z = np.asarray(z, dtype=float)
    source = np.arange(z.shape[0]) * source_dt
    target = np.arange(0.0, source[-1], target_dt)
    resampled = np.empty((len(target), z.shape[1], z.shape[2]))
    for env in range(z.shape[1]):
        for k in range(z.shape[2]):
            resampled[:, env, k] = np.interp(target, source, z[:, env, k])
    span = layout.slice_of("foot_load")
    if loads == "log":
        logged = np.log(np.clip(z[..., span], 0.0, None) + LOAD_EPSILON)
        for env in range(z.shape[1]):
            for k in range(logged.shape[-1]):
                value = np.exp(np.interp(target, source, logged[:, env, k])) - LOAD_EPSILON
                resampled[:, env, span.start + k] = value
    elif loads == "nearest":
        nearest = np.clip(np.rint(target / source_dt).astype(int), 0, z.shape[0] - 1)
        resampled[..., span] = z[nearest][..., span]
    ended = np.cumsum(done, axis=0)
    index = np.minimum(np.searchsorted(source, target, side="right") - 1, len(source) - 1)
    counts = ended[index]
    resampled_done = np.zeros(counts.shape, dtype=bool)
    resampled_done[1:] = counts[1:] > counts[:-1]
    return resampled, resampled_done


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, stop)`` of each run of True in a 1-D mask."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def _events(contact: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(T,)`` lift-offs and touchdowns of one foot, only between two valid samples."""
    both = valid[1:] & valid[:-1]
    liftoff = np.zeros_like(contact)
    touchdown = np.zeros_like(contact)
    liftoff[1:] = contact[:-1] & ~contact[1:] & both
    touchdown[1:] = ~contact[:-1] & contact[1:] & both
    return liftoff, touchdown


def _segment(valid: np.ndarray) -> np.ndarray:
    """``(T,)`` an id per contiguous run of valid samples (-1 where invalid)."""
    ids = np.full(valid.shape, -1)
    for n, (start, stop) in enumerate(_runs(valid)):
        ids[start:stop] = n
    return ids


def _lags(asked: np.ndarray, body: np.ndarray, segment: np.ndarray, window: int) -> list[int]:
    """Steps from each asked event to the body's nearest event within ``window``, same run."""
    body_times = np.flatnonzero(body)
    lags = []
    for t in np.flatnonzero(asked):
        near = body_times[np.abs(body_times - t) <= window]
        near = near[segment[near] == segment[t]]
        if near.size:
            lags.append(int(near[np.argmin(np.abs(near - t))] - t))
    return lags


def _clearances(height: np.ndarray, contact: np.ndarray, valid: np.ndarray) -> list[float]:
    """Per swing of one foot: its highest point above the median of its stance."""
    stance = contact & valid
    if not stance.any():
        return []
    base = float(np.median(height[stance]))
    out = []
    for start, stop in _runs(~contact & valid):
        out.append(float(height[start:stop].max() - base))
    return out


@dataclass(frozen=True)
class ContactFidelity:
    """Per foot ``(left, right)``; shares are over the valid samples."""

    duty_body: tuple[float, float]
    light_share_body: tuple[float, float]
    clearance_body: tuple[float, float]
    """Median over swings, leg lengths."""

    duty_asked: tuple[float, float] | None = None
    light_share_asked: tuple[float, float] | None = None
    airborne_when_light: tuple[float, float] | None = None
    """``P(body foot below threshold | plan asks a light contact)``."""

    graze_in_swing: tuple[float, float] | None = None
    """``P(body foot above threshold | plan lifts it)``."""

    lifted_in_stance: tuple[float, float] | None = None
    """``P(body foot below threshold | plan asks more than a light contact)``."""

    agreement: tuple[float, float] | None = None
    """``P(body contact == planned contact)``."""

    liftoff_lag: tuple[float, float] | None = None
    """Median steps from a planned lift-off to the body's, in ``T`` (positive: later)."""

    touchdown_lag: tuple[float, float] | None = None
    liftoff_lag_mean: tuple[float, float] | None = None
    """The mean of the same lags, ``T``: a bias below one control step shows here."""

    touchdown_lag_mean: tuple[float, float] | None = None
    liftoff_matched: tuple[float, float] | None = None
    """Share of planned lift-offs the body matched within the window."""

    touchdown_matched: tuple[float, float] | None = None
    clearance_asked: tuple[float, float] | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _pair(values: list[float]) -> tuple[float, float]:
    return tuple(float(v) for v in values)  # type: ignore[return-value]


def foot_height(z: np.ndarray, layout: ZLayout = SKILL_LAYOUT) -> np.ndarray:
    """``(..., 2)`` each sole point's height above the ground, leg lengths."""
    fields = layout.unpack(z)
    return fields["height"][..., None] + fields["foot_position"][..., 2]


def contact_fidelity(
    body: np.ndarray,
    done: np.ndarray,
    dt_star: float,
    asked: np.ndarray | None = None,
    threshold: float = DEFAULT_CONTACT_THRESHOLD,
    light: float = LIGHT_LOAD,
    window: float = 0.5,
    layout: ZLayout = SKILL_LAYOUT,
) -> ContactFidelity:
    """Contact transitions of ``body`` ``(T, N, 27)``, and against ``asked`` if given.

    ``asked[t]`` is the plan's due state for ``body[t]``: callers align them (a capture's
    ``contract_due[t]`` is the target of the state recorded at ``t + 1``). A sample is
    valid unless it is the first of an episode (``done`` at the step before) -- it holds
    no plan. ``window`` is in ``T``: how far an event may be matched.
    """
    load = layout.unpack(body)["foot_load"]
    contact = load > threshold
    first = np.zeros_like(done, dtype=bool)
    first[0] = True
    first[1:] = done[:-1]
    valid = ~first
    height = foot_height(body, layout)
    steps = max(1, round(window / dt_star))
    n_env = body.shape[1]

    duty, light_body, clearance = [], [], []
    for side in range(2):
        c, v = contact[..., side], valid
        duty.append(float(c[v].mean()))
        light_body.append(
            float(((load[..., side] > threshold) & (load[..., side] <= light))[v].mean())
        )
        swings = []
        for env in range(n_env):
            swings += _clearances(height[:, env, side], c[:, env], v[:, env])
        clearance.append(float(np.median(swings)) if swings else float("nan"))
    if asked is None:
        return ContactFidelity(_pair(duty), _pair(light_body), _pair(clearance))

    asked_load = layout.unpack(asked)["foot_load"]
    asked_contact = asked_load > threshold
    asked_height = foot_height(asked, layout)
    out: dict[str, list] = {
        k: []
        for k in (
            "duty_asked",
            "light_share_asked",
            "airborne_when_light",
            "graze_in_swing",
            "lifted_in_stance",
            "agreement",
            "liftoff_lag",
            "touchdown_lag",
            "liftoff_lag_mean",
            "touchdown_lag_mean",
            "liftoff_matched",
            "touchdown_matched",
            "clearance_asked",
        )
    }
    for side in range(2):
        a, b, v = asked_contact[..., side], contact[..., side], valid
        a_light = (asked_load[..., side] > threshold) & (asked_load[..., side] <= light)
        firm = asked_load[..., side] > light
        out["duty_asked"].append(float(a[v].mean()))
        out["light_share_asked"].append(float(a_light[v].mean()))
        out["airborne_when_light"].append(
            float((~b & a_light & v).sum() / max((a_light & v).sum(), 1))
        )
        out["graze_in_swing"].append(float((b & ~a & v).sum() / max((~a & v).sum(), 1)))
        out["lifted_in_stance"].append(float((~b & firm & v).sum() / max((firm & v).sum(), 1)))
        out["agreement"].append(float((a == b)[v].mean()))
        lags = {"liftoff": [], "touchdown": []}
        asked_events = {"liftoff": 0, "touchdown": 0}
        swings = []
        for env in range(n_env):
            segment = _segment(v[:, env])
            a_off, a_on = _events(a[:, env], v[:, env])
            b_off, b_on = _events(b[:, env], v[:, env])
            for kind, (ae, be) in {"liftoff": (a_off, b_off), "touchdown": (a_on, b_on)}.items():
                asked_events[kind] += int(ae.sum())
                lags[kind] += _lags(ae, be, segment, steps)
            swings += _clearances(asked_height[:, env, side], a[:, env], v[:, env])
        for kind in ("liftoff", "touchdown"):
            found = lags[kind]
            out[f"{kind}_lag"].append(float(np.median(found)) * dt_star if found else float("nan"))
            out[f"{kind}_lag_mean"].append(
                float(np.mean(found)) * dt_star if found else float("nan")
            )
            out[f"{kind}_matched"].append(len(found) / max(asked_events[kind], 1))
        out["clearance_asked"].append(float(np.median(swings)) if swings else float("nan"))
    return ContactFidelity(
        _pair(duty), _pair(light_body), _pair(clearance), **{k: _pair(v) for k, v in out.items()}
    )


def capture_fidelity(data, dt_star: float, **kwargs) -> ContactFidelity:
    """:func:`contact_fidelity` of a capture (an ``np.load`` archive), its plan aligned.

    With ``contract_due`` recorded, the due state of step ``t`` is scored against the
    state recorded at ``t + 1``; the step after a reset's last state starts the next
    episode and is cut there.
    """
    z = data["skill_state"].astype(float)
    done = data["done"].astype(bool)
    if "contract_due" not in data.files:
        return contact_fidelity(z, done, dt_star, **kwargs)
    due = data["contract_due"][:-1].astype(float)
    body = z[1:]
    cut = done[:-1].copy()
    cut[:-1] |= done[1:-1]
    return contact_fidelity(body, cut, dt_star, asked=due, **kwargs)


__all__ = [
    "LIGHT_LOAD",
    "LOAD_EPSILON",
    "LOAD_MODES",
    "ContactFidelity",
    "capture_fidelity",
    "contact_fidelity",
    "foot_height",
    "resample",
]
