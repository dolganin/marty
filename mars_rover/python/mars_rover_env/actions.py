from __future__ import annotations

# Public compact action vocabulary used by all benchmark reference policies.
ACTION_MACROS: tuple[int, ...] = (
    0,
    1,
    1 << 1,
    1 | (1 << 2),
    (1 << 3) | (1 << 6),
    # Accelerate while requesting the next legal gear.  The native gearbox only
    # engages after its normal speed/RPM/cooldown checks, so this is a learnable
    # macro request rather than an unsafe direct gear teleport.
    1 | (1 << 6),
    # The same committed run with physical pitch control.  The policy must choose
    # the landing attitude; these do not alter terrain or touchdown thresholds.
    1 | (1 << 4) | (1 << 6),
    1 | (1 << 5) | (1 << 6),
    (1 << 3) | (1 << 7),
    1 << 9,
    1 | (1 << 4),
    1 | (1 << 5),
    1 << 11,
    1 << 8,
    1 << 10,
)
