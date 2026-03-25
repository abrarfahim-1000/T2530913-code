import grid2op
from lightsim2grid import LightSimBackend

# Local dev — tiny 5-bus env
env = grid2op.make("l2rpn_case14_sandbox", test='true', backend=LightSimBackend())
print(f"Buses: {env.n_sub}, Lines: {env.n_line}")  # → Buses: 14, Lines: 20

# ── Research PC swap ─────────────────────────────────────────────────────────
# env = grid2op.make("l2rpn_neurips_2020_track1_small", backend=LightSimBackend())
# → Buses: 36, Lines: 59