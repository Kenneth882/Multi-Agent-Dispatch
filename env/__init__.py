# SP1 compatibility: RideShareEnv is importable as env.RideShareEnv but is
# loaded lazily so that SP2 code (env.matching_env) does not force PettingZoo,
# Gymnasium, or stable-baselines3 to load when they are not needed.

def __getattr__(name):
    if name == "RideShareEnv":
        from env.ride_share_env import RideShareEnv
        return RideShareEnv
    raise AttributeError(f"module 'env' has no attribute {name!r}")
