"""Resolve the training-only command references used by stationary rewards."""


def racket_reward_command(env, name="racket_target"):
    reference = getattr(env, "_mocap_reward_reference", None)
    if reference is not None and name == "racket_target":
        return reference
    return env.command_manager.get_term(name)


def motion_reward_command(env, name="motion"):
    reference = getattr(env, "_mocap_reward_reference", None)
    if reference is not None and name == "motion":
        return reference.motion_reference
    return env.command_manager.get_term(name)
