"""Pulse boundaries and matched outcomes, without model/simulator imports."""
def conditions(config):
    return [dict(name=f'{mode}_{duration:02d}', mode=mode, duration=duration)
            for duration in config['durations'] for mode in config['modes']]


def contact_onset(contact, chunk=8):
    if contact is None:
        return None, 'NO_TARGET_CONTACT'
    if contact <= 0:
        return None, 'CONTACT_AT_RESET'
    # Contact is measured AFTER control step c; the decision must precede it.
    return ((int(contact) - 1) // chunk) * chunk, 'ELIGIBLE'


def pulse_active(step, onset, duration):
    return onset <= step < onset + duration


def pulse_outcome(success, terminal_step, onset, duration, limit):
    return dict(success=int(success), nominal_removal_step=onset + duration,
                pulse_window_steps=max(0, min(terminal_step, onset + duration) - onset),
                removal_reached=terminal_step >= onset + duration,
                post_removal_steps=max(0, terminal_step - onset - duration),
                terminal_phase='pulse' if terminal_step <= onset + duration else 'clean_continuation',
                remaining_budget_at_onset=limit-onset,
                remaining_budget_at_nominal_removal=max(0, limit-onset-duration))
