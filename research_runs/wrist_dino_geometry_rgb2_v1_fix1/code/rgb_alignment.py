"""Cross-render RGB is diagnostic when exact sampled-state provenance is proved."""
from functools import wraps
from rgb_common import Camera as NativeCamera
from oracle_camera import fingerprint

class Camera(NativeCamera):
    def __enter__(self):
        super().__enter__()
        inner = self.observable._sensor
        @wraps(inner)
        def sampled(cache):
            pixels = inner(cache)
            if self.key is not None:
                # Still inside the native _sensor call: Observable has not advanced
                # its clocks. NativeCamera already proved render before==after.
                self.latest['sample_fingerprint'] = fingerprint(self.env)
            return pixels
        self.observable._sensor = sampled
        return self

def assess(error, current, reference):
    """Never increase either existing numeric tolerance or allow physical drift."""
    legacy_rgb = max(error['full_image'], error['wrist_image']) <= 1e-4
    proprio = error['proprio_max_abs'] <= 1e-6
    arrays = sorted(set((current or {}).get('arrays', {})) | set((reference or {}).get('arrays', {})))
    differing = [key for key in arrays if (current or {}).get('arrays', {}).get(key) != (reference or {}).get('arrays', {}).get(key)]
    required = {'sim_state','body_xpos','body_xquat','geom_xpos','geom_xmat','site_xpos','site_xmat','cam_xpos','cam_xmat'}
    fingerprints_present = bool(current and reference and required <= set(current.get('arrays', {}))
                                and required <= set(reference.get('arrays', {})))
    time_equal = bool(current and reference and 'time' in current and 'time' in reference and current['time'] == reference['time'])
    clocks_equal = bool(current and reference and 'observable_clocks' in current and 'observable_clocks' in reference
                        and current['observable_clocks'] == reference['observable_clocks'])
    exact = fingerprints_present and not differing and time_equal and clocks_equal
    accepted = proprio and (legacy_rgb or exact)
    return {'accepted': accepted, 'accepted_by': 'legacy_rgb_alignment' if accepted and legacy_rgb else
            'exact_sampled_state' if accepted else None, 'historical_rgb_within_old_tolerance': legacy_rgb,
            'proprio_within_old_tolerance': proprio, 'exact_sampled_state': exact,
            'differing_arrays': differing, 'sample_time_equal': time_equal, 'observable_clocks_equal': clocks_equal}
