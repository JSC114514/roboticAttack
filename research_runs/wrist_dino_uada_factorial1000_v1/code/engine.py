"""Dispatch the byte-identical, frozen parent engine for the explicit arm."""
from common import arm_info
if arm_info()["protocol"] == "legacy":
    from engine_legacy import OFTEngine
else:
    from engine_original import OFTEngine
