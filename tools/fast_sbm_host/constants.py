"""Table-loader dimensions and refusal, from WRF v4.8.0 FAST_SBM.

WRF source SHA256 fd72158202a98d4ad618ebb608eeadffd50852a9875f8bfd9bb8ad63c15f1a2e.
NCAR/UCAR public domain; see gpuwm/data/fast_sbm/LICENSE-WRF.txt and NOTICE.
NKR, ICEMAX and JBREAK are the bin counts at WRF :3894-3902.
"""

NKR = 33

ICEMAX = 3

JBREAK = 18

class FsbmRefusal(RuntimeError):
    """A WRF fatal check, or an input the reference cannot honour.

    ``where`` names the WRF routine (and line) or the port function that
    refused; ``condition`` states what was found.  WRF's ``wrf_error_fatal``
    calls on the transcribed paths become this exception.
    """

    def __init__(self, where: str, condition: str) -> None:
        self.where = where
        self.condition = condition
        super().__init__(f"{where}: {condition}")
