"""The CUAD labels used by this project, spelled exactly as in the CUAD JSON."""

from enum import StrEnum


class CuadLabel(StrEnum):
    GOVERNING_LAW = "Governing Law"
    TERMINATION_FOR_CONVENIENCE = "Termination For Convenience"
    CAP_ON_LIABILITY = "Cap On Liability"
    RENEWAL_TERM = "Renewal Term"
    NOTICE_PERIOD_TO_TERMINATE_RENEWAL = "Notice Period To Terminate Renewal"
