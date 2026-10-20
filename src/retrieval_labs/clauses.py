"""Clause types the application analyzes, and how they group into report sections.

Independent of any dataset: evaluation code maps these to CUAD labels.
"""

from enum import StrEnum


class ClauseType(StrEnum):
    GOVERNING_LAW = "governing_law"
    TERMINATION_FOR_CONVENIENCE = "termination_for_convenience"
    CAP_ON_LIABILITY = "cap_on_liability"
    RENEWAL_TERM = "renewal_term"
    RENEWAL_NOTICE_PERIOD = "renewal_notice_period"


class ReportSection(StrEnum):
    GOVERNING_LAW = "Governing Law"
    TERMINATION_FOR_CONVENIENCE = "Termination for Convenience"
    LIMITATION_OF_LIABILITY = "Limitation of Liability"
    RENEWAL = "Renewal"


SECTION_OF: dict[ClauseType, ReportSection] = {
    ClauseType.GOVERNING_LAW: ReportSection.GOVERNING_LAW,
    ClauseType.TERMINATION_FOR_CONVENIENCE: ReportSection.TERMINATION_FOR_CONVENIENCE,
    ClauseType.CAP_ON_LIABILITY: ReportSection.LIMITATION_OF_LIABILITY,
    ClauseType.RENEWAL_TERM: ReportSection.RENEWAL,
    ClauseType.RENEWAL_NOTICE_PERIOD: ReportSection.RENEWAL,
}
