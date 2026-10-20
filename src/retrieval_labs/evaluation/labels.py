"""Mapping between application clause types and CUAD labels (evaluation-side only)."""

from retrieval_labs.clauses import ClauseType
from retrieval_labs.cuad.labels import CuadLabel

CUAD_LABEL_OF: dict[ClauseType, CuadLabel] = {
    ClauseType.GOVERNING_LAW: CuadLabel.GOVERNING_LAW,
    ClauseType.TERMINATION_FOR_CONVENIENCE: CuadLabel.TERMINATION_FOR_CONVENIENCE,
    ClauseType.CAP_ON_LIABILITY: CuadLabel.CAP_ON_LIABILITY,
    ClauseType.RENEWAL_TERM: CuadLabel.RENEWAL_TERM,
    ClauseType.RENEWAL_NOTICE_PERIOD: CuadLabel.NOTICE_PERIOD_TO_TERMINATE_RENEWAL,
}

CLAUSE_TYPE_OF: dict[CuadLabel, ClauseType] = {label: ct for ct, label in CUAD_LABEL_OF.items()}
