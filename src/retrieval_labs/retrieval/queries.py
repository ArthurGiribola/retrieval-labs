"""One fixed query per clause type, shared by every retrieval method.

Written from the public CUAD category descriptions plus standard contract
vocabulary, never from gold answers. Changes must be justified on the dev split.
BM25 has no stemming, so inflected forms are listed explicitly.
"""

from retrieval_labs.clauses import ClauseType

QUERIES: dict[ClauseType, str] = {
    ClauseType.GOVERNING_LAW: (
        "governing law governed by construed interpreted in accordance with the laws "
        "of the state country jurisdiction"
    ),
    ClauseType.TERMINATION_FOR_CONVENIENCE: (
        "terminate termination terminated for convenience without cause for any reason "
        "or no reason at any time upon prior written notice"
    ),
    ClauseType.CAP_ON_LIABILITY: (
        "limitation of liability limited aggregate total liability shall not exceed "
        "maximum amount cap damages fees paid claims brought within"
    ),
    ClauseType.RENEWAL_TERM: (
        "renewal renew renewed renews automatically successive additional term terms "
        "extend extension extended after the initial term expiration"
    ),
    ClauseType.RENEWAL_NOTICE_PERIOD: (
        "notice of non-renewal nonrenewal not to renew terminate renewal days prior "
        "to the expiration of the then current term written notice"
    ),
}
