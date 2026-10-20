"""CUAD v1 dataset preparation: download, parsing, gold annotations and splits."""


class CuadDataError(ValueError):
    """Raised when CUAD source files or prepared data fail an integrity check."""
