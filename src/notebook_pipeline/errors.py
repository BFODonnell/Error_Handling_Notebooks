"""Actionable domain errors, separate from infrastructure/programming failures."""


class PipelineError(Exception):
    """Base for failures callers may explicitly handle."""


class DataContractError(PipelineError):
    """Input is readable but does not satisfy the analysis contract."""


class ArtifactError(PipelineError):
    """A required input/output cannot be read, verified, or written."""


class RunStateError(PipelineError):
    """The requested operation violates notebook execution order."""
