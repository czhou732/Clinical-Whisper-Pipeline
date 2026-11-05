"""Which latent process drives speech changes in depression?

Model A (timing: psychomotor/processing slowing) versus Model B (reward
reactivity: blunted response to positive content), fit to diarized clinical
interviews and validated against item-level PHQ-8. See ``analysis`` for the
end-to-end run and the pre-registration addendum for the hypotheses.
"""

from research.speech_mechanisms.analysis import analyze, load_scores, load_valence, parameters
from research.speech_mechanisms.model_reactivity import participant_reactivity, reactivity_test
from research.speech_mechanisms.model_timing import fit_exgauss, participant_timing
from research.speech_mechanisms.transcripts import load_daic, load_dir, prompt_inventory
from research.speech_mechanisms.validate import agreement, compare_models, discriminant, icc_2_1

__all__ = [
    "agreement", "analyze", "compare_models", "discriminant", "fit_exgauss", "icc_2_1",
    "load_daic", "load_dir", "load_scores", "load_valence", "parameters",
    "participant_reactivity", "participant_timing", "prompt_inventory", "reactivity_test",
]
