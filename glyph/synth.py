"""Placeholder for the synthetic-glyph generator behind the training data, which was not published. data.py imports this
module at load time (evaluate.py imports data.py); scoring never calls it."""


def _missing(*args, **kwargs):
    raise NotImplementedError('x_glyph synth.py (the training-data generator) was not published')


make_text = make_artefact = composite = _missing
