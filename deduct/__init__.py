"""DEDUCT: reference-free, byte-exact compression of genome collections.

The public surface is `pipeline.compress`, `pipeline.decompress` and `pipeline.verify`, with
`encode.Config` carrying the codec settings. Everything else is a stage those three call.
"""
from . import align, coders, common, encode, fallback, group, model, normalise, pipeline, separate

__all__ = ["align", "coders", "common", "encode", "fallback", "group", "model",
           "normalise", "pipeline", "separate"]
__version__ = "1.0.0"
