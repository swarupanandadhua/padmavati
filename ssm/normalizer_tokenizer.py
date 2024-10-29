from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

normalized_text = IndicNormalizerFactory().get_normalizer('sa').normalize("श्रीमद्भगवद्गीता अध्यायः १")  ## does it really do anything ???
print("Normalized Text:", normalized_text)

tokens = indic_tokenize.trivial_tokenize(normalized_text, 'sa') ## does it really do anything ???
print("Tokens:", tokens)