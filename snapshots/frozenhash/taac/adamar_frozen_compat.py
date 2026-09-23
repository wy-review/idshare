"""Backfill the read-only quantizer accessor absent from the older KuaiRand API."""


def install():
    from recscale.utils import zero_anchor_regularization as original

    if hasattr(original, "get_zero_anchor_identity_quantizer"):
        return False

    def get_zero_anchor_identity_quantizer(model):
        raw_model = original._raw_model(model)
        nested = getattr(getattr(raw_model, "encoder", None), "zero_anchor_identity_quantizer", None)
        direct = getattr(raw_model, "zero_anchor_identity_quantizer", None)
        if nested is not None and direct is not None and nested is not direct:
            raise ValueError("model exposes two different zero-anchor identity quantizers")
        return nested if nested is not None else direct

    original.get_zero_anchor_identity_quantizer = get_zero_anchor_identity_quantizer
    return True
