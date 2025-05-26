import math

def get_scale_range(scaled_weights_bits):
    a = 0.0
    b = math.pow(2, scaled_weights_bits) - 1
    return a, b

def scale_weight(min_w, max_w, weight, scaled_weights_bits):
    a, b = get_scale_range(scaled_weights_bits)
    return round(a + ((weight - min_w) * (b - a) / (max_w - min_w)))

def descale_weight(min_w, max_w, weight, scaled_weights_bits):
    a, b = get_scale_range(scaled_weights_bits)
    return min_w + ((weight - a) * (max_w - min_w) / (b - a))

def scale_weights(weights: list, scaled_weights_bits: int):
    min_w = min(weights)
    max_w = max(weights)
    return min_w, max_w, [scale_weight(min_w, max_w, w, scaled_weights_bits) for w in weights]

def descale_weights(weights: list, scaled_weights_bits: int, min_w: float, max_w: float):
    return [descale_weight(min_w, max_w, w, scaled_weights_bits) for w in weights]