"""Sample reviews for ReviewPulse demo."""

import random
from typing import List

POSITIVE_SAMPLES = [
    "This blender is incredible — smoothies in under 30 seconds, easy to clean, and still going strong after six months of daily use.",
    "Absolutely love this book. The writing is sharp, the characters feel real, and I stayed up until 2 AM to finish it. Highly recommended.",
    "Perfect headphones for the price. Sound quality rivals sets twice the cost, and the battery easily lasts two full days.",
    "The kitchen knife set exceeded my expectations. Razor sharp out of the box, balanced grip, and the block keeps everything organised.",
    "Bought this for my daughter's birthday and she hasn't put it down. Great educational toy that actually holds a child's attention.",
    "The mattress is incredibly comfortable. I wake up refreshed every morning with no back pain. Best purchase in years.",
]

NEGATIVE_SAMPLES = [
    "Arrived with a cracked screen and the seller took three weeks to respond. Complete waste of money — avoid.",
    "The straps broke on the second use. Cheap stitching, flimsy buckles. Returned immediately and won't be buying from this brand again.",
    "Sound cuts out every few minutes. I thought it was a pairing issue but the replacement unit had the same problem. Terrible quality control.",
    "This DVD player skips on every disc I try. The remote is unresponsive half the time. Returned after one day.",
    "Poorly written instructions, missing hardware, and customer support just copy-pasted the same unhelpful reply three times. Deeply frustrating.",
    "The product stopped working after two weeks. Customer service is impossible to reach. Total disappointment.",
]

SAMPLES: List[str] = POSITIVE_SAMPLES + NEGATIVE_SAMPLES


def get_random_sample(current_text: str = "") -> str:
    """Return a random sample different from the current text, if possible."""
    if not SAMPLES:
        return "No samples available."

    candidates = [sample for sample in SAMPLES if sample != current_text.strip()]
    if not candidates:
        return random.choice(SAMPLES)

    return random.choice(candidates)


def get_all_samples() -> List[str]:
    """Return all available sample reviews."""
    return SAMPLES.copy()


def get_positive_samples() -> List[str]:
    """Return only positive samples."""
    return POSITIVE_SAMPLES.copy()


def get_negative_samples() -> List[str]:
    """Return only negative samples."""
    return NEGATIVE_SAMPLES.copy()
