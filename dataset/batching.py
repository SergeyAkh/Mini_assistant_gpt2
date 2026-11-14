"""Length-bucketed batch sampler (plain python, no torch import).

Shuffles the data, sorts inside large chunks by length, cuts batches, then shuffles
the batches. Gives much less padding than random batches (faster on a Mac) while
keeping batches random. Pass it to DataLoader(batch_sampler=...).
"""
import math
import random


class BucketBatchSampler:
    def __init__(self, lengths, batch_size, shuffle=True, bucket_mult=50, seed=0):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_mult = bucket_mult
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(idx)
        chunk = self.batch_size * self.bucket_mult
        batches = []
        for i in range(0, len(idx), chunk):
            part = sorted(idx[i:i + chunk], key=lambda j: self.lengths[j])
            batches += [part[k:k + self.batch_size] for k in range(0, len(part), self.batch_size)]
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)
