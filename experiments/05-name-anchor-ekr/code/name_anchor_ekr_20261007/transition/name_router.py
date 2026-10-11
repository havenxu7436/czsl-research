"""Name-only object context; same registered capacity for clean/name/shuffle."""
import torch
from .router import Router


class NameRouter(Router):
    def __init__(self, bank, mode='ekr', rank=32, eta_max=0.1,
                 temperature=0.2, object_chunk=16):
        if mode not in {'name_only', 'ekr', 'shuffle', 'no_object', 'no_null'}:
            raise ValueError('Not a declared name-anchor exploration variant')
        super().__init__(bank, 'ekr' if mode == 'name_only' else mode,
                         rank, eta_max, temperature, object_chunk)
        self.mode = mode
        # OE descriptions must not leak into the attribution control or object query.
        self.object_source = bank['o']['unit'].float().clone()
        if mode == 'name_only':
            # Repeat the name to keep four attention slots and the same parameter count.
            # These are intentionally identical slots; they contain no description text.
            names = bank['a']['unit'].float()[:, None, :].expand(-1, 4, -1).clone()
            self.key_source = names
            self.value_source = names / 2
            self.slot_mask = torch.ones(names.shape[:2], dtype=torch.bool)

