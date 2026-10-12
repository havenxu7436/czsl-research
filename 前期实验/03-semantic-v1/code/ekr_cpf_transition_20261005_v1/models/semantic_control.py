"""Only primitive input tables change; native CPF layers/losses are retained."""
from .kcc_cpf import DDM as KCCDDM
from transition.common import load_bank


class DDM(KCCDDM):
    def __init__(self, dset, cfg):
        super().__init__(dset, cfg)
        mode = str(getattr(cfg.MODEL, "semantic_text", "correct"))
        if mode not in {"raw", "unit", "correct", "shuffle"}:
            raise ValueError("Unknown semantic input mode: " + mode)
        if bool(cfg.MODEL.kcc_calib) or float(cfg.MODEL.kcc_lambda) != 0:
            raise ValueError("Transition protocol disables static K calibration")
        if mode == "raw":
            if self.kcc_text:
                raise ValueError("A_raw requires kcc_text=False")
        else:
            if not self.kcc_text:
                raise ValueError("Non-raw input controls require kcc_text=True")
            bank, manifest = load_bank(cfg.MODEL.description_bank, dset.attrs, dset.objs)
            self.register_buffer("semantic_a", bank["a"][mode].float(), persistent=False)
            self.register_buffer("semantic_o", bank["o"][mode].float(), persistent=False)
            self.description_manifest = manifest

    def _kcc_attr_embeddings(self, indices):
        if hasattr(self, "semantic_a"):
            return self.semantic_a.index_select(0, indices)
        return super()._kcc_attr_embeddings(indices)

    def _kcc_obj_embeddings(self, indices):
        if hasattr(self, "semantic_o"):
            return self.semantic_o.index_select(0, indices)
        return super()._kcc_obj_embeddings(indices)
