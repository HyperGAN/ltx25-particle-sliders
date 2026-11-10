"""A caller's live owner, not an admission issuer or resource manager."""
from dataclasses import dataclass
import os

STORAGE = "LTX25-original48-DiT-CUDA1-CPU16to31-inactive-decoders-storage-v1"
DECODER = "LTX25-original-three-decoders-BF16-CUDA0-v1"


@dataclass(frozen=True)
class OwnedAdmission:
    plan: dict
    owner_receipt: dict
    check_live: object

    def check(self, profile):
        count = 2 if profile == STORAGE else 1 if profile == DECODER else 0
        p, r = self.plan, self.owner_receipt
        if (not count or not isinstance(p, dict) or set(p) != {
                "profile", "CUDA_VISIBLE_DEVICES", "physical_GPU_UUIDs"}
                or p["profile"] != profile or not isinstance(p["CUDA_VISIBLE_DEVICES"], str)
                or len(p["CUDA_VISIBLE_DEVICES"].split(",")) != count
                or len(set(p["CUDA_VISIBLE_DEVICES"].split(","))) != count
                or any(not x or x.strip() != x for x in p["CUDA_VISIBLE_DEVICES"].split(","))
                or not isinstance(p["physical_GPU_UUIDs"], list)
                or len(p["physical_GPU_UUIDs"]) != count
                or any(not isinstance(x, str) or not x.startswith("GPU-") for x in p["physical_GPU_UUIDs"])
                or len(set(p["physical_GPU_UUIDs"])) != count):
            raise ValueError("explicit original execution profile and physical device mapping required")
        if (not isinstance(r, dict) or set(r) != {"path", "bytes", "sha256"}
                or not isinstance(r["path"], str) or not os.path.isabs(r["path"])
                or type(r["bytes"]) is not int or r["bytes"] <= 0
                or not isinstance(r["sha256"], str) or len(r["sha256"]) != 64
                or any(x not in "0123456789abcdef" for x in r["sha256"])
                or not callable(self.check_live)):
            raise ValueError("actual external owner receipt and live callback required")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != p["CUDA_VISIBLE_DEVICES"]:
            raise ValueError("runner's exact admitted visible devices required before framework import")
        if self.check_live() is not True:
            raise RuntimeError("external live ownership check did not affirm admission")


def checked(owner, profile):
    if not isinstance(owner, OwnedAdmission):
        raise ValueError("runner must supply OwnedAdmission; no implicit admission")
    owner.check(profile)
    return lambda: owner.check(profile)
