"""Request and response models: the contract between frontend and backend."""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class EmptyParams(BaseModel):
    pass


class BrightnessParams(BaseModel):
    beta: float = Field(ge=-255, le=255, description="Additive offset β, Eq. (8)")


class ContrastParams(BaseModel):
    r_min: float = Field(ge=0, le=254, description="Input level mapped to 0")
    r_max: float = Field(ge=1, le=255, description="Input level mapped to L-1 = 255")

    @model_validator(mode="after")
    def min_below_max(self):
        if self.r_min >= self.r_max:
            raise ValueError("r_min must be smaller than r_max.")
        return self


class CropParams(BaseModel):
    x: int = Field(ge=0, description="Left edge x_c, Eq. (4)")
    y: int = Field(ge=0, description="Top edge y_c, Eq. (4)")
    width: int = Field(ge=1, description="Output width M'")
    height: int = Field(ge=1, description="Output height N'")


class ResizeParams(BaseModel):
    width: int = Field(ge=1, le=4000)
    height: int = Field(ge=1, le=4000)
    method: Literal["nearest", "bilinear"] = "bilinear"


class GaussianBlurParams(BaseModel):
    sigma: float = Field(ge=0.3, le=15, description="Blur scale sigma_b, Eq. (10)")


class WindowParams(BaseModel):
    """Spatial filters over an m x n = size x size window S_xy (Section 5.4)."""

    size: int = Field(ge=3, le=15, description="Window size m = n")

    @field_validator("size")
    @classmethod
    def must_be_odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("Window size must be odd.")
        return v


# Noise is random, and /process re-runs the whole list on every change, so a
# fixed seed keeps the same noise pattern across previews, undo/redo and
# /export. Send a new seed to draw a new pattern.
Seed = Field(default=0, ge=0, le=2**32 - 1, description="Random seed")


class SaltPepperParams(BaseModel):
    p_pepper: float = Field(ge=0, le=1, description="Pepper probability P_p, Eq. (18)")
    p_salt: float = Field(ge=0, le=1, description="Salt probability P_s, Eq. (18)")
    seed: int = Seed

    @model_validator(mode="after")
    def probabilities_sum_at_most_one(self):
        if self.p_pepper + self.p_salt > 1:
            raise ValueError("P_p + P_s must not exceed 1.")
        return self


class QuantizationParams(BaseModel):
    bits: int = Field(ge=1, le=8, description="Bit depth B (2^B levels), Eq. (22)")


class RayleighParams(BaseModel):
    a: float = Field(ge=-255, le=255, description="Location a_R, Eq. (24)")
    b: float = Field(gt=0, le=65025, description="Spread b_R = 2 sigma_R^2, Eq. (24)")
    zero_mean: bool = Field(
        default=False,
        description="Subtract the mean a_R + sqrt(pi b_R / 4); a_R then has no effect",
    )
    seed: int = Seed


class UniformParams(BaseModel):
    a: float = Field(ge=-255, le=255, description="Lower bound a_U, Eq. (25)")
    b: float = Field(ge=-255, le=255, description="Upper bound b_U, Eq. (25)")
    seed: int = Seed

    @model_validator(mode="after")
    def lower_below_upper(self):
        if self.a >= self.b:
            raise ValueError("a_U must be smaller than b_U.")
        return self


class CropOp(BaseModel):
    type: Literal["crop"]
    params: CropParams


class ResizeOp(BaseModel):
    type: Literal["resize"]
    params: ResizeParams


class GaussianBlurOp(BaseModel):
    type: Literal["gaussian_blur"]
    params: GaussianBlurParams


class MedianOp(BaseModel):
    type: Literal["median"]
    params: WindowParams


class ArithmeticMeanOp(BaseModel):
    type: Literal["arithmetic_mean"]
    params: WindowParams


class GeometricMeanOp(BaseModel):
    type: Literal["geometric_mean"]
    params: WindowParams


class HarmonicMeanOp(BaseModel):
    type: Literal["harmonic_mean"]
    params: WindowParams


class SaltPepperNoiseOp(BaseModel):
    type: Literal["salt_pepper_noise"]
    params: SaltPepperParams


class QuantizationNoiseOp(BaseModel):
    type: Literal["quantization_noise"]
    params: QuantizationParams


class RayleighNoiseOp(BaseModel):
    type: Literal["rayleigh_noise"]
    params: RayleighParams


class UniformNoiseOp(BaseModel):
    type: Literal["uniform_noise"]
    params: UniformParams


class GrayscaleOp(BaseModel):
    type: Literal["grayscale"]
    params: EmptyParams | None = None


class BrightnessOp(BaseModel):
    type: Literal["brightness"]
    params: BrightnessParams


class ContrastOp(BaseModel):
    type: Literal["contrast"]
    params: ContrastParams


Operation = Annotated[
    Union[
        CropOp, ResizeOp, GrayscaleOp, BrightnessOp, ContrastOp,
        GaussianBlurOp, MedianOp, ArithmeticMeanOp, GeometricMeanOp, HarmonicMeanOp,
        SaltPepperNoiseOp, QuantizationNoiseOp, RayleighNoiseOp, UniformNoiseOp,
    ],
    Field(discriminator="type"),
]


class ProcessRequest(BaseModel):
    image_id: str
    operations: list[Operation] = Field(default_factory=list, max_length=50)


class UploadResponse(BaseModel):
    image_id: str
    width: int
    height: int
    scaled: bool
    source_format: str


class ExportRequest(ProcessRequest):
    format: Literal["png", "jpeg", "tiff"] = "png"
    quality: int = Field(default=92, ge=1, le=100, description="JPEG quality (ignored for PNG/TIFF)")
