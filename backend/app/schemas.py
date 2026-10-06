"""Request and response models: the contract between frontend and backend."""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator


class EmptyParams(BaseModel):
    pass


class BrightnessParams(BaseModel):
    beta: float = Field(ge=-255, le=255, description="Additive offset β, Eq. (8)")


class ContrastParams(BaseModel):
    alpha: float = Field(gt=0, le=5, description="Gain α, Eq. (9)")


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


class MedianParams(BaseModel):
    size: int = Field(ge=3, le=15, description="Window size m = n, Eq. (33)")

    @field_validator("size")
    @classmethod
    def must_be_odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("Median window size must be odd.")
        return v


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
    params: MedianParams


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
    Union[CropOp, ResizeOp, GrayscaleOp, BrightnessOp, ContrastOp, GaussianBlurOp, MedianOp],
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
