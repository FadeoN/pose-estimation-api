from pydantic import BaseModel
from pydantic.networks import HttpUrl


class PredictVideoPoseRequest(BaseModel):
    url: HttpUrl
