from typing import List

from fastapi import APIRouter
from starlette import status


from application.command import predict_video_pose_command_handler
from application.command.model.frame_pose_dto import VideoKeypointDTO
from application.command.predict_video_pose_command import PredictVideoPoseCommand
from infrastructure.controller.request.predict_video_pose_request import PredictVideoPoseRequest

router = APIRouter()


@router.post("/video", status_code=status.HTTP_200_OK)
async def predict_video_pose(request: PredictVideoPoseRequest) -> VideoKeypointDTO:
    try:
        return await predict_video_pose_command_handler.handle(PredictVideoPoseCommand(
            url=request.url,
        ))
    except Exception as e:
        raise e


