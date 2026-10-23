from dataclasses import dataclass

from fastapi import Request

from app.core.config import Settings
from app.services.chat_service import ChatService
from app.services.ingestion_service import IngestionService
from app.services.providers.ollama import OllamaClient


@dataclass
class Container:
    settings: Settings
    ollama: OllamaClient
    chat: ChatService
    ingestion: IngestionService
    session_factory: object
    embedder: object = None
    ocr: object = None


def get_container(request: Request) -> Container:
    return request.app.state.container
