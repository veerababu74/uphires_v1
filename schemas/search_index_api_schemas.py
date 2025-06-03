# Pydantic models for request/response
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class VectorFieldDefinition(BaseModel):
    type: str = "knnVector"
    dimensions: int = 384
    similarity: str = "cosine"


class SearchIndexField(BaseModel):
    field_name: str
    field_definition: VectorFieldDefinition


class SearchIndexDefinition(BaseModel):
    name: str
    dynamic: bool = False
    fields: Dict[str, VectorFieldDefinition]


class CreateSearchIndexRequest(BaseModel):
    name: str
    definition: Dict[str, Any]


class UpdateSearchIndexRequest(BaseModel):
    name: str
    new_definition: Dict[str, Any]


class AddFieldRequest(BaseModel):
    index_name: str
    field_name: str
    field_definition: VectorFieldDefinition


class RemoveFieldRequest(BaseModel):
    index_name: str
    field_name: str


class UpdateFieldRequest(BaseModel):
    index_name: str
    field_name: str
    new_field_definition: VectorFieldDefinition
