from pydantic import BaseModel, Field

class FormatAgentSchema(BaseModel):
    """Schema for the output of the format agent."""
    formatted_query: str = Field(..., description="The formatted query string having fixed spelling.")


class ClassifierAgentSchema(BaseModel):
    """Schema for the output of the classifier agent."""
    category: str = Field(..., description="The category of the query.")
    reason: str = Field(..., description="The reason for the classification.")
