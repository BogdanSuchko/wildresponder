from pydantic import BaseModel
from typing import Optional, List

class GenerateResponsePayload(BaseModel):
    id: str
    text: str
    prompt: Optional[str] = None
    rating: Optional[int] = None
    force: bool = False
    productName: Optional[str] = None
    advantages: Optional[List[str]] = None
    pluses: Optional[str] = None
    minuses: Optional[str] = None

class ReplyPayload(BaseModel):
    id: str
    type: str
    text: Optional[str] = None
    answer: Optional[dict] = None
    state: Optional[str] = None

class ProductDetails(BaseModel):
    nmId: int
    productName: str
    photo: Optional[str] = None

class Feedback(BaseModel):
    id: str
    text: Optional[str] = ""
    productValuation: int
    createdDate: str
    productDetails: ProductDetails
    pluses: Optional[str] = None
    minuses: Optional[str] = None
    userName: str
    advantages: Optional[List[str]] = None

class Question(BaseModel):
    id: str
    text: str
    createdDate: str
    productDetails: ProductDetails
    # No 'answer' field here because we only fetch unanswered items