from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from triage_webserver.database.connection import get_db
from triage_webserver.models.data_models import Example
from triage_webserver.schemas.example_schemas import (
    ExampleList,
    ExampleResponse,
    ExampleUpdate,
)

router = APIRouter()


@router.get("/{example_id}", response_model=ExampleResponse)
def get_example(example_id: int, db: Session = Depends(get_db)):
    example = db.query(Example).filter(Example.id == example_id).first()
    if example is None:
        raise HTTPException(status_code=404, detail="Example not found")
    return example


@router.get("/", response_model=ExampleList)
def get_examples(
    dataset_id: int | None = None,
    target_agent: str | None = None,
    special_case: str | None = None,
    turns: int | None = None,
    search: str | None = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    query = db.query(Example)

    if dataset_id is not None:
        query = query.filter(Example.dataset_id == dataset_id)

    if target_agent is not None:
        query = query.filter(Example.target_agent == target_agent)

    if special_case is not None:
        if special_case.lower() == "null":
            query = query.filter(Example.special_case.is_(None))
        else:
            query = query.filter(Example.special_case == special_case)

    if turns is not None:
        query = query.filter(Example.turns == turns)

    if search:
        search_pattern = f"%{search}%"
        query = query.filter(Example.conversation.ilike(search_pattern))

    total = query.count()
    examples = query.order_by(Example.id.desc()).offset(skip).limit(limit).all()

    return {"examples": examples, "total": total}


@router.put("/{example_id}", response_model=ExampleResponse)
def update_example(
    example_id: int, example_data: ExampleUpdate, db: Session = Depends(get_db)
):
    example = db.query(Example).filter(Example.id == example_id).first()
    if example is None:
        raise HTTPException(status_code=404, detail="Example not found")

    for field, value in example_data.model_dump(exclude_unset=True).items():
        setattr(example, field, value)

    db.commit()
    db.refresh(example)
    return example


@router.delete("/{example_id}")
def delete_example(example_id: int, db: Session = Depends(get_db)):
    example = db.query(Example).filter(Example.id == example_id).first()
    if example is None:
        raise HTTPException(status_code=404, detail="Example not found")

    db.delete(example)
    db.commit()

    return {"message": "Example deleted successfully"}
