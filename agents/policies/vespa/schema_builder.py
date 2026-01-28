from datetime import datetime, timedelta

from vespa.package import (
    Document,
    Field,
    FieldSet,
    RankProfile,
    Schema,
    Validation,
    ValidationID,
)

from libraries.observability.logger import get_console_logger

logger = get_console_logger("vespa_package_generator")


def create_validation_overrides() -> list[Validation]:
    future_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    validations = []

    validations.append(
        Validation(
            validation_id=ValidationID.contentClusterRemoval,
            until=future_date,
            comment="Allow content cluster removal during schema updates",
        )
    )

    validations.append(
        Validation(
            validation_id=ValidationID.redundancyIncrease,
            until=future_date,
            comment="Allow redundancy increase for multi-node deployment",
        )
    )

    logger.info(f"Created validation overrides until {future_date} (tomorrow)")
    return validations


def create_policy_document_schema() -> Schema:
    return Schema(
        name="policy_document",
        document=Document(
            fields=[
                Field(
                    name="id",
                    type="string",
                    indexing=["summary", "index"],
                    match=["word"],
                ),
                Field(
                    name="title",
                    type="string",
                    indexing=["summary", "index"],
                    match=["text"],
                    index="enable-bm25",
                ),
                Field(
                    name="text",
                    type="string",
                    indexing=["summary", "index"],
                    match=["text"],
                    index="enable-bm25",
                ),
                Field(
                    name="category",
                    type="string",
                    indexing=["summary", "attribute"],
                ),
                Field(
                    name="chunk_index",
                    type="int",
                    indexing=["summary", "attribute"],
                ),
                Field(
                    name="source_file",
                    type="string",
                    indexing=["summary", "attribute"],
                ),
                Field(
                    name="page_numbers",
                    type="array<int>",
                    indexing=["summary", "attribute"],
                ),
                Field(
                    name="page_range",
                    type="string",
                    indexing=["summary", "attribute"],
                ),
                Field(
                    name="headings",
                    type="array<string>",
                    indexing=["summary", "attribute"],
                ),
                Field(
                    name="char_count",
                    type="int",
                    indexing=["summary", "attribute"],
                ),
                Field(
                    name="token_count",
                    type="int",
                    indexing=["summary", "attribute"],
                ),
                Field(
                    name="document_id",
                    type="string",
                    indexing=["summary", "attribute"],
                ),
                Field(
                    name="previous_chunk_id",
                    type="string",
                    indexing=["summary", "attribute"],
                ),
                Field(
                    name="next_chunk_id",
                    type="string",
                    indexing=["summary", "attribute"],
                ),
                Field(
                    name="chunk_position",
                    type="float",
                    indexing=["summary", "attribute"],
                ),
                Field(
                    name="section_path",
                    type="array<string>",
                    indexing=["summary", "attribute"],
                ),
                Field(
                    name="embedding",
                    type="tensor<float>(x[384])",
                    indexing=["attribute", "index"],
                    attribute=["distance-metric: angular"],
                ),
            ]
        ),
        fieldsets=[FieldSet(name="default", fields=["title", "text"])],
        rank_profiles=[
            RankProfile(name="default", first_phase="nativeRank(title, text)"),
            RankProfile(
                name="with_position",
                first_phase="nativeRank(title, text) * (1.0 - 0.3 * attribute(chunk_position))",
            ),
            RankProfile(
                name="semantic",
                first_phase="closeness(field, embedding)",
                inputs=[("query(query_embedding)", "tensor<float>(x[384])")],
            ),
            RankProfile(
                name="hybrid",
                first_phase="(1.0 - query(alpha)) * nativeRank(title, text) + query(alpha) * closeness(field, embedding)",
                inputs=[
                    ("query(alpha)", "double", "0.7"),
                    ("query(query_embedding)", "tensor<float>(x[384])"),
                ],
            ),
        ],
    )
