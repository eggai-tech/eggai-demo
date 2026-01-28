#!/usr/bin/env python3

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from vespa.package import ApplicationPackage

from libraries.observability.logger import get_console_logger

from .schema_builder import create_policy_document_schema, create_validation_overrides
from .xml_generators import create_hosts_xml, create_services_xml

logger = get_console_logger("vespa_package_generator")


def create_application_package(app_name: str = "policies") -> ApplicationPackage:
    logger.info(f"Creating enhanced Vespa application package with name: {app_name}")

    schema = create_policy_document_schema()
    validations = create_validation_overrides()

    app_package = ApplicationPackage(
        name=app_name, schema=[schema], validations=validations
    )

    logger.info("Enhanced application package created successfully")
    return app_package


def save_package_to_zip(
    app_package: ApplicationPackage,
    output_path: Path,
    deployment_mode: str = "local",
    node_count: int = 1,
    hosts: list[dict[str, str]] | None = None,
    services_xml: Path | None = None,
) -> Path:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        logger.info("Saving application package to temporary directory")
        app_package.to_files(temp_path)

        if deployment_mode == "production":
            redundancy = min(node_count, 2)

            if services_xml and services_xml.exists():
                logger.info(f"Using provided services.xml from {services_xml}")
                shutil.copy(services_xml, temp_path / "services.xml")
            else:
                logger.info(
                    f"Creating custom services.xml for {node_count} nodes with redundancy {redundancy}"
                )
                services_xml_content = create_services_xml(node_count, redundancy)
                services_path = temp_path / "services.xml"
                services_path.write_text(services_xml_content)

            if hosts:
                hosts_xml = create_hosts_xml(hosts)
                hosts_path = temp_path / "hosts.xml"
                hosts_path.write_text(hosts_xml)
                logger.info(f"Created hosts.xml with {len(hosts)} host definitions")

        zip_path = output_path / "vespa-application.zip"
        logger.info(f"Creating zip file at {zip_path}")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in temp_path.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(temp_path)
                    zipf.write(file_path, arcname)

        logger.info(f"Application package saved to {zip_path}")
        return zip_path


def save_package_metadata(output_path: Path, schema_info: dict) -> Path:
    metadata_path = output_path / "package-metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(schema_info, f, indent=2)

    logger.info(f"Package metadata saved to {metadata_path}")
    return metadata_path


def generate_package_artifacts(
    output_dir: Path | None = None,
    deployment_mode: str = "local",
    node_count: int = 1,
    hosts: list[dict[str, str]] | None = None,
    services_xml: Path | None = None,
    app_name: str = "policies",
) -> tuple[Path, Path]:
    if output_dir is None:
        output_dir = Path(__file__).parent / "artifacts"

    output_dir.mkdir(parents=True, exist_ok=True)

    app_package = create_application_package(app_name=app_name)

    zip_path = save_package_to_zip(
        app_package, output_dir, deployment_mode, node_count, hosts, services_xml
    )

    schema_info = {
        "name": app_name,
        "generated_at": datetime.now().isoformat(),
        "deployment": {
            "mode": deployment_mode,
            "node_count": node_count,
            "hosts": hosts or [],
        },
        "schema": {
            "name": "policy_document",
            "fields": {
                "core": [
                    "id",
                    "title",
                    "text",
                    "category",
                    "chunk_index",
                    "source_file",
                ],
                "metadata": [
                    "page_numbers",
                    "page_range",
                    "headings",
                    "char_count",
                    "token_count",
                ],
                "relationships": [
                    "document_id",
                    "previous_chunk_id",
                    "next_chunk_id",
                    "chunk_position",
                ],
                "context": ["section_path"],
                "vector": ["embedding (384 dimensions)"],
            },
            "rank_profiles": ["default", "with_position", "semantic", "hybrid"],
            "indexing": {
                "bm25_enabled": ["title", "text"],
                "vector_search": "embedding field with angular distance metric",
            },
        },
    }

    metadata_path = save_package_metadata(output_dir, schema_info)

    return zip_path, metadata_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Vespa application package for policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save package artifacts (default: ./artifacts)",
    )

    parser.add_argument(
        "--deployment-mode",
        choices=["local", "production"],
        default="local",
        help="Deployment mode: local (single-node) or production (multi-node)",
    )

    parser.add_argument(
        "--node-count",
        type=int,
        default=3,
        help="Number of nodes for production deployment (default: 3)",
    )

    parser.add_argument(
        "--hosts-config",
        type=Path,
        help="JSON file with host configurations for production deployment",
    )

    parser.add_argument(
        "--services-xml",
        type=Path,
        help="XML file with services configurations for production deployment",
    )

    args = parser.parse_args()

    hosts = None
    if args.deployment_mode == "production" and args.hosts_config:
        if args.hosts_config.exists():
            with open(args.hosts_config) as f:
                hosts = json.load(f)
        else:
            hosts = []
            for i in range(args.node_count):
                hosts.append(
                    {
                        "name": f"vespa-node-{i}.vespa-internal.svc.cluster.local",
                        "alias": f"node{i}",
                    }
                )
            logger.info(
                f"Generated default host configuration for {args.node_count} nodes"
            )

    print("🚀 Vespa Package Generator")
    print("=" * 50)

    try:
        zip_path, metadata_path = generate_package_artifacts(
            args.output_dir,
            deployment_mode=args.deployment_mode,
            node_count=args.node_count,
            hosts=hosts,
            services_xml=args.services_xml,
        )

        print()
        print("🎉 Package generation completed successfully!")
        print(f"   Package ZIP: {zip_path}")
        print(f"   Metadata: {metadata_path}")
        print(f"   Deployment mode: {args.deployment_mode}")
        if args.deployment_mode == "production":
            print(f"   Node count: {args.node_count}")
            print(f"   Redundancy: {min(args.node_count, 2)}")
        print()
        print("   Schema details:")
        print("   - Name: policy_document")
        print("   - Core fields: id, title, text, category, chunk_index, source_file")
        print("   - Metadata fields: page numbers, headings, char/token counts")
        print("   - Relationship tracking: previous/next chunks, document associations")
        print("   - Vector search: 384-dimensional embeddings")
        print("   - Rank profiles: default, with_position, semantic, hybrid")

    except Exception as e:
        logger.error(f"Failed to generate package: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
