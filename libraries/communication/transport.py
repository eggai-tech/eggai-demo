import ssl

from eggai.transport import KafkaTransport
from faststream.kafka import KafkaBroker
from faststream.security import BaseSecurity


def create_kafka_transport(
    bootstrap_servers: str, ssl_cert: str | None = None
) -> KafkaTransport:
    servers_list = bootstrap_servers.split(",")

    if ssl_cert and ssl_cert.strip():
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        ssl_context.load_verify_locations(cadata=ssl_cert)

        broker = KafkaBroker(
            bootstrap_servers=servers_list,
            security=BaseSecurity(use_ssl=True, ssl_context=ssl_context),
        )
    else:
        broker = KafkaBroker(bootstrap_servers=servers_list)

    return KafkaTransport(broker=broker)
