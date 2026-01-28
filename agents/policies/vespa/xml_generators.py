import xml.etree.ElementTree as ET
from xml.dom import minidom


def create_hosts_xml(hosts: list[dict[str, str]]) -> str:
    root = ET.Element("hosts")

    for host in hosts:
        host_elem = ET.SubElement(root, "host", name=host["name"])
        ET.SubElement(host_elem, "alias").text = host["alias"]

    rough_string = ET.tostring(root, encoding="unicode")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")


def create_services_xml(node_count: int = 1, redundancy: int = 1) -> str:
    root = ET.Element("services", version="1.0")

    admin = ET.SubElement(root, "admin", version="2.0")

    if node_count > 1:
        configservers = ET.SubElement(admin, "configservers")
        for i in range(min(3, node_count)):
            ET.SubElement(configservers, "configserver", hostalias=f"node{i}")

        cluster_controllers = ET.SubElement(admin, "cluster-controllers")
        for i in range(min(3, node_count)):
            ET.SubElement(
                cluster_controllers, "cluster-controller", hostalias=f"node{i}"
            )

        slobroks = ET.SubElement(admin, "slobroks")
        for i in range(min(3, node_count)):
            ET.SubElement(slobroks, "slobrok", hostalias=f"node{i}")

    adminserver_node = "node3" if node_count > 3 else "node0"
    ET.SubElement(admin, "adminserver", hostalias=adminserver_node)

    container = ET.SubElement(root, "container", id="policies_container", version="1.0")
    ET.SubElement(container, "search")
    ET.SubElement(container, "document-api")
    ET.SubElement(container, "document-processing")

    if node_count > 1:
        nodes = ET.SubElement(container, "nodes")
        for i in range(node_count):
            ET.SubElement(nodes, "node", hostalias=f"node{i}")

    content = ET.SubElement(root, "content", id="policies_content", version="1.0")
    ET.SubElement(content, "redundancy").text = str(redundancy)

    documents = ET.SubElement(content, "documents")
    ET.SubElement(documents, "document", type="policy_document", mode="index")

    nodes = ET.SubElement(content, "nodes")
    for i in range(node_count):
        ET.SubElement(
            nodes, "node", **{"distribution-key": str(i), "hostalias": f"node{i}"}
        )

    rough_string = ET.tostring(root, encoding="unicode")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")
