import requests
import urllib.parse
import xml.etree.ElementTree as ET
import re

def search_arxiv(query, max_results=5):
    """
    Search arXiv with support for:
    - Boolean operators: AND / OR
    - Date range: submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]
    - Multiple terms
    """
    # Normalize spaces and plus signs
    query = query.replace("+", " ").strip()

    # Split on AND/OR while keeping them in the list
    tokens = re.split(r"\s+(AND|OR)\s+", query, flags=re.IGNORECASE)

    encoded_parts = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue

        if token.upper() in ("AND", "OR"):
            # Keep operators unchanged
            encoded_parts.append(token.upper())
        elif token.lower().startswith("submitteddate:["):
            # Encode the date range safely
            encoded_parts.append(urllib.parse.quote(token, safe="[]:"))
        else:
            # Encode normal search terms
            encoded_parts.append(urllib.parse.quote(token))

    # Join with + (arXiv treats + as space)
    query_encoded = "+".join(encoded_parts)

    url = f"http://export.arxiv.org/api/query?search_query={query_encoded}&start=0&max_results={max_results}"

    response = requests.get(url)

    # Parse XML
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(response.content)

    papers = []
    idx = 0
    for entry in root.findall('atom:entry', namespace):
        authors = [
            author.find('atom:name', namespace).text
            for author in entry.findall('atom:author', namespace)
        ]

        # Find link with rel="alternate"
        link_elem = next((link for link in entry.findall('atom:link', namespace)
                          if link.attrib.get('rel') == 'alternate'), None)

        papers.append({
            'id': idx,
            'title': entry.find('atom:title', namespace).text.strip().replace('\n', ' '),
            'authors': authors,
            'summary': entry.find('atom:summary', namespace).text.strip().replace('\n', ' '),
            'arxiv_id': entry.find('atom:id', namespace).text.split('/abs/')[-1],
            'published_date': entry.find('atom:published', namespace).text,
            'link': link_elem.attrib.get('href') if link_elem is not None else None
        })
        idx += 1

    return papers