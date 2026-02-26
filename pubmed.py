"""PubMed search and paper download module.

Search PubMed by topic/keywords, fetch metadata, and download
full-text PDFs (via PMC Open Access) or save abstracts for ingestion.
"""

from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from Bio import Entrez, Medline

from config import get_settings


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class PubMedArticle:
    """Parsed PubMed article metadata."""

    pmid: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    doi: str = ""
    journal: str = ""
    year: str = ""
    keywords: list[str] = field(default_factory=list)
    pmc_id: str = ""  # PMC ID for open access full text
    mesh_terms: list[str] = field(default_factory=list)

    @property
    def has_fulltext(self) -> bool:
        return bool(self.pmc_id)


# ============================================================================
# Entrez Setup
# ============================================================================

def _setup_entrez(email: str | None = None, api_key: str | None = None) -> None:
    """Configure Entrez with credentials."""
    Entrez.email = email or os.getenv("NCBI_EMAIL", os.getenv("EMAIL", "user@example.com"))
    api = api_key or os.getenv("NCBI_API_KEY", "")
    if api:
        Entrez.api_key = api


# ============================================================================
# PubMed Search
# ============================================================================

def search_pubmed(
    query: str,
    max_results: int = 20,
    sort: str = "relevance",
    min_date: str = "",
    max_date: str = "",
    email: str | None = None,
    api_key: str | None = None,
) -> list[str]:
    """
    Search PubMed and return a list of PMIDs.

    Args:
        query: Search query (PubMed syntax supported)
               e.g., "single-cell RNA-seq B cell germinal center"
        max_results: Maximum number of results. 0 = fetch ALL results.
        sort: Sort order ('relevance', 'pub_date', 'first_author')
        min_date: Minimum publication date (YYYY/MM/DD)
        max_date: Maximum publication date (YYYY/MM/DD)

    Returns:
        List of PMID strings
    """
    _setup_entrez(email, api_key)

    # First search to get total count and history
    search_params: dict[str, Any] = {
        "db": "pubmed",
        "term": query,
        "retmax": 0,  # Just get count first
        "sort": sort,
        "usehistory": "y",
    }
    if min_date:
        search_params["mindate"] = min_date
        search_params["datetype"] = "pdat"
    if max_date:
        search_params["maxdate"] = max_date
        search_params["datetype"] = "pdat"

    handle = Entrez.esearch(**search_params)
    results = Entrez.read(handle)
    handle.close()

    total = int(results.get("Count", "0"))
    webenv = results.get("WebEnv", "")
    query_key = results.get("QueryKey", "")

    fetch_count = total if max_results == 0 else min(max_results, total)
    print(f"  Found {total} results, fetching {fetch_count}")

    if fetch_count == 0:
        return []

    # Fetch PMIDs in batches using history server
    pmids: list[str] = []
    batch_size = 500

    for start in range(0, fetch_count, batch_size):
        retmax = min(batch_size, fetch_count - start)
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retstart=start,
            retmax=retmax,
            sort=sort,
            webenv=webenv,
            query_key=query_key,
            usehistory="y",
        )
        batch_results = Entrez.read(handle)
        handle.close()

        pmids.extend(batch_results.get("IdList", []))
        time.sleep(0.34)  # Rate limit

        if len(pmids) % 1000 == 0 and len(pmids) > 0:
            print(f"    ... fetched {len(pmids)}/{fetch_count} PMIDs")

    return pmids[:fetch_count]


# ============================================================================
# Fetch Article Metadata
# ============================================================================

def fetch_articles(
    pmids: list[str],
    email: str | None = None,
    api_key: str | None = None,
) -> list[PubMedArticle]:
    """
    Fetch detailed metadata for a list of PMIDs.

    Returns:
        List of PubMedArticle objects
    """
    if not pmids:
        return []

    _setup_entrez(email, api_key)
    articles = []

    # Fetch in batches of 50
    batch_size = 50
    for start in range(0, len(pmids), batch_size):
        batch = pmids[start:start + batch_size]

        # Use efetch with XML for rich metadata
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(batch),
            rettype="xml",
            retmode="xml",
        )
        xml_data = handle.read()
        handle.close()

        # Parse XML
        articles.extend(_parse_pubmed_xml(xml_data))

        # Rate limit
        time.sleep(0.34)

    return articles


def _parse_pubmed_xml(xml_data: str | bytes) -> list[PubMedArticle]:
    """Parse PubMed XML response into PubMedArticle objects."""
    if isinstance(xml_data, str):
        xml_data = xml_data.encode("utf-8")

    articles = []
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError:
        return articles

    for article_elem in root.findall(".//PubmedArticle"):
        try:
            articles.append(_parse_single_article(article_elem))
        except Exception as e:
            print(f"  [WARN] Failed to parse article: {e}")

    return articles


def _parse_single_article(elem: ET.Element) -> PubMedArticle:
    """Parse a single PubmedArticle XML element."""
    medline = elem.find(".//MedlineCitation")
    article = medline.find(".//Article") if medline is not None else None

    # PMID
    pmid_elem = medline.find(".//PMID") if medline is not None else None
    pmid = pmid_elem.text if pmid_elem is not None else ""

    # Title
    title = ""
    title_elem = article.find(".//ArticleTitle") if article is not None else None
    if title_elem is not None:
        title = "".join(title_elem.itertext()).strip()

    # Authors
    authors = []
    if article is not None:
        for author in article.findall(".//Author"):
            last = author.find("LastName")
            fore = author.find("ForeName")
            if last is not None and fore is not None:
                authors.append(f"{fore.text} {last.text}")
            elif last is not None:
                authors.append(last.text)

    # Abstract
    abstract = ""
    if article is not None:
        abs_texts = article.findall(".//AbstractText")
        abstract_parts = []
        for abs_part in abs_texts:
            label = abs_part.get("Label", "")
            text = "".join(abs_part.itertext()).strip()
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = "\n".join(abstract_parts)

    # DOI
    doi = ""
    if article is not None:
        for eid in article.findall(".//ELocationID"):
            if eid.get("EIdType") == "doi":
                doi = eid.text or ""
                break
    # Also check ArticleIdList
    if not doi:
        pub_data = elem.find(".//PubmedData")
        if pub_data is not None:
            for aid in pub_data.findall(".//ArticleId"):
                if aid.get("IdType") == "doi":
                    doi = aid.text or ""
                    break

    # Journal
    journal = ""
    journal_elem = article.find(".//Journal/Title") if article is not None else None
    if journal_elem is not None:
        journal = journal_elem.text or ""

    # Year
    year = ""
    if article is not None:
        year_elem = article.find(".//Journal/JournalIssue/PubDate/Year")
        if year_elem is not None:
            year = year_elem.text or ""
        else:
            medline_date = article.find(".//Journal/JournalIssue/PubDate/MedlineDate")
            if medline_date is not None and medline_date.text:
                year = medline_date.text[:4]

    # Keywords
    keywords = []
    if medline is not None:
        for kw in medline.findall(".//KeywordList/Keyword"):
            if kw.text:
                keywords.append(kw.text)

    # MeSH terms
    mesh_terms = []
    if medline is not None:
        for mesh in medline.findall(".//MeshHeadingList/MeshHeading/DescriptorName"):
            if mesh.text:
                mesh_terms.append(mesh.text)

    # PMC ID
    pmc_id = ""
    pub_data = elem.find(".//PubmedData")
    if pub_data is not None:
        for aid in pub_data.findall(".//ArticleId"):
            if aid.get("IdType") == "pmc":
                pmc_id = aid.text or ""
                break

    return PubMedArticle(
        pmid=pmid,
        title=title,
        authors=authors,
        abstract=abstract,
        doi=doi,
        journal=journal,
        year=year,
        keywords=keywords,
        pmc_id=pmc_id,
        mesh_terms=mesh_terms,
    )


# ============================================================================
# Download Full-Text PDF
# ============================================================================

def download_pdf(
    article: PubMedArticle,
    output_dir: str | Path,
    email: str | None = None,
) -> Path | None:
    """
    Try to download full-text PDF for an article.

    Strategy:
    1. PMC Open Access (if PMC ID available)
    2. Unpaywall (if DOI available)
    3. Publisher direct via DOI (works on institutional IP)
    4. Fallback: save abstract as text

    Returns:
        Path to downloaded file, or None if all methods fail
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_title = re.sub(r'[^\w\s-]', '', article.title)[:60].strip()
    base_name = f"PMID{article.pmid}_{safe_title}".replace(" ", "_")

    # --- Strategy 1: PMC Open Access PDF ---
    if article.pmc_id:
        pdf_path = output_dir / f"{base_name}.pdf"
        if _download_from_pmc(article.pmc_id, pdf_path):
            return pdf_path

    # --- Strategy 2: Unpaywall ---
    if article.doi:
        pdf_path = output_dir / f"{base_name}.pdf"
        unpaywall_email = email or os.getenv("NCBI_EMAIL", os.getenv("EMAIL", ""))
        if unpaywall_email and _download_from_unpaywall(article.doi, pdf_path, unpaywall_email):
            return pdf_path

    # --- Strategy 3: Publisher direct (institutional IP) ---
    if article.doi:
        pdf_path = output_dir / f"{base_name}.pdf"
        if _download_from_publisher(article.doi, pdf_path):
            return pdf_path

    # --- Fallback: Save abstract as markdown for parsing ---
    if article.abstract:
        md_path = output_dir / f"{base_name}.md"
        _save_as_markdown(article, md_path)
        return md_path

    return None


def _download_from_pmc(pmc_id: str, output_path: Path) -> bool:
    """Download PDF from PubMed Central Open Access."""
    # Clean PMC ID
    pmc_num = pmc_id.replace("PMC", "")
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_num}/pdf/"

    try:
        resp = requests.get(url, timeout=30, allow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (PaperGraphRAG; mailto:user@example.com)"
        })
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("application/pdf"):
            output_path.write_bytes(resp.content)
            return True
    except Exception as e:
        print(f"    [WARN] PMC download failed: {e}")

    return False


def _download_from_unpaywall(doi: str, output_path: Path, email: str) -> bool:
    """Download PDF via Unpaywall API."""
    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"

    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return False

        data = resp.json()
        best_oa = data.get("best_oa_location") or {}
        pdf_url = best_oa.get("url_for_pdf") or best_oa.get("url")

        if not pdf_url:
            return False

        pdf_resp = requests.get(pdf_url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (PaperGraphRAG)"
        })
        if pdf_resp.status_code == 200 and len(pdf_resp.content) > 1000:
            output_path.write_bytes(pdf_resp.content)
            return True
    except Exception as e:
        print(f"    [WARN] Unpaywall download failed: {e}")

    return False


def _download_from_publisher(doi: str, output_path: Path) -> bool:
    """Download PDF directly from publisher via DOI resolution.

    Works when running on institutional IP that has journal access.
    Resolves DOI to publisher page and tries to find/download the PDF.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        # Step 1: Resolve DOI to publisher URL
        doi_url = f"https://doi.org/{doi}"
        resp = requests.get(doi_url, headers=headers, timeout=30, allow_redirects=True)
        if resp.status_code != 200:
            return False

        final_url = resp.url
        page_html = resp.text

        # Step 2: Try known publisher PDF URL patterns
        pdf_url = _resolve_publisher_pdf_url(final_url, page_html, doi)
        if not pdf_url:
            return False

        # Step 3: Download PDF
        pdf_resp = requests.get(pdf_url, headers=headers, timeout=60, allow_redirects=True)
        content_type = pdf_resp.headers.get("content-type", "")

        if (pdf_resp.status_code == 200
                and len(pdf_resp.content) > 5000
                and ("application/pdf" in content_type
                     or pdf_resp.content[:5] == b"%PDF-")):
            output_path.write_bytes(pdf_resp.content)
            return True

    except Exception as e:
        print(f"    [WARN] Publisher download failed: {e}")

    return False


def _resolve_publisher_pdf_url(
    page_url: str, html: str, doi: str
) -> str | None:
    """Try to resolve PDF URL from publisher page URL and HTML."""
    from urllib.parse import urljoin

    # --- Elsevier / ScienceDirect ---
    if "sciencedirect.com" in page_url:
        pii_match = re.search(r'/pii/(S[0-9X-]+)', page_url)
        if pii_match:
            return f"https://www.sciencedirect.com/science/article/pii/{pii_match.group(1)}/pdfft"

    # --- Springer / Nature ---
    if "springer.com" in page_url or "nature.com" in page_url:
        if "/article/" in page_url:
            return page_url.rstrip("/") + ".pdf"

    # --- Wiley ---
    if "onlinelibrary.wiley.com" in page_url:
        doi_match = re.search(r'/doi/(10\.[^/]+/.+?)(?:/|$)', page_url)
        if doi_match:
            return f"https://onlinelibrary.wiley.com/doi/pdfdirect/{doi_match.group(1)}"

    # --- MDPI ---
    if "mdpi.com" in page_url:
        return page_url.rstrip("/") + "/pdf"

    # --- Frontiers ---
    if "frontiersin.org" in page_url:
        if "/full" in page_url:
            return page_url.replace("/full", "/pdf")

    # --- Taylor & Francis ---
    if "tandfonline.com" in page_url:
        doi_match = re.search(r'/doi/(?:full|abs)/(10\.[^?]+)', page_url)
        if doi_match:
            return f"https://www.tandfonline.com/doi/pdf/{doi_match.group(1)}"

    # --- Oxford Academic ---
    if "academic.oup.com" in page_url:
        if "/article/" in page_url:
            # Try PDF endpoint
            pdf_match = re.search(r'(https?://academic\.oup\.com/[^"]+/article-pdf/[^"]+)', html)
            if pdf_match:
                return pdf_match.group(1)

    # --- ACS Publications ---
    if "pubs.acs.org" in page_url:
        if "/doi/" in page_url:
            return page_url.replace("/doi/abs/", "/doi/pdf/").replace("/doi/full/", "/doi/pdf/")

    # --- Generic: look for PDF link in HTML ---
    pdf_patterns = [
        r'"(https?://[^"]+\.pdf(?:\?[^"]*)?)"',
        r"'(https?://[^']+\.pdf(?:\?[^']*)?)'",
        r'href="([^"]+/pdf[^"]*?)"',
        r'"pdfUrl"\s*:\s*"([^"]+)"',
        r'"downloadUrl"\s*:\s*"([^"]+pdf[^"]*)"',
    ]
    for pattern in pdf_patterns:
        match = re.search(pattern, html)
        if match:
            url = match.group(1)
            if not url.startswith("http"):
                url = urljoin(page_url, url)
            return url

    return None


def _save_as_markdown(article: PubMedArticle, output_path: Path) -> None:
    """Save article metadata and abstract as markdown (fallback)."""
    content = f"""---
pmid: "{article.pmid}"
title: "{article.title}"
authors: {article.authors}
doi: "{article.doi}"
journal: "{article.journal}"
year: "{article.year}"
pmc_id: "{article.pmc_id}"
keywords: {article.keywords}
mesh_terms: {article.mesh_terms}
---

# {article.title}

**Authors**: {', '.join(article.authors)}
**Journal**: {article.journal} ({article.year})
**DOI**: {article.doi}
**PMID**: {article.pmid}

## Abstract

{article.abstract}

## Keywords

{', '.join(article.keywords)}

## MeSH Terms

{', '.join(article.mesh_terms)}
"""
    output_path.write_text(content, encoding="utf-8")


# ============================================================================
# High-Level: Search + Fetch + Download
# ============================================================================

def _find_existing_pmids(output_dir: Path) -> set[str]:
    """Scan output directory for already-downloaded papers by PMID."""
    existing: set[str] = set()
    if not output_dir.exists():
        return existing

    for f in output_dir.iterdir():
        # Files are named PMID{pmid}_... (.pdf or .md)
        match = re.match(r'PMID(\d+)', f.stem)
        if match:
            existing.add(match.group(1))

    return existing

def fetch_papers_by_topic(
    topic: str,
    max_results: int = 10,
    output_dir: str | Path | None = None,
    sort: str = "relevance",
    min_date: str = "",
    max_date: str = "",
    email: str | None = None,
    api_key: str | None = None,
    download_pdf_flag: bool = True,
) -> list[dict[str, Any]]:
    """
    Search PubMed by topic, fetch metadata, and optionally download PDFs.

    Args:
        topic: Search query (e.g., "scRNA-seq germinal center B cell")
        max_results: Number of papers to fetch
        output_dir: Directory to save downloaded papers
        sort: 'relevance' or 'pub_date'
        min_date: Filter by min publication date (YYYY/MM/DD)
        max_date: Filter by max publication date
        download_pdf_flag: Whether to download PDFs

    Returns:
        List of dicts with article metadata and file paths
    """
    if output_dir is None:
        settings = get_settings()
        settings.paths.ensure_dirs()
        output_dir = settings.paths.papers_dir

    output_dir = Path(output_dir)

    # Step 1: Search
    print(f"🔍 Searching PubMed: \"{topic}\"")
    pmids = search_pubmed(
        topic, max_results=max_results, sort=sort,
        min_date=min_date, max_date=max_date,
        email=email, api_key=api_key,
    )

    if not pmids:
        print("  No results found.")
        return []

    # Step 1.5: Filter out already-downloaded PMIDs
    existing_pmids = _find_existing_pmids(output_dir)
    new_pmids = [p for p in pmids if p not in existing_pmids]
    skipped = len(pmids) - len(new_pmids)
    if skipped > 0:
        print(f"  ⏭️  Skipping {skipped} already-downloaded papers")
    if not new_pmids:
        print("  All papers already downloaded.")
        return []

    # Step 2: Fetch metadata
    print(f"📥 Fetching metadata for {len(new_pmids)} new articles...")
    articles = fetch_articles(new_pmids, email=email, api_key=api_key)

    results = []
    for i, article in enumerate(articles, 1):
        print(f"\n  [{i}/{len(articles)}] PMID:{article.pmid} — {article.title[:70]}...")

        entry: dict[str, Any] = {
            "pmid": article.pmid,
            "title": article.title,
            "authors": article.authors,
            "doi": article.doi,
            "journal": article.journal,
            "year": article.year,
            "abstract": article.abstract[:200] + "..." if len(article.abstract) > 200 else article.abstract,
            "pmc_id": article.pmc_id,
            "keywords": article.keywords,
            "mesh_terms": article.mesh_terms,
            "file_path": None,
            "file_type": None,
            "skipped": False,
        }

        # Step 3: Download (with duplicate check at file level)
        if download_pdf_flag:
            file_path = download_pdf(article, output_dir, email)
            if file_path:
                entry["file_path"] = str(file_path)
                entry["file_type"] = file_path.suffix
                print(f"    ✅ Downloaded: {file_path.name}")
            else:
                print(f"    ⚠️  No full text available (abstract only)")
                # Still save abstract as markdown
                safe_title = re.sub(r'[^\w\s-]', '', article.title)[:60].strip()
                md_path = output_dir / f"PMID{article.pmid}_{safe_title.replace(' ', '_')}.md"
                _save_as_markdown(article, md_path)
                entry["file_path"] = str(md_path)
                entry["file_type"] = ".md"

        results.append(entry)
        time.sleep(0.5)  # Rate limiting

    print(f"\n✅ Fetched {len(results)} articles")
    pdf_count = sum(1 for r in results if r.get("file_type") == ".pdf")
    md_count = sum(1 for r in results if r.get("file_type") == ".md")
    print(f"   📄 PDF: {pdf_count}, 📝 Abstract-only: {md_count}")

    return results
