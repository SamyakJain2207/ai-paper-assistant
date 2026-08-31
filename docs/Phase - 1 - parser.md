# Phase 1 — Parser Development Notes

## Initial Approach

The first section detector primarily relied on:

- Font size
- Heading length
- Section numbering

## Problems Found During Testing

We tested the parser on:

- Attention Is All You Need
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

The initial approach produced several problems:

1. Paper titles were incorrectly detected as section headings.
2. Smaller numbered subsections were missed.
3. Unnumbered headings/pointers were missed.
4. PDF metadata was incomplete or incorrect.
5. Some headings and paragraph text were combined into the same PDF text block.

## Lessons

Research papers do not follow one universal PDF structure.

Therefore, heading detection should combine multiple signals:

- Section numbering
- Font size
- Font style/boldness
- Text length
- Page position
- Span-level formatting
- First-page/title handling

PDF embedded metadata should not be assumed to be reliable. A fallback based on the actual document content will eventually be required.

## Current Strategy

The parser currently uses:

- An inline run-in heading splitting pass to separate consecutive bold heading spans from paragraph text within a single block.
- A pre-processing pass to merge consecutive blocks of multi-line headings based on page alignment, matching font formatting, unnumbered continuation checks, and small vertical gaps.
- Numbering as the strongest signal for numbered headings (supporting Roman numerals and Appendix prefixes).
- Font size and boldness for unnumbered headings (requiring minimum size of 9.5 and first-span boldness).
- A stack to construct section hierarchy.
- PyMuPDF span information for formatting signals.
- Robust block filters to ignore footnote superscripts, bibliography years, preprint watermarks, and table/figure captions.

This is still a heuristic parser and will be improved through testing against different real-world research papers.

## Diagnostic Methodology & Scratch Scripts

During the debugging of the heuristic parser, several targeted diagnostic scripts were created in `backend/tests/` and the workspace `scratch/` to inspect raw PDF blocks and span-level metadata in detail. 

Although these were temporary diagnostic scripts and were subsequently removed to maintain a clean codebase, the record of their purpose and findings is documented below:

1. **`inspect_pdf_blocks.py`**: A broad-spectrum analysis script that scanned both candidate papers and outputted bounding boxes, font sizes, font families, and bold flags for any text block starting with numbers or matching heading patterns.
2. **`inspect_selected_blocks.py`**: An inspection script that target-extracted properties for all known correct headings and known false positives. This helped establish exact contrasts in line count, word count, and font sizing.
3. **`inspect_specific_positives_and_negatives.py`**: A refined version of the selected blocks analyzer to focus only on highly specific problem blocks (preventing console buffer truncation).
4. **`inspect_arxiv_header.py`**: Target-inspected the `arXiv:1810.04805v2` block. It revealed that PyMuPDF extracted it with a non-bold font size of `20.0`, which had bypassed the simple `font_size >= 12.0` unnumbered heading heuristic.
5. **`inspect_squad_ner.py`**: Target-inspected bold table label blocks like `SQuAD` and `NER MNLI` under the *Related Work* section, revealing they were extracted as separate blocks with a size of `6.94`, allowing them to bypass the bold length checks.
6. **`inspect_attention_headings.py`**: Mapped out the font size hierarchy in the Attention paper, confirming that level-1 headings are `11.95` (bold) and levels 2 & 3 are `9.96` (bold).

### Key Insights Derived
- **Font Size Constraints**: Heading text blocks styled in bold must have a font size of at least `9.5` to separate them from small bold table cells or chart labels.
- **Watermark/Preprint Exclusions**: Non-bold metadata or preprint blocks that have large extracted font sizes (due to PDF layering tricks) must be explicitly filtered out using keywords (e.g. `arXiv:`, `doi:`).
- **Footnote / Superscript Ratios**: Blocks starting with a number should only be considered section headings if the numbering span's font size is relatively equal to the rest of the block (ratio $\ge 0.90$). Superscript footnotes usually have a ratio of $\le 0.70$.
- **Bibliography Year Filters**: Section numbering should exclude values $\ge 50$ to avoid false positive matches on bibliography years (e.g. `2016.` or `2003.`).

## Generalized Metadata Extraction & Title Similarity

To support research papers downloaded from any source (not just arXiv), the metadata extraction pipeline uses a multi-layered local-first and remote-enrichment approach:

- **Local DOI Extraction**: Scans the first two pages of the PDF text for standard publisher DOIs matching the regex pattern `10.\d{4,9}/[-._;()/:a-zA-Z0-9]+` (standard for IEEE, ACM, Springer, NeurIPS, ACL, and other journals).
- **Jaccard Title Similarity Filtering**: Calculates Jaccard similarity (Intersection over Union of words) between our local fallback title and the API search results to prevent false matches (e.g. accepting correct matches while rejecting partial title clashes). Jaccard threshold is set to $\ge 0.70$.
- **Local Fallbacks**: Extracts Page 1 titles, parses bold name spans for authors, and looks for preprint watermarks/conference years.
- **Enrichment Layers**: 
  1. Exact DOI query on Crossref API
  2. ArXiv ID query on arXiv API
  3. Title search on arXiv API with Jaccard filter
  4. Title search on Crossref API with Jaccard filter

## Additional Diagnostic Scripts (Turn 2)

The following temporary scripts were used to develop these features:

- **`inspect_missed_headings.py` / `inspect_all_bold.py`**: Investigated why unnumbered headings inside paragraphs (like "Model Architecture") were merged into text blocks, leading to the creation of the inline run-in heading splitter.
- **`search_doi_dates.py` / `search_doi_text.py`**: Scanned text for DOI and date identifiers, confirming that arXiv preprints do not print DOIs in their body text.
- **`test_arxiv_api.py` / `test_arxiv_api_https.py`**: Tested official arXiv metadata lookup, demonstrating the need for HTTPS.
- **`test_similarity.py` / `test_jaccard.py`**: Evaluated overlap coefficient vs. Jaccard similarity, confirming Jaccard is far more precise for title lookups.