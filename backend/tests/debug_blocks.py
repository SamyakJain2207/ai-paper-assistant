from pathlib import Path

import pymupdf

PDF_PATH = Path("data/papers/BERT.pdf")


with pymupdf.open(PDF_PATH) as pdf:
    for page_number, page in enumerate(pdf, start=1):
        if page_number > 3:
            break

        print(f"\n{'=' * 70}")
        print(f"PAGE {page_number}")
        print("=" * 70)

        blocks = page.get_text("dict")["blocks"]

        for block_index, block in enumerate(blocks):
            if "lines" not in block:
                continue

            print(f"\nBLOCK {block_index}")
            print(f"BBox: {block['bbox']}")

            for line in block["lines"]:
                for span in line["spans"]:
                    print(
                        f"  TEXT: {span['text']!r}"
                        f" | size={span['size']:.1f}"
                        f" | font={span['font']!r}"
                        f" | flags={span['flags']}"
                    )