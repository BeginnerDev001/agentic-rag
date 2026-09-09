"""
Unit tests for 10-K filing parser and preprocessing cleaner.
"""

import unittest
from bs4 import BeautifulSoup
from src.ingestion.models import FilingSection
from src.ingestion.parser import SEC10KParser
from src.preprocessing.cleaner import (
    clean_text,
    html_table_to_markdown,
    strip_xbrl_tags,
)


class TestCleaner(unittest.TestCase):
    def test_clean_text_entities_and_quotes(self):
        raw = "Apple&#8217;s revenue was &amp; remains &#8220;strong&#8221;.&#160;&#160;Growth: 10%&#8212;15%."
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "Apple's revenue was & remains \"strong\". Growth: 10% - 15%.")

    def test_clean_text_whitespace(self):
        raw = "Line 1   with   spaces\n\n\n\n\nLine 2\n\n"
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "Line 1 with spaces\n\nLine 2")

    def test_table_to_markdown(self):
        html_table = """
        <table>
            <tr><th>Year</th><th>Revenue</th><th>Change</th></tr>
            <tr><td>2025</td><td>$ 416,161</td><td>6 %</td></tr>
            <tr><td>2024</td><td>$ 391,035</td><td>2 %</td></tr>
        </table>
        """
        soup = BeautifulSoup(html_table, "html.parser")
        table_tag = soup.find("table")
        md = html_table_to_markdown(table_tag)
        self.assertIsNotNone(md)
        self.assertIn("| Year | Revenue | Change |", md)
        self.assertIn("| 2025 | $416,161 | 6 % |", md)
        self.assertIn("| 2024 | $391,035 | 2 % |", md)

    def test_table_to_markdown_empty(self):
        html_table = "<table><tr><td></td><td></td></tr></table>"
        soup = BeautifulSoup(html_table, "html.parser")
        md = html_table_to_markdown(soup.find("table"))
        self.assertIsNone(md)

    def test_strip_xbrl(self):
        html_doc = """
        <div>
            <div style="display:none"><ix:header><ix:hidden>secret</ix:hidden></ix:header></div>
            <p>Net income was <ix:nonfraction>112010</ix:nonfraction> million.</p>
        </div>
        """
        soup = BeautifulSoup(html_doc, "html.parser")
        strip_xbrl_tags(soup)
        txt = clean_text(soup.get_text())
        self.assertNotIn("secret", txt)
        self.assertIn("Net income was 112010 million.", txt)


class TestParser(unittest.TestCase):
    def setUp(self):
        self.parser = SEC10KParser()

    def test_parse_sample_filing(self):
        sample_html = """
        <html><body>
            <div style="text-align:center">ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d)</div>
            <div>Apple Inc. Form 10-K</div>
            <hr style="page-break-after:always"/>
            <div>Item 1. Business</div>
            <div>Apple designs, manufactures and markets smartphones.</div>
            <hr style="page-break-after:always"/>
            <div>Item 1A. Risk Factors</div>
            <div>The Company faces global competition and supply chain risks.</div>
            <hr style="page-break-after:always"/>
            <div>Item 8. Financial Statements and Supplementary Data</div>
            <div>Index to Consolidated Financial Statements</div>
            <hr style="page-break-after:always"/>
            <div>CONSOLIDATED STATEMENTS OF OPERATIONS</div>
            <table>
                <tr><th>Years ended</th><th>2025</th><th>2024</th></tr>
                <tr><td>Total net sales</td><td>$ 416,161</td><td>$ 391,035</td></tr>
            </table>
            <div>Apple Inc. | 2025 Form 10-K | 31</div>
            <hr style="page-break-after:always"/>
            <div>CONSOLIDATED BALANCE SHEETS</div>
            <table>
                <tr><th>As of</th><th>2025</th><th>2024</th></tr>
                <tr><td>Total assets</td><td>$ 359,241</td><td>$ 364,980</td></tr>
            </table>
            <div>Apple Inc. | 2025 Form 10-K | 32</div>
        </body></html>
        """
        meta = {
            "company": "Apple Inc.",
            "ticker": "AAPL",
            "cik": "0000320193",
            "fiscal_year": 2025,
            "filing_date": "2025-10-31",
            "accession_number": "0000320193-25-000079",
        }

        sections = self.parser.parse_filing(sample_html, meta)
        self.assertGreaterEqual(len(sections), 4)

        section_names = [s.section for s in sections]
        self.assertIn("Item 1. Business", section_names)
        self.assertIn("Item 1A. Risk Factors", section_names)
        self.assertIn("Consolidated Statements of Operations", section_names)
        self.assertIn("Consolidated Balance Sheets", section_names)

        # Check Operations table record
        ops_tables = [
            s for s in sections
            if s.section == "Consolidated Statements of Operations" and s.content_type == "table"
        ]
        self.assertEqual(len(ops_tables), 1)
        ops_tbl = ops_tables[0]
        self.assertEqual(ops_tbl.fiscal_year, 2025)
        self.assertEqual(ops_tbl.ticker, "AAPL")
        self.assertEqual(ops_tbl.page, 31)
        self.assertIn("416,161", ops_tbl.text)

        # Check Balance Sheet table record
        bs_tables = [
            s for s in sections
            if s.section == "Consolidated Balance Sheets" and s.content_type == "table"
        ]
        self.assertEqual(len(bs_tables), 1)
        bs_tbl = bs_tables[0]
        self.assertEqual(bs_tbl.page, 32)
        self.assertIn("359,241", bs_tbl.text)

    def test_model_fields(self):
        sec = FilingSection(
            company="Apple",
            filing_type="10-K",
            fiscal_year=2025,
            ticker="AAPL",
            cik="0000320193",
            section="Consolidated Statements of Operations",
            section_id="item_8_operations",
            text="Revenue: 416161",
            page=31,
            content_type="table",
        )
        data = sec.model_dump()
        self.assertEqual(data["company"], "Apple")
        self.assertEqual(data["filing_type"], "10-K")
        self.assertEqual(data["fiscal_year"], 2025)
        self.assertEqual(data["section"], "Consolidated Statements of Operations")
        self.assertEqual(data["page"], 31)
        self.assertEqual(data["text"], "Revenue: 416161")


if __name__ == "__main__":
    unittest.main()
